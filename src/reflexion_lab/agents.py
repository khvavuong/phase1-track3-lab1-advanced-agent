from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .provider import OpenAIProvider
from .schemas import AttemptTrace, JudgeResult, QAExample, ReflectionEntry, RunRecord
from .utils import _context_to_text, _extract_json_object, _safe_int_score, normalize_answer


@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    provider: OpenAIProvider = field(default_factory=OpenAIProvider)

    def _actor_prompt(self, example: QAExample, reflection_memory: list[str]) -> str:
        reflection_block = "\n".join(f"- {item}" for item in reflection_memory) if reflection_memory else "- (none)"
        return (
            f"{ACTOR_SYSTEM.strip()}\n\n"
            f"Question:\n{example.question}\n\n"
            f"Context:\n{_context_to_text(example)}\n\n"
            f"Reflection memory:\n{reflection_block}\n\n"
            "Return final answer only."
        )

    def _evaluator_prompt(self, example: QAExample, answer: str) -> str:
        return (
            f"{EVALUATOR_SYSTEM.strip()}\n\n"
            f"Question:\n{example.question}\n\n"
            f"Gold answer:\n{example.gold_answer}\n\n"
            f"Candidate answer:\n{answer}\n\n"
            f"Context:\n{_context_to_text(example)}"
        )

    def _reflector_prompt(self, example: QAExample, answer: str, judge: JudgeResult, attempt_id: int) -> str:
        return (
            f"{REFLECTOR_SYSTEM.strip()}\n\n"
            f"Attempt ID: {attempt_id}\n"
            f"Question: {example.question}\n"
            f"Answer: {answer}\n"
            f"Evaluator reason: {judge.reason}\n"
            f"Missing evidence: {judge.missing_evidence}\n"
            f"Spurious claims: {judge.spurious_claims}\n"
            f"Context:\n{_context_to_text(example)}"
        )

    def _call_actor(self, example: QAExample, reflection_memory: list[str]) -> tuple[str, int, int]:
        resp = self.provider.generate(
            prompt=self._actor_prompt(example, reflection_memory),
            max_new_tokens=80,
            do_sample=False,
        )
        return resp.text.strip(), resp.token_estimate, resp.latency_ms

    def _call_evaluator(self, example: QAExample, answer: str) -> tuple[JudgeResult, int, int]:
        resp = self.provider.generate(
            prompt=self._evaluator_prompt(example, answer),
            max_new_tokens=180,
            do_sample=False,
        )
        try:
            payload = _extract_json_object(resp.text)
            judge = JudgeResult(
                score=_safe_int_score(payload.get("score", 0)),
                reason=str(payload.get("reason", "")).strip() or "No reason returned.",
                missing_evidence=[str(x) for x in payload.get("missing_evidence", []) if str(x).strip()],
                spurious_claims=[str(x) for x in payload.get("spurious_claims", []) if str(x).strip()],
            )
        except Exception:
            exact = normalize_answer(answer) == normalize_answer(example.gold_answer)
            judge = JudgeResult(
                score=1 if exact else 0,
                reason="Evaluator JSON parse failed; used string-match fallback.",
                missing_evidence=[] if exact else ["Evaluator output was not valid JSON."],
                spurious_claims=[] if exact else [answer],
            )
        return judge, resp.token_estimate, resp.latency_ms

    def _call_reflector(
        self, example: QAExample, answer: str, judge: JudgeResult, attempt_id: int
    ) -> tuple[ReflectionEntry, int, int]:
        resp = self.provider.generate(
            prompt=self._reflector_prompt(example, answer, judge, attempt_id),
            max_new_tokens=140,
            do_sample=False,
        )
        try:
            payload = _extract_json_object(resp.text)
            reflection = ReflectionEntry(
                attempt_id=attempt_id,
                failure_reason=judge.reason,
                lesson=str(payload.get("lesson", "")).strip() or "Need to improve evidence grounding.",
                next_strategy=str(payload.get("next_strategy", "")).strip()
                or "Use both context chunks and verify final entity before answering.",
            )
        except Exception:
            reflection = ReflectionEntry(
                attempt_id=attempt_id,
                failure_reason=judge.reason,
                lesson="Previous attempt did not satisfy evaluator checks.",
                next_strategy="Do explicit two-hop reasoning and ensure final answer matches context.",
            )
        return reflection, resp.token_estimate, resp.latency_ms

    def _infer_failure_mode(self, judge: JudgeResult, answer: str) -> str:
        if judge.score == 1:
            return "none"
        reason = judge.reason.lower()
        if "loop" in reason:
            return "looping"
        if "drift" in reason:
            return "entity_drift"
        if "first hop" in reason or "incomplete" in reason or "missing evidence" in reason:
            return "incomplete_multi_hop"
        if "overfit" in reason:
            return "reflection_overfit"
        if answer.strip() == "":
            return "incomplete_multi_hop"
        return "wrong_final_answer"

    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_judge = JudgeResult(score=0, reason="No attempts executed.")

        for attempt_id in range(1, self.max_attempts + 1):
            answer, actor_tokens, actor_latency = self._call_actor(example, reflection_memory)
            judge, eval_tokens, eval_latency = self._call_evaluator(example, answer)
            token_estimate = actor_tokens + eval_tokens
            latency_ms = actor_latency + eval_latency

            reflection_entry: ReflectionEntry | None = None
            if self.agent_type == "reflexion" and judge.score == 0 and attempt_id < self.max_attempts:
                reflection_entry, ref_tokens, ref_latency = self._call_reflector(
                    example, answer, judge, attempt_id
                )
                reflections.append(reflection_entry)
                reflection_memory.append(reflection_entry.next_strategy)
                token_estimate += ref_tokens
                latency_ms += ref_latency

            trace = AttemptTrace(
                attempt_id=attempt_id,
                answer=answer,
                score=judge.score,
                reason=judge.reason,
                reflection=reflection_entry,
                token_estimate=token_estimate,
                latency_ms=latency_ms,
            )
            traces.append(trace)
            final_answer = answer
            final_judge = judge
            if judge.score == 1:
                break

        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        failure_mode = self._infer_failure_mode(final_judge, final_answer)
        return RunRecord(
            qid=example.qid,
            question=example.question,
            gold_answer=example.gold_answer,
            agent_type=self.agent_type,
            predicted_answer=final_answer,
            is_correct=bool(final_judge.score),
            attempts=len(traces),
            token_estimate=total_tokens,
            latency_ms=total_latency,
            failure_mode=failure_mode,
            reflections=reflections,
            traces=traces,
        )


class ReActAgent(BaseAgent):
    def __init__(self, provider: OpenAIProvider | None = None) -> None:
        super().__init__(agent_type="react", max_attempts=1, provider=provider or OpenAIProvider())


class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3, provider: OpenAIProvider | None = None) -> None:
        super().__init__(
            agent_type="reflexion",
            max_attempts=max_attempts,
            provider=provider or OpenAIProvider(),
        )
