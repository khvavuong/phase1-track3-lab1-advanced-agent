from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResponse:
    text: str
    token_estimate: int
    latency_ms: int


class OpenAIProvider:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("Missing OpenAI dependency. Install: pip install openai") from exc

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY environment variable.")

        self._client = OpenAI(api_key=api_key)
        self.model = model

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
    ) -> LLMResponse:
        del do_sample  # Sampling behavior is controlled by temperature/top_p in OpenAI models.
        started = time.perf_counter()
        if self.model.startswith("gpt-3.5"):
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            latency_ms = int((time.perf_counter() - started) * 1000)
            text = (response.choices[0].message.content or "").strip()
            token_estimate = int(getattr(response.usage, "total_tokens", 0) or 0)
        else:
            response = self._client.responses.create(
                model=self.model,
                input=prompt,
                max_output_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            latency_ms = int((time.perf_counter() - started) * 1000)
            text = self._extract_text(response).strip()
            token_estimate = int(getattr(getattr(response, "usage", None), "total_tokens", 0) or 0)

        if token_estimate <= 0:
            token_estimate = max(1, int((len(prompt) + len(text)) / 4))
        return LLMResponse(text=text, token_estimate=token_estimate, latency_ms=latency_ms)

    @staticmethod
    def _extract_text(response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        parts: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if isinstance(text, str) and text:
                    parts.append(text)
        if parts:
            return "\n".join(parts)
        return ""
