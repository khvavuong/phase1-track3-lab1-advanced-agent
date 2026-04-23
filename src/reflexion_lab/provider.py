from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResponse:
    text: str
    token_estimate: int
    latency_ms: int


class QwenHFProvider:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-1.5B",
        cache_dir: str | None = None,
        use_4bit_if_available: bool = True,
        local_files_only: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "Missing local inference dependencies. Install: pip install torch transformers accelerate"
            ) from exc

        self._torch = torch
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.local_files_only = local_files_only

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )

        model_kwargs: dict[str, Any] = {
            "cache_dir": cache_dir,
            "trust_remote_code": True,
            "local_files_only": local_files_only,
        }

        if self.device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = {"": "cpu"}

        if self.device == "cuda" and use_4bit_if_available:
            try:
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            except Exception:
                pass

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 96,
        temperature: float = 0.0,
        top_p: float = 0.9,
        do_sample: bool = False,
    ) -> LLMResponse:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        started = time.perf_counter()
        with self._torch.no_grad():
            output_ids = self.model.generate(**inputs, **generation_kwargs)
        latency_ms = int((time.perf_counter() - started) * 1000)

        prompt_len = int(inputs["input_ids"].shape[1])
        full_ids = output_ids[0].tolist()
        new_token_ids = full_ids[prompt_len:]
        text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
        token_estimate = len(full_ids)
        return LLMResponse(text=text, token_estimate=token_estimate, latency_ms=latency_ms)
