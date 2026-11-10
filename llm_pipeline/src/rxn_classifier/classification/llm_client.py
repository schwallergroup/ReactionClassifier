from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Any, Dict, Tuple
import json
import time as _time


@dataclass
class LLMResponse:
    text: str


class BaseLLMClient:
    def __init__(
        self,
        model: str,
        temperature: Optional[float] = 0.1,
        thinking_budget: Optional[int] = None,
        log_io: bool = False,
        save_io_dir: Optional[Path] = None,
        client_tag: str = "",
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.thinking_budget = thinking_budget
        self.log_io = bool(log_io)
        self.save_io_dir = Path(save_io_dir) if save_io_dir else None
        self.client_tag = client_tag or ""

    def _serialize_response(self, resp: Any) -> Any:
        # Try successive strategies to serialize SDK response objects
        for attr in ("to_dict", "model_dump", "dict"):
            fn = getattr(resp, attr, None)
            if callable(fn):
                try:
                    return fn()
                except Exception:
                    pass
        try:
            return json.loads(json.dumps(resp, default=lambda o: getattr(o, "__dict__", str(o))))
        except Exception:
            try:
                return str(resp)
            except Exception:
                return None

    def _maybe_save_io(self, kind: str, payload: Dict[str, Any]) -> None:
        if not self.save_io_dir:
            return
        try:
            self.save_io_dir.mkdir(parents=True, exist_ok=True)
            ts = int(_time.time() * 1000)
            fname = f"{ts}_{kind}_{self.model.replace('/', '_')}.json"
            (self.save_io_dir / fname).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

    def generate(self, system_prompt: str, user_content: str) -> LLMResponse:
        raise NotImplementedError

    def generate_batch(self, system_prompt: str, user_contents: List[str]) -> List[LLMResponse]:
        # Default fallback: call generate repeatedly
        return [self.generate(system_prompt, uc) for uc in user_contents]

    def generate_batch_multi(self, system_prompts: List[str], user_contents: List[str]) -> List[LLMResponse]:
        """Batch with per-request system prompts.

        Lengths of system_prompts and user_contents must match.
        Default fallback: loop over generate().
        """
        if len(system_prompts) != len(user_contents):
            raise ValueError("system_prompts and user_contents must be same length")
        return [self.generate(sp, uc) for sp, uc in zip(system_prompts, user_contents)]


class GoogleGenAIClient(BaseLLMClient):
    """Thin wrapper around google.genai client with the minimal API we need.

    This avoids depending on project-specific wrappers and external decorators.
    Supports Vertex AI (default) with parallel individual calls for batching.
    """

    def __init__(
        self,
        model: str,
        temperature: Optional[float] = 0.1,
        thinking_budget: Optional[int] = None,
        thinking_level: Optional[str] = None,
        log_io: bool = False,
        save_io_dir: Optional[Path] = None,
        client_tag: str = "",
        use_vertex_ai: bool = False,
        vertex_project: Optional[str] = None,
        vertex_location: str = "global",
    ) -> None:
        # Strip "models/" prefix if present - Vertex AI expects just the model name
        if model.startswith("models/"):
            model = model[7:]  # Remove "models/" prefix

        super().__init__(
            model=model,
            temperature=temperature,
            thinking_budget=thinking_budget,
            log_io=log_io,
            save_io_dir=save_io_dir,
        )
        # Lazy import to avoid hard dependency when unused
        from google import genai  # type: ignore

        self._genai = genai
        self.client_tag = client_tag
        self.thinking_level = thinking_level

        self._use_vertex_ai = use_vertex_ai
        if use_vertex_ai:
            # Use Vertex AI authentication (requires gcloud auth application-default login)
            # Note: Vertex AI doesn't support inline batch API, so we use parallel individual calls
            self._client = genai.Client(
                vertexai=True,
                project=vertex_project,
                location=vertex_location,
            )
            print(f"[GoogleGenAIClient] Initialized with Vertex AI (project={vertex_project}, location={vertex_location})")
        else:
            # Fall back to API key authentication
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY is not set in environment and use_vertex_ai=False.")
            # Pin the beta API: newer google-genai releases default to v1, which does
            # not serve preview models.
            self._client = genai.Client(
                api_key=api_key, http_options=genai.types.HttpOptions(api_version="v1beta")
            )
            print(f"[GoogleGenAIClient] Initialized with API key")

    def _make_config(self, system_prompt: str):
        cfg = {"system_instruction": system_prompt}
        if self.temperature is not None:
            cfg["temperature"] = self.temperature

        # Thinking configuration
        try:
            if self.thinking_level:
                # Explicit thinking_level set by caller
                cfg["thinking_config"] = self._genai.types.ThinkingConfig(
                    thinking_level=self.thinking_level
                )
            elif self.thinking_budget is not None:
                # Use thinking_budget (supported by Gemini 2.5+ and gemini-3-pro-preview)
                cfg["thinking_config"] = self._genai.types.ThinkingConfig(
                    thinking_budget=self.thinking_budget
                )
        except Exception:
            pass
        return self._genai.types.GenerateContentConfig(**cfg)

    def generate(self, system_prompt: str, user_content: str) -> LLMResponse:
        if self.log_io:
            try:
                print(f"\n[LLM:{self.model}] generate() CALL")
                print("[SYSTEM_PROMPT]\n" + (system_prompt or ""))
                print("[USER_CONTENT]\n" + (user_content or ""))
            except Exception:
                pass
        cfg = self._make_config(system_prompt)
        resp = self._client.models.generate_content(
            model=self.model,
            contents=user_content,
            config=cfg,
        )
        # The SDK returns an object with .text convenience
        text = getattr(resp, "text", "") or ""
        # Save IO if requested
        self._maybe_save_io(
            "single",
            {
                "model": self.model,
                "temperature": self.temperature,
                "thinking_budget": self.thinking_budget,
                "system_prompt": system_prompt,
                "user_content": user_content,
                "response_text": text,
                "raw_response": self._serialize_response(resp),
            },
        )
        # Weave logging (must be called for every invocation)
        try:
            from .weave_logger import log_single_call
            log_single_call(
                agent=self.client_tag or "",
                model=self.model,
                temperature=self.temperature,
                thinking_budget=self.thinking_budget,
                system_prompt=system_prompt,
                user_content=user_content,
                response_text=text,
                raw_response=resp,
            )
        except Exception:
            pass
        if self.log_io:
            try:
                print("[LLM_RESPONSE]\n" + text + "\n[END]")
            except Exception:
                pass
        return LLMResponse(text=text)

    def generate_batch(self, system_prompt: str, user_contents: List[str]) -> List[LLMResponse]:
        """Batch with single shared system prompt. Delegates to generate_batch_multi."""
        # Delegate to generate_batch_multi with repeated system prompts
        # This ensures consistent size-aware batching behavior
        system_prompts = [system_prompt] * len(user_contents)
        return self.generate_batch_multi(system_prompts, user_contents)

    def generate_batch_multi(self, system_prompts: List[str], user_contents: List[str]) -> List[LLMResponse]:
        """Batch generation using parallel individual calls.

        Vertex AI doesn't support inline batch requests, so we use ThreadPoolExecutor
        to run multiple generate() calls in parallel for efficiency.
        """
        if len(system_prompts) != len(user_contents):
            raise ValueError("system_prompts and user_contents must be same length")

        n = len(user_contents)
        print(f"[LLM:{self.model}] generate_batch_multi() running {n} calls in parallel...")

        results: List[LLMResponse] = [LLMResponse(text="") for _ in range(n)]

        def _call_single(idx: int) -> Tuple[int, LLMResponse]:
            try:
                resp = self.generate(system_prompts[idx], user_contents[idx])
                return idx, resp
            except Exception as e:
                print(f"[LLM:{self.model}] Call {idx} failed: {e}")
                return idx, LLMResponse(text="")

        # Run in parallel with reasonable concurrency
        max_workers = min(n, 8)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_call_single, i) for i in range(n)]
            completed = 0
            for future in as_completed(futures):
                idx, resp = future.result()
                results[idx] = resp
                completed += 1
                if completed % 50 == 0 or completed == n:
                    print(f"[LLM:{self.model}] Completed {completed}/{n} calls")
        # Save IO if requested
        self._maybe_save_io(
            "batch_multi",
            {
                "model": self.model,
                "temperature": self.temperature,
                "thinking_budget": self.thinking_budget,
                "n_calls": n,
                "responses": [r.text for r in results],
            },
        )

        return results


class NoOpLLMClient(BaseLLMClient):
    """Offline stub. Returns empty strings to keep the pipeline wiring testable."""

    def generate(self, system_prompt: str, user_content: str) -> LLMResponse:
        try:
            print(f"[NoOpLLMClient:{self.client_tag}] generate() returning empty string. Configure GEMINI_API_KEY and google-genai to enable real calls.")
        except Exception:
            pass
        return LLMResponse(text="")

    def generate_batch(self, system_prompt: str, user_contents: List[str]) -> List[LLMResponse]:
        try:
            print(f"[NoOpLLMClient:{self.client_tag}] generate_batch() returning {len(user_contents)} empty responses.")
        except Exception:
            pass
        return [LLMResponse(text="") for _ in user_contents]

    def generate_batch_multi(self, system_prompts: List[str], user_contents: List[str]) -> List[LLMResponse]:
        if len(system_prompts) != len(user_contents):
            raise ValueError("system_prompts and user_contents must be same length")
        try:
            print(f"[NoOpLLMClient:{self.client_tag}] generate_batch_multi() returning {len(user_contents)} empty responses.")
        except Exception:
            pass
        return [LLMResponse(text="") for _ in user_contents]
