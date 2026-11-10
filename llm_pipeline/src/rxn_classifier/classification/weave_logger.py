from __future__ import annotations

import json
from typing import Any, Dict

try:
    import weave  # type: ignore
except Exception:  # pragma: no cover - weave is optional
    weave = None  # type: ignore


def _safe_jsonable(obj: Any) -> Any:
    try:
        return json.loads(json.dumps(obj, default=lambda o: getattr(o, "__dict__", str(o))))
    except Exception:
        try:
            return str(obj)
        except Exception:
            return None


def log_single_call(
    agent: str,
    model: str,
    temperature: float | None,
    thinking_budget: int | None,
    system_prompt: str,
    user_content: str,
    response_text: str,
    raw_response: Any,
) -> Dict[str, Any]:
    """Record one LLM call. Traced by Weave when Weave is installed and initialised."""
    return {
        "agent": agent,
        "model": model,
        "temperature": temperature,
        "thinking_budget": thinking_budget,
        "system_prompt": system_prompt,
        "user_content": user_content,
        "response_text": response_text,
        "raw_response": _safe_jsonable(raw_response),
    }


if weave is not None:
    log_single_call = weave.op()(log_single_call)
