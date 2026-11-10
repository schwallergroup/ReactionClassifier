from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

from rxn_classifier.classification.models import PipelineConfig
from rxn_classifier.classification.pipeline import ClassificationPipeline


def _load_config(path: Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "YAML config provided but PyYAML is not installed. Install pyyaml or use JSON."
            ) from e
        return yaml.safe_load(text) or {}
    # default to JSON
    return json.loads(text)

class _TemplateVars(dict):
    """dict that leaves unknown {placeholders} unchanged."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _expand_templates(obj: Any, vars: Mapping[str, Any]) -> Any:
    """Recursively expand {placeholders} in all string values."""
    if isinstance(obj, str):
        try:
            return obj.format_map(_TemplateVars(vars))  # type: ignore[arg-type]
        except Exception:
            return obj
    if isinstance(obj, list):
        return [_expand_templates(x, vars) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_expand_templates(x, vars) for x in obj)
    if isinstance(obj, dict):
        return {k: _expand_templates(v, vars) for k, v in obj.items()}
    return obj


def _apply_config_templates(data: Dict[str, Any]) -> Dict[str, Any]:
    """Apply templating using start/end/max rows derived from the config."""
    start_row = data.get("start_row")
    end_row = data.get("end_row")
    max_rows = data.get("max_rows")
    try:
        start_int = int(start_row) if start_row is not None else 0
    except Exception:
        start_int = 0
    try:
        end_int = int(end_row) if end_row is not None else None
    except Exception:
        end_int = None
    try:
        max_int = int(max_rows) if max_rows is not None else None
    except Exception:
        max_int = None

    derived_n = None
    if max_int is not None:
        derived_n = max_int
    elif end_int is not None:
        derived_n = max(0, end_int - start_int)

    vars: Dict[str, Any] = {
        "start_row": start_int,
        "start_idx": start_int,
        "end_row": end_int if end_int is not None else "",
        "end_idx": end_int if end_int is not None else "",
        "max_rows": max_int if max_int is not None else "",
        "n_rows": derived_n if derived_n is not None else "",
        "prev_end": start_int,
        "prev_end_idx": start_int,
    }
    return _expand_templates(data, vars)


def run_from_config(config_path: str):
    data = _load_config(Path(config_path))
    data = _apply_config_templates(data)
    cfg = PipelineConfig(**data)
    # If mapping_json points at a per-run output that doesn't exist yet, fall back.
    try:
        if cfg.mapping_json and not cfg.mapping_json.exists():
            if cfg.mapping_json_fallback and cfg.mapping_json_fallback.exists():
                print(
                    f"[Config] mapping_json not found ({cfg.mapping_json}); "
                    f"falling back to {cfg.mapping_json_fallback}"
                )
                cfg = cfg.model_copy(update={"mapping_json": cfg.mapping_json_fallback})
    except Exception:
        pass
    # A missing mapping would otherwise start the run from an empty taxonomy without a word.
    if cfg.mapping_json and not cfg.mapping_json.exists():
        raise FileNotFoundError(f"mapping_json does not exist: {cfg.mapping_json}")
    # Trace LLM calls with Weave when a project is configured
    if cfg.weave_project:
        try:
            import weave  # type: ignore
            weave.init(cfg.weave_project)
        except Exception as e:
            print(f"[Weave] tracing disabled: {e}")
    # Best-effort diagnostics for LLM availability
    try:
        import os
        key_present = bool(os.getenv("GEMINI_API_KEY"))
        print(f"[LLM diag] GEMINI_API_KEY present: {key_present}")
        try:
            from google import genai  # type: ignore
            print("[LLM diag] google.genai import: OK")
        except Exception as _e:
            print(f"[LLM diag] google.genai import failed: {_e}")
    except Exception:
        pass
    pipeline = ClassificationPipeline(config=cfg)
    return pipeline.run()
