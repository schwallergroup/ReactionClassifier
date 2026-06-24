"""Access to the LLM-derived reaction taxonomy and the granularity examples."""
from __future__ import annotations

import functools
import json
from importlib import resources
from typing import Dict, List, Optional


def _data(name: str):
    return resources.files("reactionclassifier.data").joinpath(name)


@functools.lru_cache(maxsize=1)
def load_taxonomy() -> Dict[str, str]:
    """Full hierarchy: flat mapping of class code (e.g. ``"1.3.6.2"``) -> name."""
    return json.loads(_data("taxonomy.json").read_text())


def name_for(code: Optional[str]) -> Optional[str]:
    if code is None:
        return None
    return load_taxonomy().get(str(code))


def tier_path(code: Optional[str]) -> List[str]:
    """Ancestor codes present in the taxonomy, from tier 2 down to ``code``."""
    if code is None:
        return []
    parts = str(code).split(".")
    tax = load_taxonomy()
    return [".".join(parts[:n]) for n in range(2, len(parts) + 1) if ".".join(parts[:n]) in tax]


@functools.lru_cache(maxsize=1)
def load_granularity() -> Dict[str, dict]:
    """The two granularity-comparison tables (general + med-chem). These also
    carry the small illustrative subset of generalised SMIRKS that is released."""
    out = {}
    for nm in ("granularity_examples.json", "granularity_examples_medchem.json"):
        out[nm.replace(".json", "")] = json.loads(_data(nm).read_text())
    return out
