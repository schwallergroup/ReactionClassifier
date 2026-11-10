from __future__ import annotations

"""
Lightweight utilities to extract the reaction hierarchy from the prompt file and
derive a two-level (X.Y) -> descendants mapping.

This module intentionally has no external dependencies beyond the standard library.

Exposed API (used elsewhere in the codebase):
 - default_input_path() -> Path
 - extract_hierarchy_text(path: Path) -> str
 - parse_hierarchy_lines(text: str) -> list[tuple[str, str]]
 - build_xx_mapping(items: list[tuple[str, str]]) -> dict[str, list[str]]
"""

from pathlib import Path
import re
from typing import List, Tuple, Dict


def default_input_path() -> Path:
    """Return the path to the prompt file containing the hierarchy text.

    We extract the hierarchy from COHORT_SYSTEM_PROMPT in
    rxn_classifier/prompts/hierarchy_classification.py
    """
    # This file is: .../llm/classification/generate_xx_mapping.py
    # Prompt lives at: .../llm/prompts/hierarchy_classification.py
    return Path(__file__).resolve().parents[1] / "prompts" / "hierarchy_classification.py"


def _extract_system_prompt(text: str) -> str:
    """Extract the content of SYSTEM_PROMPT triple-quoted string from a Python file."""
    pat = re.compile(
        r"SYSTEM_PROMPT\s*=\s*(?P<q>\"\"\"|''')(?:\r?\n)?(?P<body>.*?)(?P=q)",
        re.DOTALL,
    )
    m = pat.search(text)
    if not m:
        return text
    return m.group("body")


def extract_hierarchy_text(path: Path) -> str:
    """Extract the 'Reaction Hierarchy' section from the hierarchy prompt file.

    The implementation is robust to small changes: it looks inside SYSTEM_PROMPT and
    selects only lines that look like numbered hierarchy entries if section
    boundaries are not obvious.
    """
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    body = _extract_system_prompt(raw)

    # Try to slice between 'Reaction Hierarchy' and the next major header
    low = body.lower()
    start = low.find("reaction hierarchy")
    if start >= 0:
        tail = body[start:]
        # Find earliest occurrence of any known section that follows the hierarchy
        next_markers = [
            "analysis instructions",
            "output format",
            "reactions to analyze",
            "cohort reactions",
        ]
        ends = [tail.lower().find(m) for m in next_markers]
        ends = [e for e in ends if e > 0]
        if ends:
            end = min(ends)
            return tail[:end].strip()
        return tail.strip()

    # Fallback: return only lines that look like hierarchy entries
    lines = []
    for ln in body.splitlines():
        line = ln.rstrip()
        if re.match(r"^\s*\d+(?:\.\d+)+\s+\S", line):
            lines.append(line.strip())
    return "\n".join(lines)


def parse_hierarchy_lines(text: str) -> List[Tuple[str, str]]:
    """Parse lines like '2.1 N-acylation to amide' into (code, label)."""
    out: List[Tuple[str, str]] = []
    pat = re.compile(r"^\s*(\d+(?:\.\d+)+)\s+(.*\S)\s*$")
    for raw in (text or "").splitlines():
        ln = raw.strip()
        if not ln:
            continue
        m = pat.match(ln)
        if not m:
            continue
        code, label = m.group(1), m.group(2)
        out.append((code, label))
    return out


def build_xx_mapping(items: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    """Build a mapping X.Y -> list of 'code label' descendants.

    - Keys are two-level codes (e.g., '2.1').
    - Values are lines like '2.1.1 Label', '2.1.1.1 Label', etc.
    Two-level entries themselves are not included in the descendant lists.
    """
    mapping: Dict[str, List[str]] = {}
    for code, label in items:
        parts = code.split(".")
        if len(parts) < 2:
            # Skip malformed
            continue
        two = ".".join(parts[:2])
        # Ensure key exists
        mapping.setdefault(two, [])
        # Only add deeper descendants (3+ parts)
        if len(parts) >= 3:
            line = f"{code} {label}".strip()
            if line not in mapping[two]:
                mapping[two].append(line)
    # Stable sort by code prefix
    for k, v in list(mapping.items()):
        try:
            mapping[k] = sorted(v, key=lambda s: s.split()[0])
        except Exception:
            pass
    return mapping


# Optional: simple CLI to print mapping (useful for debugging)
if __name__ == "__main__":
    p = default_input_path()
    txt = extract_hierarchy_text(p)
    pairs = parse_hierarchy_lines(txt)
    mp = build_xx_mapping(pairs)
    import json
    print(json.dumps(mp, indent=2))

