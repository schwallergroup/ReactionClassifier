from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from rxn_classifier.classification.llm_client import BaseLLMClient
from rxn_classifier.classification.models import ClassProposal
from rxn_classifier.classification.utils import build_hierarchy_subtree, format_rxn_roles_with_strip
from rxn_classifier.prompts.aggregate_proposals import system_prompt as AGG_SYS


_NH_RE = re.compile(r"<new_hierarchy>\s*(.*?)\s*</new_hierarchy>", re.DOTALL | re.IGNORECASE)
_ASSIGN_RE = re.compile(r"<assignments>\s*(\{.*?\})\s*</assignments>", re.DOTALL | re.IGNORECASE)


def _extract(text: str, pat: re.Pattern) -> str:
    m = pat.search(text or "")
    return m.group(1).strip() if m else ""


def _parse_new_lines(text: str) -> List[str]:
    body = _extract(text, _NH_RE)
    return [line.strip() for line in body.splitlines() if line.strip()]


def _parse_assignments(text: str) -> Dict[str, str]:
    blob = _extract(text, _ASSIGN_RE)
    if not blob:
        return {}
    try:
        data = json.loads(blob)
        return {str(k): str(v) for k, v in data.items()}
    except Exception:
        return {}


def _build_hierarchy_context(parent_codes: List[str], mapping: Optional[dict]) -> str:
    """Subtrees of the distinct two-level parents, in first-seen order."""
    seen: List[str] = []
    for p in parent_codes:
        base = (p or "").strip()
        if not base:
            continue
        base2 = ".".join(base.split(".")[:2]) if "." in base else base
        if base2 not in seen:
            seen.append(base2)
    return "\n\n".join(build_hierarchy_subtree(code, mapping) for code in seen)


@dataclass
class AggregatorAgent:
    llm: BaseLLMClient
    strip_mapping: bool = False
    # The run's current mapping (X.Y -> deeper hierarchy lines); set by the pipeline.
    mapping_override: Optional[Dict[str, List[str]]] = None

    def aggregate(self, proposals: List[ClassProposal], indices: List[int], parent_two_codes: List[str]) -> Tuple[List[str], Dict[str, str], str]:
        """Merge proposals into new hierarchy lines and assign each a final code.

        Returns (new_hierarchy_lines, {index: assigned_code}, raw_response_text).
        """
        payload: List[Dict[str, str]] = []
        for idx, prop in zip(indices, proposals):
            payload.append(
                {
                    "reaction_smiles": format_rxn_roles_with_strip(prop.reaction_smiles, strip_mapping=self.strip_mapping),
                    "index": str(idx),
                    "suggested_parent_code": prop.suggested_parent_code or "",
                    "new_entry_code": prop.new_entry_code or "",
                    "new_entry_name": prop.new_entry_name or "",
                    "hierarchy_line": prop.hierarchy_line or "",
                    "rationale": prop.rationale or "",
                }
            )

        proposals_json = json.dumps(payload, indent=2)
        hierarchy_text = _build_hierarchy_context(parent_two_codes, self.mapping_override)
        sys_prompt = (
            AGG_SYS.replace("{{proposals}}", proposals_json)
            .replace("{{hierarchy}}", hierarchy_text)
        )
        resp = self.llm.generate(system_prompt=sys_prompt, user_content=" ")
        txt = resp.text or ""
        return _parse_new_lines(txt), _parse_assignments(txt), txt
