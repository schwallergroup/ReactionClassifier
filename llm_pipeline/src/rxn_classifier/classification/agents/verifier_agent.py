from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from rxn_classifier.classification.llm_client import BaseLLMClient
from rxn_classifier.classification.models import DetailedResult, VerificationResult
from rxn_classifier.classification.utils import build_hierarchy_subtree, format_rxn_roles_with_strip
from rxn_classifier.prompts.verify_classification import cohort_system_prompt as VERIFY_COHORT_SYS


_MATCH = re.compile(r"<match>\s*(true|false)\s*</match>", re.IGNORECASE)
_REASON = re.compile(r"<reason>\s*(.*?)\s*</reason>", re.DOTALL | re.IGNORECASE)
_ACTION = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL | re.IGNORECASE)


def _extract(text: str, pat: re.Pattern) -> str:
    m = pat.search(text or "")
    return m.group(1).strip() if m else ""


@dataclass
class VerifierAgent:
    llm: BaseLLMClient
    strip_mapping: bool = False
    # The run's current mapping (X.Y -> deeper hierarchy lines); set by the pipeline.
    mapping_override: Optional[Dict[str, List[str]]] = None

    def verify_cohort_batch(
        self,
        cohorts: List[tuple[List[str], str]],
        detailed_results: List[DetailedResult],
        parent_codes: List[str],
    ) -> List[VerificationResult]:
        """Check one proposed classification per cohort.

        cohorts: list of (reactions_list, shared_template)
        detailed_results: the detailed agent's answer per cohort
        parent_codes: two-level code per cohort, whose subtree the verifier sees
        """
        n = len(cohorts)
        if len(detailed_results) != n or len(parent_codes) != n:
            raise ValueError("verify_cohort_batch inputs must have the same length")
        sys_prompts: List[str] = []
        user_contents: List[str] = []
        for (rxns, tpl), det, parent in zip(cohorts, detailed_results, parent_codes):
            hierarchy_text = build_hierarchy_subtree(parent, self.mapping_override)
            sys_prompts.append(
                VERIFY_COHORT_SYS
                .replace("{{reaction_hierarchy}}", hierarchy_text)
                .replace("{{reactions}}", "\n".join(format_rxn_roles_with_strip(r, strip_mapping=self.strip_mapping) for r in rxns))
                .replace("{{retro_template}}", tpl or "")
                .replace("{{reaction_super_class}}", det.reaction_super_class)
                .replace("{{reaction_sub_class}}", det.reaction_sub_class)
                .replace("{{reaction_type}}", det.reaction_type)
                .replace("{{reaction_subtype}}", det.reaction_subtype)
                .replace("{{class_number}}", det.class_number)
            )
            user_contents.append(" ")
        responses = self.llm.generate_batch_multi(sys_prompts, user_contents)
        out: List[VerificationResult] = []
        for (rxns, _), det, resp in zip(cohorts, detailed_results, responses):
            txt = resp.text or ""
            out.append(
                VerificationResult(
                    reaction_smiles=(rxns[0] if rxns else ""),
                    class_number=det.class_number,
                    match=(_extract(txt, _MATCH).lower() == "true"),
                    reason=_extract(txt, _REASON),
                    action=_extract(txt, _ACTION),
                )
            )
        return out
