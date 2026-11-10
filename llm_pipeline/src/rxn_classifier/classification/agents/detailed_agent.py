from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from rxn_classifier.classification.llm_client import BaseLLMClient
from rxn_classifier.classification.models import DetailedResult
from rxn_classifier.classification.utils import build_hierarchy_subtree, format_rxn_roles_with_strip
from rxn_classifier.prompts.detailed_classification import cohort_system_prompt as DETAILED_COHORT_SYS


_TAGS = ["reaction_super_class", "reaction_sub_class", "reaction_type", "reaction_subtype", "class_number"]
TAG_PATTERNS = {
    tag: re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.DOTALL | re.IGNORECASE) for tag in _TAGS
}


def _extract_tag(text: str, tag: str) -> str:
    m = TAG_PATTERNS[tag].search(text)
    return m.group(1).strip() if m else ""


@dataclass
class DetailedAgent:
    llm: BaseLLMClient
    strip_mapping: bool = False
    # The run's current mapping (X.Y -> deeper hierarchy lines); set by the pipeline.
    mapping_override: Optional[Dict[str, List[str]]] = None

    def classify_cohort_batch(
        self,
        cohorts: List[tuple[List[str], str]],
        two_level_codes: List[str],
    ) -> List[DetailedResult]:
        """Classify cohorts inside the subtree of their two-level code.

        cohorts: list of (reactions_list, shared_template)
        two_level_codes: one two-level parent code per cohort
        """
        if len(two_level_codes) != len(cohorts):
            raise ValueError("two_level_codes length must match cohorts")

        sys_prompts: List[str] = []
        user_contents: List[str] = []
        for (rxns, tpl), code in zip(cohorts, two_level_codes):
            hierarchy_text = build_hierarchy_subtree(code, self.mapping_override)
            sys_prompts.append(
                DETAILED_COHORT_SYS
                .replace("{{reaction_hierarchy}}", hierarchy_text)
                .replace("{{reactions}}", "\n".join(format_rxn_roles_with_strip(r, strip_mapping=self.strip_mapping) for r in rxns))
                .replace("{{retro_template}}", tpl or "")
            )
            user_contents.append(" ")
        responses = self.llm.generate_batch_multi(sys_prompts, user_contents)
        out: List[DetailedResult] = []
        for (rxns, _), resp in zip(cohorts, responses):
            text = resp.text or ""
            out.append(
                DetailedResult(
                    reaction_smiles=(rxns[0] if rxns else ""),
                    reaction_super_class=_extract_tag(text, "reaction_super_class"),
                    reaction_sub_class=_extract_tag(text, "reaction_sub_class"),
                    reaction_type=_extract_tag(text, "reaction_type"),
                    reaction_subtype=_extract_tag(text, "reaction_subtype"),
                    class_number=_extract_tag(text, "class_number"),
                )
            )
        return out
