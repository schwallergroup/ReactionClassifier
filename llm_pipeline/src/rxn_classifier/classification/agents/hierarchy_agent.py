from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from rxn_classifier.classification.llm_client import BaseLLMClient
from rxn_classifier.classification.utils import strip_atom_mapping_reactants_products
from rxn_classifier.prompts.hierarchy_classification import COHORT_SYSTEM_PROMPT as HIER_COHORT_SYS


_TAG_RE = re.compile(r"<reaction_class>\s*([^<\n\r]+?)\s*</reaction_class>", re.IGNORECASE)


def _parse_single_code(text: str) -> str:
    m = _TAG_RE.search(text or "")
    return m.group(1).strip() if m else ""


@dataclass
class HierarchyAgent:
    llm: BaseLLMClient
    strip_mapping: bool = False

    def classify_cohort_batch(self, cohorts: List[tuple[list[str], str]]) -> List[str]:
        """Classify cohorts. Each cohort is (smiles_list, retro_template).

        Returns one two-level code per cohort.
        """
        sys_prompts: List[str] = []
        user_contents: List[str] = []
        for smiles_list, retro_template in cohorts:
            inputs = smiles_list
            if self.strip_mapping:
                inputs = [strip_atom_mapping_reactants_products(s) for s in inputs]
            joined = "\n".join(inputs)
            sys_prompts.append(
                HIER_COHORT_SYS.replace("{{reaction_smiles}}", joined).replace("{{retro_template}}", retro_template or "")
            )
            user_contents.append(" ")
        responses = self.llm.generate_batch_multi(sys_prompts, user_contents)
        return [_parse_single_code(r.text) for r in responses]
