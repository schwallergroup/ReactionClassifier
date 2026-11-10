from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from rxn_classifier.classification.llm_client import BaseLLMClient
from rxn_classifier.classification.models import ClassProposal
from rxn_classifier.classification.utils import build_hierarchy_subtree, format_rxn_roles_with_strip
from rxn_classifier.prompts.generate_new_class import cohort_system_prompt as GEN_COHORT_SYS


_PARENT = re.compile(r"<suggested_parent_code>\s*(.*?)\s*</suggested_parent_code>", re.DOTALL | re.IGNORECASE)
_CODE = re.compile(r"<new_entry_code>\s*(.*?)\s*</new_entry_code>", re.DOTALL | re.IGNORECASE)
_NAME = re.compile(r"<new_entry_name>\s*(.*?)\s*</new_entry_name>", re.DOTALL | re.IGNORECASE)
_LINE = re.compile(r"<hierarchy_line>\s*(.*?)\s*</hierarchy_line>", re.DOTALL | re.IGNORECASE)
_PARENTS_LINES = re.compile(r"<new_parent_lines>\s*(.*?)\s*</new_parent_lines>", re.DOTALL | re.IGNORECASE)
_WHY = re.compile(r"<rationale>\s*(.*?)\s*</rationale>", re.DOTALL | re.IGNORECASE)


def _extract(text: str, pat: re.Pattern) -> str:
    m = pat.search(text or "")
    return m.group(1).strip() if m else ""


@dataclass
class GeneratorAgent:
    llm: BaseLLMClient
    strip_mapping: bool = False
    # The run's current mapping (X.Y -> deeper hierarchy lines); set by the pipeline.
    mapping_override: Optional[Dict[str, List[str]]] = None

    def propose_cohort_batch(
        self,
        cohorts: List[tuple[List[str], str]],
        parent_codes: List[str],
    ) -> List[ClassProposal]:
        """Propose one new class per cohort.

        cohorts: list of (reactions_list, shared_template)
        parent_codes: two-level code per cohort, whose subtree the generator sees

        Note: the prompt's {{cause}} placeholder is left unfilled, as in the
        runs behind the published results.
        """
        if len(parent_codes) != len(cohorts):
            raise ValueError("parent_codes length must match cohorts")
        sys_prompts: List[str] = []
        user_contents: List[str] = []
        for (rxns, tpl), parent in zip(cohorts, parent_codes):
            hierarchy_text = build_hierarchy_subtree(parent, self.mapping_override)
            sys_prompts.append(
                GEN_COHORT_SYS
                .replace("{{reaction_hierarchy}}", hierarchy_text)
                .replace("{{reactions}}", "\n".join(format_rxn_roles_with_strip(r, strip_mapping=self.strip_mapping) for r in rxns))
                .replace("{{retro_template}}", tpl or "")
            )
            user_contents.append(" ")
        responses = self.llm.generate_batch_multi(sys_prompts, user_contents)
        out: List[ClassProposal] = []
        for (rxns, _), resp in zip(cohorts, responses):
            txt = resp.text or ""
            parents_blob = _extract(txt, _PARENTS_LINES)
            add_lines = [ln.strip() for ln in parents_blob.splitlines() if ln.strip()] if parents_blob else []
            out.append(
                ClassProposal(
                    reaction_smiles=(rxns[0] if rxns else ""),
                    suggested_parent_code=_extract(txt, _PARENT),
                    new_entry_code=_extract(txt, _CODE),
                    new_entry_name=_extract(txt, _NAME),
                    hierarchy_line=_extract(txt, _LINE),
                    rationale=_extract(txt, _WHY),
                    additional_lines=add_lines,
                )
            )
        return out
