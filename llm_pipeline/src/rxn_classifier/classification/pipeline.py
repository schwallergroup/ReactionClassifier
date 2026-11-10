from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from rxn_classifier.classification.agents.detailed_agent import DetailedAgent
from rxn_classifier.classification.agents.verifier_agent import VerifierAgent
from rxn_classifier.classification.agents.generator_agent import GeneratorAgent
from rxn_classifier.classification.agents.hierarchy_agent import HierarchyAgent
from rxn_classifier.classification.agents.aggregator_agent import AggregatorAgent
from rxn_classifier.classification.llm_client import (
    BaseLLMClient,
    GoogleGenAIClient,
    NoOpLLMClient,
)
from rxn_classifier.classification.models import (
    AgentBaseConfig,
    CohortModeConfig,
    DetailedResult,
    PipelineConfig,
    ClassProposal,
    VerificationResult,
)
from rxn_classifier.classification.utils import (
    save_json,
    load_hierarchy_codes,
    load_detailed_results,
    load_verify_results,
    apply_lines_to_mapping_json,
    load_mapping,
    apply_lines_to_mapping_dict,
    build_code_label_lookup,
    next_mapping_path,
)

LLM_COLUMNS = [
    "LLM reaction super class",
    "LLM reaction sub class",
    "LLM reaction type",
    "LLM reaction subtype",
    "LLM class number",
]


def _make_llm_client(
    model: str,
    temperature: float | None,
    thinking_budget: int | None,
    log_io: bool,
    save_io_dir,
    client_tag: str = "",
    fail_on_noop: bool = False,
) -> BaseLLMClient:
    try:
        return GoogleGenAIClient(
            model=model,
            temperature=temperature,
            thinking_budget=thinking_budget,
            log_io=log_io,
            save_io_dir=save_io_dir,
            client_tag=client_tag,
        )
    except Exception as e:
        print(f"[LLM setup] Falling back to NoOpLLMClient for tag='{client_tag}'. Reason: {e}")
        if fail_on_noop:
            raise
        return NoOpLLMClient(
            model=model,
            temperature=temperature,
            thinking_budget=thinking_budget,
            log_io=log_io,
            save_io_dir=save_io_dir,
            client_tag=client_tag,
        )


def _two_level(code: str) -> str:
    """Cut a code to its first two levels: '2.1.3' -> '2.1'."""
    two = (code or "").strip()
    if two and two.count('.') >= 2:
        two = '.'.join(two.split('.')[:2])
    return two


def _blank_row(class_number: str) -> dict:
    """Output values for a row that has a marker instead of a class."""
    row = {col: '' for col in LLM_COLUMNS}
    row["LLM class number"] = class_number
    return row


@dataclass
class ClassificationPipeline:
    config: PipelineConfig

    # ---------- Setup ----------
    def _client_for(self, tag: str, agent_cfg: AgentBaseConfig | None) -> BaseLLMClient:
        """LLM client for one agent: its own settings where given, else the global ones."""
        cfg = self.config
        model = (agent_cfg.model if agent_cfg else None) or cfg.llm_model
        temp = agent_cfg.temperature if agent_cfg and agent_cfg.temperature is not None else cfg.temperature
        thinking = (
            agent_cfg.thinking_budget
            if agent_cfg and agent_cfg.thinking_budget is not None
            else cfg.thinking_budget
        )
        log_io = bool(agent_cfg.log_io) if agent_cfg and agent_cfg.log_io is not None else False
        global_log = bool(cfg.log_llm_io)
        # If logging is on but no save dir is given, save under ./llm_io
        global_save = cfg.llm_io_save_dir or (Path("llm_io") if global_log else None)
        save_dir = agent_cfg.save_io_dir if agent_cfg and agent_cfg.save_io_dir else global_save
        return _make_llm_client(
            model, temp, thinking, log_io or global_log, save_dir,
            client_tag=tag,
            fail_on_noop=cfg.fail_on_noop,
        )

    @staticmethod
    def _strip(agent_cfg: AgentBaseConfig | None) -> bool:
        return bool(agent_cfg.strip_mapping) if agent_cfg and agent_cfg.strip_mapping is not None else False

    def _set_mapping(self, mapping: dict, *agents) -> None:
        for agent in agents:
            agent.mapping_override = mapping

    def _sample_rows(self, key: str, idxs: list[int], smiles_all: list[str]) -> list[int]:
        """Pick the rows of a cohort that are shown to the agents.

        A cohort within the size limit is shown whole. A larger one is sampled at
        random, so the agents do not just see whichever reactions come first in the
        file (often one patent). The draw is seeded by the template hash and made
        over the reactions in SMILES order, so a cohort gets the same sample however
        the input rows are ordered or batched. Sampled rows keep their input order.
        """
        cm = self.config.cohort_mode or CohortModeConfig()
        n = cm.reactions_per_cohort
        if len(idxs) <= n:
            return idxs
        if cm.sample_seed is None:
            return idxs[:n]
        by_smiles = sorted(idxs, key=lambda j: smiles_all[j])
        return sorted(random.Random(f"{cm.sample_seed}:{key}").sample(by_smiles, n))

    def _make_chunks(self, cohort_items: list, smiles_all: list[str], templates_all: list[str]) -> list[dict]:
        """Turn (template_hash, row positions) pairs into cohorts.

        'indices' and 'smiles' are the sampled rows the agents see; 'all_indices'
        is every row of the cohort, which all receive the cohort's class.
        """
        chunks = []
        for key, idxs in cohort_items:
            sel = self._sample_rows(key, idxs, smiles_all)
            templates_list = [templates_all[j] for j in sel]
            chunks.append({
                'key': key,
                'indices': sel,
                'all_indices': idxs,
                'smiles': [smiles_all[j] for j in sel],
                'shared_template': next((t for t in templates_list if t), ""),
            })
        return chunks

    # ---------- Stage: Hierarchy ----------
    def _run_hierarchy_stage(self, i: int, cohort_chunks: list[dict], test_enabled: bool, test_cfg, hier_agent: HierarchyAgent) -> list[str]:
        cohorts_in = [(g['smiles'], g['shared_template']) for g in cohort_chunks]
        cbs = int(getattr(self.config.cohort_mode, 'cohort_batch_size', 10))
        mcej = int(getattr(self.config.cohort_mode, 'max_cohorts_per_llm_job', 200))
        job_size = max(1, min(cbs, mcej))
        cohort_codes_all: list[str] = []
        call_idx = 0
        for j in range(0, len(cohorts_in), job_size):
            sub = cohorts_in[j:j+job_size]
            rxn_total = sum(len(x[0]) for x in sub)
            call_idx += 1
            print(f"[Pipeline][Hierarchy][Batch {i+1}][Call {call_idx}] cohorts={len(sub)} reactions={rxn_total}")
            cohort_codes_all.extend(hier_agent.classify_cohort_batch(sub))
        # Broadcast the cohort's code to each member, cut to two levels
        two_level_codes: list[str] = []
        for g, cc in zip(cohort_chunks, cohort_codes_all):
            two_level_codes.extend([_two_level(cc)] * len(g['smiles']))
        if test_enabled:
            out_dir = getattr(test_cfg, "hierarchy_dir", None) or test_cfg.output_dir
            save_json(out_dir / f"hierarchy_codes_batch_{i+1}.json", {sm: code for g, code in zip(cohort_chunks, cohort_codes_all) for sm in g['smiles']})
        return two_level_codes

    # ---------- Stage: Detailed ----------
    def _run_detailed_stage(self, i: int, cohort_chunks: list[dict], two_level_codes: list[str], det_agent: DetailedAgent, test_enabled: bool, test_cfg) -> list[DetailedResult]:
        cohorts_in = [(g['smiles'], g['shared_template']) for g in cohort_chunks]
        # Parent code per cohort = the two-level code of its first member
        parents_all = []
        offset = 0
        for g in cohort_chunks:
            parents_all.append(two_level_codes[offset] if offset < len(two_level_codes) else "")
            offset += len(g['smiles'])
        job_size = int(getattr(self.config.cohort_mode, 'max_cohorts_per_llm_job', 200))
        det_list_all: list[DetailedResult] = []
        call_idx = 0
        for j in range(0, len(cohorts_in), job_size):
            sub = cohorts_in[j:j+job_size]
            sub_parents = parents_all[j:j+job_size]
            rxn_total = sum(len(x[0]) for x in sub)
            call_idx += 1
            print(f"[Pipeline][Detailed][Batch {i+1}][Call {call_idx}] cohorts={len(sub)} reactions={rxn_total}")
            det_list_all.extend(det_agent.classify_cohort_batch(sub, sub_parents))
        # The detailed prompt asks for a class below X.Y, or X.Y.Other. Hold a bare
        # X.Y answer to that rule, else the verifier can accept it as a final class.
        if not (self.config.detailed_agent and self.config.detailed_agent.allow_level2_class):
            for det in det_list_all:
                if re.fullmatch(r"\d+\.\d+", det.class_number.strip()):
                    det.class_number = det.class_number.strip() + ".Other"
        # Broadcast the cohort's result to each member
        detailed_results: list[DetailedResult] = []
        for g, det in zip(cohort_chunks, det_list_all):
            for rxn in g['smiles']:
                detailed_results.append(det.model_copy(update={"reaction_smiles": rxn}))
        if test_enabled:
            out_dir = getattr(test_cfg, "detailed_dir", None) or test_cfg.output_dir
            save_json(out_dir / f"detailed_results_batch_{i+1}.json", [res.model_dump() for res in detailed_results])
        return detailed_results

    # ---------- Stage: Verify ----------
    def _run_verify_stage(self, i: int, cohort_chunks: list[dict], detailed_results: list[DetailedResult], two_level_codes: list[str], ver_agent: VerifierAgent, test_enabled: bool, test_cfg):
        cohorts_in = [(g['smiles'], g['shared_template']) for g in cohort_chunks]
        # Per cohort: its first member's detailed result, and the parent code whose
        # subtree the verifier sees (top two levels of the detailed class number,
        # else the hierarchy agent's code)
        dets_all = []
        parent_codes = []
        offset = 0
        for g in cohort_chunks:
            det = detailed_results[offset]
            dets_all.append(det)
            if det.class_number:
                parent_codes.append('.'.join(det.class_number.split('.')[:2]))
            else:
                parent_codes.append(two_level_codes[offset] if offset < len(two_level_codes) else "")
            offset += len(g['smiles'])
        job_size = int(getattr(self.config.cohort_mode, 'max_cohorts_per_llm_job', 200))
        vcohorts_all: list[VerificationResult] = []
        call_idx = 0
        for j in range(0, len(cohorts_in), job_size):
            sub = cohorts_in[j:j+job_size]
            rxn_total = sum(len(x[0]) for x in sub)
            call_idx += 1
            print(f"[Pipeline][Verify][Batch {i+1}][Call {call_idx}] cohorts={len(sub)} reactions={rxn_total}")
            vcohorts_all.extend(ver_agent.verify_cohort_batch(sub, dets_all[j:j+job_size], parent_codes[j:j+job_size]))
        # Broadcast the cohort's verdict to each member
        vresults: list[VerificationResult] = []
        for g, vr in zip(cohort_chunks, vcohorts_all):
            for rxn in g['smiles']:
                vresults.append(vr.model_copy(update={"reaction_smiles": rxn}))
        if test_enabled:
            out_dir = getattr(test_cfg, "verify_dir", None) or test_cfg.output_dir
            save_json(out_dir / f"verify_results_batch_{i+1}.json", [vr.model_dump() for vr in vresults])
        return vresults, parent_codes

    def _split_rejected(self, cohort_chunks: list[dict], vresults: list[VerificationResult], parent_codes: list[str]):
        """Pick the cohorts the verifier rejected.

        Returns (mis_groups, hierarchy_mismatches):
        - mis_groups: (cohort, parent_code) pairs that go on to the generator
        - hierarchy_mismatches: (row position, record) for members of cohorts flagged
          incorrect_hierarchy. Only filled when verifier_agent.exclude_hierarchy_mismatch
          is on; those cohorts are then kept away from the generator.
        """
        exclude = bool(self.config.verifier_agent and self.config.verifier_agent.exclude_hierarchy_mismatch)
        mis_groups: list[tuple[dict, str]] = []
        hierarchy_mismatches: list[tuple[int, dict]] = []
        offset = 0
        for g, parent in zip(cohort_chunks, parent_codes):
            k = len(g['smiles'])
            sub_v = vresults[offset:offset+k]
            offset += k
            if exclude and any(vr.action == "incorrect_hierarchy" for vr in sub_v):
                for pos, vr in zip(g['indices'], sub_v):
                    hierarchy_mismatches.append((pos, {
                        self.config.smiles_column: vr.reaction_smiles,
                        "class_number": vr.class_number,
                        "reason": vr.reason,
                        "action": vr.action,
                    }))
            elif any(not vr.match for vr in sub_v):
                mis_groups.append((g, parent))
        return mis_groups, hierarchy_mismatches

    # ---------- Stage: Generate ----------
    def _run_generate_stage(self, i: int, mis_groups: list[tuple[dict, str]], gen_agent: GeneratorAgent) -> list[ClassProposal]:
        if not mis_groups:
            return []
        cohorts_in = [(g['smiles'], g['shared_template']) for g, _p in mis_groups]
        parents = [p for _g, p in mis_groups]
        job_size = int(getattr(self.config.cohort_mode, 'max_cohorts_per_llm_job', 200))
        props_all: list[ClassProposal] = []
        call_idx = 0
        for j in range(0, len(cohorts_in), job_size):
            sub = cohorts_in[j:j+job_size]
            rxn_total = sum(len(x[0]) for x in sub)
            call_idx += 1
            print(f"[Pipeline][Generate][Batch {i+1}][Call {call_idx}] cohorts={len(sub)} reactions={rxn_total}")
            props_all.extend(gen_agent.propose_cohort_batch(sub, parents[j:j+job_size]))
        return props_all

    # ---------- Stage: Aggregate (group by two-level parent) ----------
    def _run_aggregate_stage(self, i: int, mis_groups: list[tuple[dict, str]], props: list[ClassProposal], agg_agent: AggregatorAgent, dynamic_mapping: dict, mapping_out_path: Path, test_enabled: bool, out_dir) -> tuple[dict, Dict[int, dict], int]:
        """Returns (updated mapping, {row position: output values}, lines added)."""
        if not mis_groups or not props:
            return dynamic_mapping, {}, 0
        if test_enabled and out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
        # row position -> final code
        assigned: Dict[int, str] = {}
        added_lines_this_batch = 0
        # Group proposals by two-level parent
        parent_to_group: dict[str, list[tuple[int, ClassProposal]]] = {}
        for idx_local, ((g, parent), prop) in enumerate(zip(mis_groups, props)):
            key = '.'.join((parent or '').split('.')[:2])
            parent_to_group.setdefault(key, []).append((idx_local, prop))
        max_group = int(self.config.aggregator_agent.max_cohorts_per_call) if self.config.aggregator_agent else 10
        # One aggregator call per chunk of each parent group
        for parent_two, items in parent_to_group.items():
            for start_idx in range(0, len(items), max_group):
                sub_items = items[start_idx:start_idx+max_group]
                grp_indices = [idx for (idx, _p) in sub_items]
                grp_props = [p for (_idx, p) in sub_items]
                print(f"[Pipeline][Aggregator][Batch {i+1}] proposals_in={len(grp_props)} parents=1 parent={parent_two}")
                new_lines, assignments, raw_txt = agg_agent.aggregate(grp_props, list(range(len(grp_props))), [parent_two] * len(grp_props))
                print(f"[Pipeline][Aggregator][Batch {i+1}] assignments_out={len(assignments or {})} new_lines_out={len(new_lines or [])}")
                # Update the mapping now so the next aggregator call sees the new entries
                conflicts_this_chunk: dict[str, str] = {}
                if new_lines:
                    dynamic_mapping, added_now, conflicts_this_chunk = apply_lines_to_mapping_dict(new_lines, dynamic_mapping)
                    added_lines_this_batch += int(added_now)
                    if conflicts_this_chunk:
                        print(f"[Pipeline][Aggregator][Batch {i+1}] CONFLICTS detected: {list(conflicts_this_chunk.keys())}")
                    agg_agent.mapping_override = dynamic_mapping
                # Map assignments back to the cohorts' rows, marking conflicts
                for k, code in (assignments or {}).items():
                    try:
                        bi = int(str(k))
                    except Exception:
                        continue
                    if 0 <= bi < len(grp_indices):
                        g, _parent = mis_groups[grp_indices[bi]]
                        final_code = str(code)
                        if final_code in conflicts_this_chunk:
                            final_code = f"CONFLICT:{final_code}"
                        for pos in g['indices']:
                            assigned[pos] = final_code
                if test_enabled:
                    (out_dir / f"aggregator_new_hierarchy_batch_{i+1}.txt").write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")
                    save_json(out_dir / f"aggregator_assignments_batch_{i+1}.json", assignments)
                    (out_dir / f"aggregator_raw_batch_{i+1}.txt").write_text(raw_txt or "", encoding="utf-8")

        if added_lines_this_batch > 0:
            total_json = apply_lines_to_mapping_json(dynamic_mapping, mapping_out_path)
            print(f"[PIPELINE][Mapping][JSON] total_lines={total_json} -> {mapping_out_path}")

        # Output values that replace the detailed agent's class for the assigned rows
        updates: Dict[int, dict] = {}
        code2label = build_code_label_lookup(dynamic_mapping)
        for pos, code_str in assigned.items():
            if code_str.startswith("CONFLICT:"):
                # Keep CONFLICT:X.Y.Z.W as the class number; leave the labels empty
                updates[pos] = _blank_row(code_str)
            else:
                parts = [p for p in code_str.split('.') if p]
                levels = ['.'.join(parts[:n]) if len(parts) >= n else '' for n in (1, 2, 3, 4)]
                updates[pos] = {
                    "LLM reaction super class": code2label.get(levels[0], ''),
                    "LLM reaction sub class": code2label.get(levels[1], ''),
                    "LLM reaction type": code2label.get(levels[2], ''),
                    "LLM reaction subtype": code2label.get(levels[3], ''),
                    "LLM class number": code_str,
                }
        return dynamic_mapping, updates, added_lines_this_batch

    # ---------- Hierarchy-only mode ----------
    def _run_hierarchy_only(self, df: pd.DataFrame, cohort_items: list, cohort_chunks: list[dict], hier_agent: HierarchyAgent, test_cfg) -> pd.DataFrame:
        out_dir = test_cfg.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        cohorts = [(g['smiles'], g['shared_template']) for g in cohort_chunks]
        job_size = max(1, int(getattr(self.config.cohort_mode, 'cohort_batch_size', 10)))
        cohort_codes: List[str] = []
        call_idx = 0
        for j in range(0, len(cohorts), job_size):
            sub = cohorts[j:j+job_size]
            rxn_total = sum(len(x[0]) for x in sub)
            call_idx += 1
            print(f"[Pipeline][Hierarchy][AllInOne][Call {call_idx}] cohorts={len(sub)} reactions={rxn_total}")
            cohort_codes.extend(hier_agent.classify_cohort_batch(sub))
        # Every row of a cohort gets the cohort's code
        code_per_row = [""] * len(df)
        for (key, idxs), code in zip(cohort_items, cohort_codes):
            for j in idxs:
                code_per_row[j] = _two_level(code)
        out_df = pd.DataFrame({
            self.config.smiles_column: df[self.config.smiles_column],
            "classification": code_per_row,
        })
        out_csv = test_cfg.hierarchy_only_output_csv or (out_dir / "hierarchy_only.csv")
        out_df.to_csv(out_csv, index=False)
        counts = pd.Series(code_per_row).value_counts().sort_values(ascending=False)
        (out_dir / "hierarchy_distribution.json").write_text(json.dumps(counts.to_dict(), indent=2), encoding="utf-8")
        return out_df

    # ---------- Run ----------
    def run(self) -> pd.DataFrame:
        cfg = self.config
        df = pd.read_csv(cfg.input_csv, sep=",")
        if cfg.smiles_column not in df.columns:
            raise ValueError(f"Column '{cfg.smiles_column}' not found in input CSV.")

        # Optional slicing by row range, then the row cap
        if cfg.start_row is not None or cfg.end_row is not None:
            s = int(cfg.start_row) if cfg.start_row is not None else 0
            e = int(cfg.end_row) if cfg.end_row is not None else None
            df = df.iloc[s:e].copy()
        if cfg.max_rows is not None:
            df = df.head(cfg.max_rows).copy()
        if df.empty:
            return pd.DataFrame()

        th_col = cfg.cohort_mode.template_hash_column if cfg.cohort_mode else "template_hash"
        if th_col not in df.columns:
            raise ValueError(f"Cohort mode requires column '{th_col}' in input CSV")

        hier_agent = HierarchyAgent(llm=self._client_for("hierarchy", cfg.hierarchy_agent), strip_mapping=self._strip(cfg.hierarchy_agent))
        det_agent = DetailedAgent(llm=self._client_for("detailed", cfg.detailed_agent), strip_mapping=self._strip(cfg.detailed_agent))
        ver_agent = VerifierAgent(llm=self._client_for("verifier", cfg.verifier_agent), strip_mapping=self._strip(cfg.verifier_agent))
        gen_agent = GeneratorAgent(llm=self._client_for("generator", cfg.generator_agent), strip_mapping=self._strip(cfg.generator_agent))
        agg_agent = AggregatorAgent(llm=self._client_for("aggregator", cfg.aggregator_agent), strip_mapping=self._strip(cfg.aggregator_agent))

        # The mapping (X.Y -> deeper hierarchy lines) is held in memory, shared by
        # all agents, and grows as the aggregator adds classes
        dynamic_mapping = load_mapping(cfg.mapping_json) or {}
        self._set_mapping(dynamic_mapping, det_agent, ver_agent, gen_agent, agg_agent)
        # The grown mapping is written to a new file, never over the one that was read
        mapping_out_path = (cfg.aggregator_agent.dynamic_mapping_output_json if cfg.aggregator_agent else None) or next_mapping_path(cfg.mapping_json, cfg.output_csv)

        smiles_all: List[str] = df[cfg.smiles_column].astype(str).tolist()
        if cfg.template_column in df.columns:
            templates_all: List[str] = df[cfg.template_column].fillna("").astype(str).tolist()
        else:
            templates_all = [""] * len(df)

        # Cohorts: row positions grouped by template hash, in order of first appearance
        cohorts_map = defaultdict(list)
        for idx, key in enumerate(df[th_col].astype(str).tolist()):
            cohorts_map[str(key)].append(idx)
        cohort_items = list(cohorts_map.items())
        cohort_batch_size = max(1, int(getattr(cfg.cohort_mode, 'cohort_batch_size', 10)))
        n_batches = int(math.ceil(len(cohort_items) / cohort_batch_size))

        test_cfg = cfg.test_mode
        test_enabled = bool(test_cfg and test_cfg.enabled)
        if test_enabled and test_cfg.hierarchy_only_all_in_one:
            return self._run_hierarchy_only(df, cohort_items, self._make_chunks(cohort_items, smiles_all, templates_all), hier_agent, test_cfg)
        if test_enabled:
            test_cfg.output_dir.mkdir(parents=True, exist_ok=True)
            print(f"[Pipeline] Test mode enabled. Saving artifacts under: {test_cfg.output_dir}")
        start_stage = test_cfg.start_stage if test_enabled else "hierarchy"
        stop_after = test_cfg.stop_after_stage if test_enabled else None

        run_detailed = bool(cfg.detailed_agent.enabled) if cfg.detailed_agent else True
        enable_verifier = cfg.verifier_agent.enabled if cfg.verifier_agent else cfg.enable_verification
        enable_generator = cfg.generator_agent.enabled if cfg.generator_agent else False
        enable_aggregator = cfg.aggregator_agent is None or cfg.aggregator_agent.enabled

        # Results, keyed by row position in df
        row_results: Dict[int, dict] = {}          # detailed agent's class
        aggregator_updates: Dict[int, dict] = {}   # aggregator's class, replaces the above
        hierarchy_mismatches: List[tuple[int, dict]] = []
        proposals: List[ClassProposal] = []
        batch_stats: List[dict] = []
        all_chunks: List[dict] = []

        for i in range(n_batches):
            cohort_chunks = self._make_chunks(cohort_items[i * cohort_batch_size:(i + 1) * cohort_batch_size], smiles_all, templates_all)
            chunk = [sm for g in cohort_chunks for sm in g['smiles']]
            chunk_rows = [pos for g in cohort_chunks for pos in g['indices']]
            all_chunks.extend(cohort_chunks)

            # 1) Hierarchy agent, or load its codes
            if start_stage == "hierarchy":
                two_level_codes = self._run_hierarchy_stage(i, cohort_chunks, test_enabled, test_cfg, hier_agent)
                if stop_after == "hierarchy":
                    continue
            else:
                if not (test_cfg and test_cfg.hierarchy_codes_path):
                    raise ValueError("test_mode.start_stage != 'hierarchy' requires hierarchy_codes_path")
                two_level_codes = [_two_level(c) for c in load_hierarchy_codes(test_cfg.hierarchy_codes_path, chunk)]

            # 2) Detailed agent, or load its results
            if start_stage in ("hierarchy", "detailed") and run_detailed:
                detailed_results = self._run_detailed_stage(i, cohort_chunks, two_level_codes, det_agent, test_enabled, test_cfg)
            elif start_stage in ("verify", "generate"):
                if not (test_cfg and test_cfg.detailed_results_path):
                    raise ValueError("test_mode.start_stage in {'verify','generate'} requires detailed_results_path")
                detailed_results = [
                    DetailedResult(
                        reaction_smiles=rxn,
                        reaction_super_class=str(dd.get("reaction_super_class", "")),
                        reaction_sub_class=str(dd.get("reaction_sub_class", "")),
                        reaction_type=str(dd.get("reaction_type", "")),
                        reaction_subtype=str(dd.get("reaction_subtype", "")),
                        class_number=str(dd.get("class_number", "")),
                    )
                    for rxn, dd in zip(chunk, load_detailed_results(test_cfg.detailed_results_path, chunk))
                ]
            else:
                detailed_results = []
            for pos, res in zip(chunk_rows, detailed_results):
                row = res.to_row(cfg.smiles_column)
                row_results[pos] = {col: row[col] for col in LLM_COLUMNS}
            if stop_after == "detailed" and start_stage in ("hierarchy", "detailed") and run_detailed:
                continue

            # 3-5) Verify -> generate -> aggregate
            added_lines_this_batch = 0
            mis_groups: list[tuple[dict, str]] = []
            proposals_batch: List[ClassProposal] = []
            stopped = False

            if enable_verifier and len(detailed_results) == len(chunk):
                if start_stage in ("hierarchy", "detailed", "verify"):
                    vresults, parent_codes = self._run_verify_stage(i, cohort_chunks, detailed_results, two_level_codes, ver_agent, test_enabled, test_cfg)
                else:
                    # Load verify results for the 'generate' start stage
                    if not (test_cfg and test_cfg.verify_results_path):
                        raise ValueError("test_mode.start_stage == 'generate' requires verify_results_path")
                    vresults = [
                        VerificationResult(
                            reaction_smiles=rxn,
                            class_number=str(vd.get("class_number", det.class_number)),
                            match=bool(vd.get("match", False)),
                            reason=str(vd.get("reason", "")),
                            action=str(vd.get("action", "")),
                        )
                        for rxn, vd, det in zip(chunk, load_verify_results(test_cfg.verify_results_path, chunk), detailed_results)
                    ]
                    # Parent code per cohort, from its first member
                    parent_codes = []
                    offset = 0
                    for g in cohort_chunks:
                        det = detailed_results[offset]
                        parent_codes.append(".".join(det.class_number.split(".")[:2]) if det.class_number else two_level_codes[offset])
                        offset += len(g['smiles'])

                mis_groups, mismatches_batch = self._split_rejected(cohort_chunks, vresults, parent_codes)
                hierarchy_mismatches.extend(mismatches_batch)
                if mismatches_batch and test_enabled:
                    out_dir = getattr(test_cfg, "verify_dir", None) or test_cfg.output_dir
                    save_json(out_dir / f"incorrect_hierarchy_batch_{i+1}.json", [rec for _pos, rec in mismatches_batch])

                stopped = stop_after == "verify"
                if not stopped:
                    if enable_generator:
                        proposals_batch = self._run_generate_stage(i, mis_groups, gen_agent)
                        proposals.extend(proposals_batch)
                        if test_enabled:
                            pdir = getattr(test_cfg, "proposals_dir", None) or test_cfg.output_dir
                            save_json(pdir / f"proposals_batch_{i+1}.json", [p.model_dump() for p in proposals_batch])

                    if enable_aggregator and proposals_batch:
                        agg_out_dir = (getattr(test_cfg, "aggregator_dir", None) or test_cfg.output_dir) if test_enabled else None
                        dynamic_mapping, agg_updates, added_lines_this_batch = self._run_aggregate_stage(
                            i, mis_groups, proposals_batch, agg_agent, dynamic_mapping, mapping_out_path, test_enabled, agg_out_dir
                        )
                        aggregator_updates.update(agg_updates)
                        # All agents see the grown mapping from the next batch on
                        self._set_mapping(dynamic_mapping, det_agent, ver_agent, gen_agent, agg_agent)

            mis_count = sum(len(g['smiles']) for g, _p in mis_groups)
            batch_stats.append({
                "batch": i + 1,
                "n_chunk": len(chunk),
                "n_to_generator": mis_count,
                "fraction_to_generator": (mis_count / float(len(chunk))) if len(chunk) else 0.0,
                "new_lines_added": 0 if stopped else added_lines_this_batch,
            })

        # ---------- Build the output: one row per input row ----------
        final_rows = dict(row_results)
        final_rows.update(aggregator_updates)
        mismatch_reason: Dict[int, str] = {}
        if hierarchy_mismatches:
            print(f"[Pipeline] Marking {len(hierarchy_mismatches)} reactions with HIERARCHY_MISMATCH")
            for pos, rec in hierarchy_mismatches:
                final_rows[pos] = _blank_row(f"HIERARCHY_MISMATCH:{rec['class_number']}")
                mismatch_reason[pos] = rec["reason"] or "Hierarchy mismatch detected"
        # Propagate along template identity: the agents classified a sample of each
        # cohort, and every reaction with the same template hash gets that class
        for g in all_chunks:
            src = g['indices'][0]
            for pos in g['all_indices']:
                if src in final_rows:
                    final_rows.setdefault(pos, final_rows[src])
                if src in mismatch_reason:
                    mismatch_reason.setdefault(pos, mismatch_reason[src])

        out = df.reset_index(drop=True)
        results_df = pd.DataFrame.from_dict(final_rows, orient="index", columns=LLM_COLUMNS).reindex(range(len(out)))
        for col in LLM_COLUMNS:
            out[col] = results_df[col].astype(object)
        if "classification" not in out.columns:
            out["classification"] = ""
        for pos, reason in mismatch_reason.items():
            out.at[pos, "classification"] = reason

        out_keep = out[[cfg.smiles_column] + LLM_COLUMNS + ["classification"]]
        output_csv_path = Path(cfg.output_csv)
        if stop_after != "hierarchy":
            out_keep.to_csv(output_csv_path, index=False)

        if hierarchy_mismatches:
            records = [rec for _pos, rec in hierarchy_mismatches]
            base = output_csv_path.parent / f"{output_csv_path.stem}_incorrect_hierarchy"
            base.with_suffix(".json").write_text(json.dumps(records, indent=2), encoding="utf-8")
            pd.DataFrame(records).to_csv(base.with_suffix(".csv"), index=False)
            print(f"[Pipeline] Wrote {len(records)} incorrect_hierarchy rows to {base}.json and .csv")

        # Per-batch stats, logged to W&B when a project is configured
        output_csv_path.with_suffix(".aggregator_stats.json").write_text(json.dumps(batch_stats, indent=2), encoding="utf-8")
        if cfg.wandb_project:
            import wandb

            run = wandb.init(project=cfg.wandb_project, name="classification-pipeline", reinit=True)
            for s in batch_stats:
                wandb.log({
                    "batch": s.get("batch"),
                    "fraction_to_generator": s.get("fraction_to_generator", 0.0),
                    "new_lines_added": s.get("new_lines_added", 0),
                }, step=s.get("batch"))
            run.finish()

        # Save all generator proposals
        if cfg.enable_verification and proposals:
            out_path = cfg.proposals_output_json or output_csv_path.with_suffix(output_csv_path.suffix + ".proposals.json")
            Path(out_path).write_text(json.dumps([p.model_dump() for p in proposals], indent=2), encoding="utf-8")
            # Optionally append the proposed hierarchy lines for manual curation
            if cfg.hierarchy_extensions_path:
                with open(cfg.hierarchy_extensions_path, "a", encoding="utf-8") as f:
                    for p in proposals:
                        for ln in p.additional_lines or []:
                            f.write(ln.strip() + "\n")
                        if p.hierarchy_line:
                            f.write(p.hierarchy_line.strip() + "\n")
        return out_keep
