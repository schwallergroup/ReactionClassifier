"""Runs all five agents end to end with a scripted LLM, so no API key is needed."""

import json

import pandas as pd

from rxn_classifier.classification import pipeline as pipeline_module
from rxn_classifier.classification.llm_client import BaseLLMClient, LLMResponse
from rxn_classifier.classification.models import PipelineConfig

REPLIES = {
    "hierarchy": "<reaction_class>2.1</reaction_class>",
    "detailed": (
        "<reaction_super_class>Acylation and related processes</reaction_super_class>"
        "<reaction_sub_class>N-acylation to amide</reaction_sub_class>"
        "<reaction_type></reaction_type><reaction_subtype></reaction_subtype>"
        "<class_number>2.1.Other</class_number>"
    ),
    "verifier": "<match>false</match><cause>core_transformation</cause>"
    "<action>continue</action><reason>Class is Other.</reason>",
    "generator": (
        "<rationale>Acid plus amine.</rationale>"
        "<suggested_parent_code>2.1</suggested_parent_code>"
        "<new_entry_code>2.1.1</new_entry_code>"
        "<new_entry_name>Amidation Of Carboxylic Acids</new_entry_name>"
        "<hierarchy_line>2.1.1 Amidation Of Carboxylic Acids</hierarchy_line>"
        "<new_parent_lines>2.1.1.1 cond: EDC coupling</new_parent_lines>"
    ),
    "aggregator": (
        "<new_hierarchy>\n2.1.1 Amidation Of Carboxylic Acids\n2.1.1.1 cond: EDC coupling\n</new_hierarchy>"
        '<assignments>{"0": "2.1.1", "1": "2.1.1"}</assignments>'
    ),
}


class ScriptedLLM(BaseLLMClient):
    def generate(self, system_prompt: str, user_content: str) -> LLMResponse:
        return LLMResponse(text=REPLIES[self.client_tag])


def _scripted_client(model, temperature, thinking_budget, log_io, save_io_dir, client_tag="", fail_on_noop=False):
    return ScriptedLLM(model=model, client_tag=client_tag)


def _write_input(tmp_path, smiles, hashes):
    input_csv = tmp_path / "in.csv"
    pd.DataFrame({"reaction_smiles": smiles, "template_hash": hashes}).to_csv(input_csv, index=False)
    return input_csv


def test_full_flow(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_module, "_make_llm_client", _scripted_client)
    seed = tmp_path / "mapping.json"
    seed.write_text(json.dumps({"3.1": ["3.1.1 Bromo Suzuki coupling"]}))
    config = PipelineConfig(
        input_csv=_write_input(
            tmp_path,
            ["CC(=O)O.NCC>>CC(=O)NCC", "OC(=O)CC.NC>>CCC(=O)NC", "OC(=O)C.NCCC>>CC(=O)NCCC"],
            ["a", "a", "b"],
        ),
        output_csv=tmp_path / "out.csv",
        llm_model="scripted",
        mapping_json=seed,
        verifier_agent={"enabled": True},
        generator_agent={"enabled": True},
    )

    out = pipeline_module.ClassificationPipeline(config=config).run()

    # Detailed said "2.1.Other", the verifier rejected it, the generator proposed
    # 2.1.1, and the aggregator assigned it to both cohorts.
    assert out["LLM class number"].tolist() == ["2.1.1", "2.1.1", "2.1.1"]
    assert out["LLM reaction type"].tolist() == ["Amidation Of Carboxylic Acids"] * 3
    assert pd.read_csv(tmp_path / "out.csv")["LLM class number"].tolist() == ["2.1.1"] * 3
    # The grown mapping goes to the next version; the seed is left alone.
    assert json.loads((tmp_path / "mapping_v1.json").read_text()) == {
        "2.1": ["2.1.1 Amidation Of Carboxylic Acids", "2.1.1.1 cond: EDC coupling"],
        "3.1": ["3.1.1 Bromo Suzuki coupling"],
    }
    assert json.loads(seed.read_text()) == {"3.1": ["3.1.1 Bromo Suzuki coupling"]}


def test_duplicate_smiles_keep_one_row_each(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_module, "_make_llm_client", _scripted_client)
    rxn = "CC(=O)O.NCC>>CC(=O)NCC"
    config = PipelineConfig(
        input_csv=_write_input(tmp_path, [rxn, rxn, rxn], ["a", "a", "b"]),
        output_csv=tmp_path / "out.csv",
        llm_model="scripted",
    )
    out = pipeline_module.ClassificationPipeline(config=config).run()
    assert len(out) == 3
    assert out["LLM class number"].tolist() == ["2.1.Other"] * 3


def test_hierarchy_mismatch_switch(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_module, "_make_llm_client", _scripted_client)
    monkeypatch.setitem(
        REPLIES, "verifier", "<match>false</match><action>incorrect_hierarchy</action><reason>Wrong branch.</reason>"
    )
    config = PipelineConfig(
        input_csv=_write_input(tmp_path, ["CC(=O)O.NCC>>CC(=O)NCC"], ["a"]),
        output_csv=tmp_path / "out.csv",
        llm_model="scripted",
        verifier_agent={"enabled": True, "exclude_hierarchy_mismatch": True},
        generator_agent={"enabled": True},
    )
    out = pipeline_module.ClassificationPipeline(config=config).run()
    assert out["LLM class number"].tolist() == ["HIERARCHY_MISMATCH:2.1.Other"]
    assert out["classification"].tolist() == ["Wrong branch."]


def test_large_cohort_is_sampled_and_all_rows_get_the_class(tmp_path, monkeypatch):
    shown = []

    class Recording(ScriptedLLM):
        def generate(self, system_prompt, user_content):
            if self.client_tag == "detailed":
                shown.append([ln for ln in system_prompt.splitlines() if ln.startswith("reactants: ")])
            return super().generate(system_prompt, user_content)

    monkeypatch.setattr(
        pipeline_module, "_make_llm_client", lambda model, *a, client_tag="", **k: Recording(model=model, client_tag=client_tag)
    )
    smiles = [f"CC(=O)O.N{'C' * (i + 1)}>>CC(=O)N{'C' * (i + 1)}" for i in range(30)]

    def run(order, seed):
        shown.clear()
        config = PipelineConfig(
            input_csv=_write_input(tmp_path, [smiles[i] for i in order], ["big"] * 30),
            output_csv=tmp_path / "out.csv",
            llm_model="scripted",
            cohort_mode={"reactions_per_cohort": 5, "sample_seed": seed},
        )
        out = pipeline_module.ClassificationPipeline(config=config).run()
        return out, sorted(shown[0])

    out, sample = run(range(30), 0)
    # Five reactions were shown, and they are not simply the first five rows.
    assert len(sample) == 5
    assert sample != sorted(f"reactants: CC(=O)O.N{'C' * (i + 1)}" for i in range(5))
    # All 30 rows carry the cohort's class.
    assert out["LLM class number"].tolist() == ["2.1.Other"] * 30

    # Same seed, same sample, even with the input rows reversed. Another seed differs.
    assert run(range(30), 0)[1] == sample
    assert run(list(reversed(range(30))), 0)[1] == sample
    assert run(range(30), 1)[1] != sample
    # Null seed takes the first rows, as the research code did.
    assert run(range(30), None)[1] == sorted(f"reactants: CC(=O)O.N{'C' * (i + 1)}" for i in range(5))


def test_bare_level2_answer_becomes_other(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_module, "_make_llm_client", _scripted_client)
    monkeypatch.setitem(REPLIES, "detailed", REPLIES["detailed"].replace("2.1.Other", "2.1"))

    def run(**detailed):
        config = PipelineConfig(
            input_csv=_write_input(tmp_path, ["CC(=O)O.NCC>>CC(=O)NCC"], ["a"]),
            output_csv=tmp_path / "out.csv",
            llm_model="scripted",
            detailed_agent=detailed,
        )
        return pipeline_module.ClassificationPipeline(config=config).run()["LLM class number"].tolist()

    # The detailed agent answered "2.1"; the pipeline holds it to "2.1.Other".
    assert run() == ["2.1.Other"]
    assert run(allow_level2_class=True) == ["2.1"]


def test_flat_taxonomy_file_loads(tmp_path):
    from rxn_classifier.classification.utils import load_mapping

    flat = tmp_path / "flat.json"
    flat.write_text(json.dumps({"2.1": "N-acylation to amide", "2.1.1": "Amidation", "3.1.1": "Suzuki coupling"}))
    assert load_mapping(flat) == {
        "2.1": ["2.1 N-acylation to amide", "2.1.1 Amidation"],
        "3.1": ["3.1.1 Suzuki coupling"],
    }
