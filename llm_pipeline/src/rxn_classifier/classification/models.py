from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------- Records passed between stages ----------

class DetailedResult(BaseModel):
    reaction_smiles: str
    reaction_super_class: str = ""
    reaction_sub_class: str = ""
    reaction_type: str = ""
    reaction_subtype: str = ""
    class_number: str = ""

    def to_row(self, smiles_column: str) -> Dict[str, Any]:
        return {
            smiles_column: self.reaction_smiles,
            "LLM reaction super class": self.reaction_super_class,
            "LLM reaction sub class": self.reaction_sub_class,
            "LLM reaction type": self.reaction_type,
            "LLM reaction subtype": self.reaction_subtype,
            "LLM class number": self.class_number,
        }


class VerificationResult(BaseModel):
    reaction_smiles: str
    class_number: str = ""
    match: bool = False
    reason: str = ""
    action: str = ""


class ClassProposal(BaseModel):
    reaction_smiles: str
    suggested_parent_code: str = ""
    new_entry_code: str = ""
    new_entry_name: str = ""
    hierarchy_line: str = ""
    rationale: str = ""
    additional_lines: List[str] = []


# ---------- Per-agent config ----------

class AgentBaseConfig(BaseModel):
    model: Optional[str] = Field(None, description="Override model for this agent.")
    temperature: Optional[float] = Field(None, description="Override temperature for this agent.")
    thinking_budget: Optional[int] = Field(
        None, description="Override thinking budget for this agent.", ge=0
    )
    log_io: Optional[bool] = Field(None, description="Print full LLM IO for this agent.")
    save_io_dir: Optional[Path] = Field(None, description="Directory to save LLM IO for this agent.")
    strip_mapping: Optional[bool] = Field(
        None,
        description="If true, remove atom mapping numbers from reactants/products before sending to the LLM.",
    )


class HierarchyAgentConfig(AgentBaseConfig):
    pass


class DetailedAgentConfig(AgentBaseConfig):
    enabled: bool = Field(True, description="Enable detailed agent stage.")
    allow_level2_class: bool = Field(
        False,
        description=(
            "If false, a bare two-level answer such as '2.1' is rewritten to '2.1.Other', "
            "so the verifier rejects it and the generator proposes a deeper class."
        ),
    )


class VerifierAgentConfig(AgentBaseConfig):
    enabled: bool = Field(False, description="Enable verification stage.")
    exclude_hierarchy_mismatch: bool = Field(
        False,
        description=(
            "If true, cohorts the verifier flags with <action>incorrect_hierarchy</action> are not sent "
            "to the generator and are written as HIERARCHY_MISMATCH:<code>. Off by default: the runs "
            "behind the published results sent every rejected cohort to the generator."
        ),
    )


class GeneratorAgentConfig(AgentBaseConfig):
    enabled: bool = Field(False, description="Enable generator stage (requires verifier).")


class AggregatorAgentConfig(AgentBaseConfig):
    enabled: bool = Field(True, description="Enable aggregator to merge proposals and assign codes.")
    dynamic_mapping_output_json: Optional[Path] = Field(
        None,
        description=(
            "Path to write the updated mapping JSON. If unset, the next version of mapping_json is "
            "written beside it (mapping.json -> mapping_v1.json, mapping_v1.json -> mapping_v2.json)."
        ),
    )
    max_cohorts_per_call: int = Field(
        50,
        ge=1,
        description="Maximum number of cohorts to include in a single aggregator call per parent group.",
    )


# ---------- Run modes ----------

class TestModeConfig(BaseModel):
    enabled: bool = Field(False, description="Enable test mode to save intermediate artifacts.")
    output_dir: Path = Field(Path("artifacts"), description="Directory to store intermediate artifacts.")
    # Optional per-stage subdirectories (each defaults to output_dir)
    hierarchy_dir: Optional[Path] = None
    detailed_dir: Optional[Path] = None
    verify_dir: Optional[Path] = None
    proposals_dir: Optional[Path] = None
    aggregator_dir: Optional[Path] = None
    start_stage: str = Field(
        "hierarchy",
        description="Stage to start from: 'hierarchy' | 'detailed' | 'verify' | 'generate'.",
    )
    # Precomputed inputs to resume from a later stage
    hierarchy_codes_path: Optional[Path] = Field(
        None, description="Two-level codes (JSON or CSV), needed when start_stage != 'hierarchy'."
    )
    detailed_results_path: Optional[Path] = Field(
        None, description="Saved detailed results, needed when start_stage is 'verify' or 'generate'."
    )
    verify_results_path: Optional[Path] = Field(
        None, description="Saved verify results, needed when start_stage is 'generate'."
    )
    hierarchy_only_all_in_one: bool = Field(
        False, description="Run only the hierarchy agent over all cohorts, save a CSV, then exit."
    )
    hierarchy_only_output_csv: Optional[Path] = Field(
        None, description="Output of hierarchy-only mode (default: output_dir/hierarchy_only.csv)."
    )
    stop_after_stage: Optional[str] = Field(
        None, description="Stop each batch after this stage: 'hierarchy' | 'detailed' | 'verify'."
    )


class CohortModeConfig(BaseModel):
    template_hash_column: str = Field(
        "template_hash", description="Column name containing the cohort key (template hash)."
    )
    cohort_batch_size: int = Field(
        200, ge=1, description="Number of cohorts per pipeline batch."
    )
    max_cohorts_per_llm_job: int = Field(
        200, ge=1, description="Maximum number of cohorts submitted to an agent in one go."
    )
    reactions_per_cohort: int = Field(
        5,
        ge=1,
        description=(
            "Number of reactions per cohort shown to the agents. The class they agree on is then "
            "given to every reaction in the cohort."
        ),
    )
    sample_seed: Optional[int] = Field(
        0,
        description=(
            "Seed for sampling the shown reactions from cohorts larger than reactions_per_cohort. "
            "Null takes the first reactions_per_cohort rows instead."
        ),
    )


# ---------- Pipeline config ----------

class PipelineConfig(BaseModel):
    input_csv: Path = Field(..., description="Input CSV with reaction SMILES and template hash columns.")
    output_csv: Path = Field(..., description="Path to write the classification CSV.")
    llm_model: str = Field(..., description="Default LLM model (e.g., models/gemini-2.5-flash).")

    start_row: Optional[int] = Field(None, ge=0, description="0-based inclusive start row of the input slice.")
    end_row: Optional[int] = Field(None, ge=0, description="0-based exclusive end row of the input slice.")
    max_rows: Optional[int] = Field(None, description="Maximum number of rows, applied after the slice.")
    smiles_column: str = Field("reaction_smiles", description="Name of the reaction SMILES column.")
    template_column: str = Field("retro_template", description="Optional column with the retro template.")

    temperature: Optional[float] = Field(0.1, description="Default LLM temperature.")
    thinking_budget: Optional[int] = Field(
        None, ge=0, description="Default Gemini thinking budget (0 disables, unset = dynamic)."
    )

    mapping_json: Optional[Path] = Field(
        None, description="Mapping from two-level code (e.g. 2.1) to its deeper hierarchy lines."
    )
    mapping_json_fallback: Optional[Path] = Field(
        None, description="Used when mapping_json is set but does not exist yet."
    )

    enable_verification: bool = Field(
        False,
        description=(
            "Runs the verifier when there is no verifier_agent block, and writes the proposals file "
            "at the end of the run."
        ),
    )
    proposals_output_json: Optional[Path] = Field(
        None, description="Where to write all class proposals (default: <output_csv>.proposals.json)."
    )
    hierarchy_extensions_path: Optional[Path] = Field(
        None, description="Text file to append proposed hierarchy lines to."
    )

    log_llm_io: bool = Field(False, description="Print full LLM prompts and responses.")
    llm_io_save_dir: Optional[Path] = Field(None, description="Save per-call LLM inputs/outputs here.")
    weave_project: Optional[str] = Field(
        None, description="Weave project to trace LLM calls to. Unset = no Weave tracing."
    )
    wandb_project: Optional[str] = Field(
        None, description="Weights & Biases project for batch stats. Unset = no W&B logging."
    )
    fail_on_noop: bool = Field(
        False,
        description="If true, abort instead of falling back to the offline stub when LLM setup fails.",
    )

    hierarchy_agent: Optional[HierarchyAgentConfig] = None
    detailed_agent: Optional[DetailedAgentConfig] = None
    verifier_agent: Optional[VerifierAgentConfig] = None
    generator_agent: Optional[GeneratorAgentConfig] = None
    aggregator_agent: Optional[AggregatorAgentConfig] = None
    test_mode: Optional[TestModeConfig] = None
    cohort_mode: Optional[CohortModeConfig] = None

    @field_validator("input_csv")
    @classmethod
    def _validate_input_csv(cls, v: Path) -> Path:
        if not Path(v).exists():
            raise ValueError(f"input_csv does not exist: {v}")
        return v
