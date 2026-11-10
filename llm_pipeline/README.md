# rxn-classifier

An LLM pipeline that assigns chemical reactions to a class hierarchy and adds
new classes to that hierarchy when no existing class fits. Five agents run in
sequence: hierarchy, detailed, verifier, generator, aggregator.

## Install

```bash
pip install -e .            # add ".[rdkit]" to strip atom maps with RDKit
export GEMINI_API_KEY=...   # the agents call Gemini through google-genai
```

Weave tracing of LLM calls and Weights & Biases logging of batch statistics are
optional. Set `weave_project` and `wandb_project` in the config to turn them on.

## Run

```bash
rxn-classify --config examples/config.example.json
# or: python -m rxn_classifier.classification --config examples/config.example.json
```

From Python:

```python
from rxn_classifier.classification.api import run_from_config
df = run_from_config("examples/config.example.json")
```

## Input

A CSV with these columns (names are set in the config):

| Column | Required | Meaning |
| --- | --- | --- |
| `reaction_smiles` | yes | `reactants>reagents>products` |
| `template_hash` | yes | Key of the reaction template. Rows with the same key form one cohort. |
| `retro_template` | no | Retrosynthetic template, shown to the agents |

## How data flows

Rows are grouped into **cohorts** by `template_hash`. A cohort is classified as
one unit and every agent returns one answer per cohort. Cohorts are processed
in batches of `cohort_batch_size`.

The agents see at most `reactions_per_cohort` reactions of a cohort (default
5). A larger cohort is sampled at random, so the agents do not only see the
reactions that come first in the file, which often belong to one patent. The
draw is seeded by `sample_seed` and the template hash, so a cohort gets the
same sample in every run, however the rows are ordered.

The class hierarchy has two parts:

- **Levels 1 and 2** (`X.Y`, for example `2.1 N-acylation to amide`) are fixed.
  They are written in `prompts/hierarchy_classification.py`.
- **Level 3 and deeper** live in the mapping JSON (`mapping_json`), shaped as
  `{"2.1": ["2.1.1 Label", "2.1.1.1 Label", ...]}`. This part grows during a run.

For each batch:

1. **Hierarchy agent.** Sees the cohort's reactions and template. Returns one
   two-level code `X.Y`.
2. **Detailed agent.** Sees the cohort plus the subtree of the mapping under
   that `X.Y`. Returns the names of each level and a full class number such as
   `2.1.3.4`. If nothing in the subtree fits, it returns `X.Y.Other`.
   If it returns a bare `X.Y`, the pipeline rewrites it to `X.Y.Other`, so
   no cohort ends at level 2 (`detailed_agent.allow_level2_class: true` turns
   this off).
3. **Verifier agent.** Sees the cohort, the proposed class, and the subtree
   under the first two levels of the proposed class number. Returns
   `match` true or false with a reason. A matched cohort keeps its class and
   stops here.
4. **Generator agent.** Runs for each cohort the verifier rejected. Proposes a
   new class: parent code, new code, name, and any condition-level variants.
5. **Aggregator agent.** Proposals are grouped by their `X.Y` parent and sent
   in chunks of `max_cohorts_per_call`. For each chunk the aggregator merges
   duplicate proposals, returns the new hierarchy lines, and assigns a final
   code to each cohort. The pipeline then:
   - adds the new lines to the mapping straight away, so the next chunk and
     the next batch see them;
   - skips a new line whose code already exists with a different label, and
     marks cohorts assigned that code as `CONFLICT:<code>`;
   - replaces the detailed agent's class for the cohort with the assigned
     code, and writes the grown mapping to disk.

The cohort's final class is then given to every reaction with that template
hash, including those the agents did not see. The output has one row per input
row, in input order.

### Mapping versions

A run never overwrites the mapping it read. With `mapping_json` set to
`mapping.json`, the grown mapping is written to `mapping_v1.json`; a run that
reads `mapping_v1.json` writes `mapping_v2.json`, and so on. With no
`mapping_json`, the run starts from the two fixed levels and writes
`mapping_v1.json` beside the output CSV. Set
`aggregator_agent.dynamic_mapping_output_json` to choose the path yourself.

### Hierarchy mismatches

The verifier prompt also asks for an `<action>` tag, which is
`incorrect_hierarchy` when a cohort sits under the wrong `X.Y` branch. By
default this tag is recorded but not acted on, and every rejected cohort goes to
the generator. With `verifier_agent.exclude_hierarchy_mismatch: true`, flagged
cohorts skip the generator and are written as `HIERARCHY_MISMATCH:<code>`.

## Taxonomy files

| File | Content |
| --- | --- |
| `taxonomy/seed_level2.json` | The level-2 classes only, with nothing below them. Use it to grow a taxonomy from scratch. |
| `taxonomy/taxonomy.json` | The full taxonomy from our runs (14,060 entries). Use it to classify new reactions into the published classes, or to keep growing them. |

The loader also reads a flat `{"2.1.1": "Label", ...}` file, such as
`data/taxonomy.json` in the `reactionclassifier` package.

Set `mapping_json` to one of them. The example config uses the seed; to start
from the full taxonomy, change that line to `"taxonomy/taxonomy.json"`.

## Output

- `output_csv`: the SMILES column plus `LLM reaction super class`,
  `LLM reaction sub class`, `LLM reaction type`, `LLM reaction subtype`,
  `LLM class number`, and `classification`.
- The next mapping version (see above): the mapping with the classes added in this run.
- `<output>.aggregator_stats.json`: per batch, the share of reactions sent to
  the generator and the number of hierarchy lines added.
- `proposals_output_json` and `hierarchy_extensions_path`: every proposal made
  by the generator.
- With `test_mode.enabled`, the output of each stage for each batch is saved
  under `test_mode.output_dir`.

## Config

See `examples/config.example.json`. Top-level `llm_model`, `temperature` and
`thinking_budget` are defaults; each `*_agent` block can override them. The
verifier and generator are off unless their blocks set `"enabled": true`.

Strings in the config may use `{start_row}`, `{end_row}`, `{max_rows}`,
`{n_rows}` and `{prev_end}` (equal to `start_row`). This lets consecutive runs
over row ranges chain their mappings: read `mapping_{prev_end}/...`, write
`mapping_{end_row}/...`. `mapping_json_fallback` is used when `mapping_json`
does not exist yet.

Our runs used `gemini-2.5-flash` for the hierarchy and detailed agents,
`gemini-3-pro-preview` for the verifier and generator, and `gemini-2.5-pro` for
the aggregator. Google has since retired these names, so the example config
uses `gemini-3.8-flash` and `gemini-3.1-pro-preview`. Any Gemini model name
works.

`test_mode.start_stage` and `test_mode.stop_after_stage` start or stop a run at
a given stage, reading earlier results from `hierarchy_codes_path`,
`detailed_results_path` and `verify_results_path`.

## Tests

```bash
pip install -e ".[test]" && pytest
```

The test runs the whole pipeline with a scripted LLM and needs no API key.
