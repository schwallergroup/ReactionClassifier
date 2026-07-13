# ReactionClassifier
Use this tool if you want to classify or name chemical reactions to the highest possible level of granularity with minimal dependencies!

Hierarchical neuro-symbolic reaction classification/reaction naming. Given a reaction SMILES, it
predicts a class in an LLM-derived reaction taxonomy and **confirms it
symbolically**: a Morgan difference–product (MDP) fingerprint MLP gate
proposes a class, the exact retrosynthetic templates in that class's tier-3
subtree are applied to the reaction, and a label is returned only if a template reproduces the recorded product

## Install

```bash
pip install reactionclassifier          # requires rdkit, torch, numpy
```

## Quickstart

```python
from reactionclassifier import ReactionClassifier

clf = ReactionClassifier()              # loads the bundled gate + templates + taxonomy
r = clf.classify("CC(=O)O.NCc1ccccc1>>CC(=O)NCc1ccccc1")
r.reaction_code     # '2.1.2.1'   (deterministically confirmed)
r.reaction_name     # 'Amidation using Carboxylic Acids | Primary Amine + Carboxylic Acid to Secondary Amide'
r.tier_path         # ['2.1', '2.1.2', '2.1.2.1']
r.confidence        # Top-1 probability of the neural layer
```

`classify()` returns a `ClassificationResult`:

| field | meaning |
|-------|---------|
| `reaction_code` | the **deterministically confirmed** class code — a template fired and reproduced the product. `None` if unconfirmed. |
| `reaction_name` | pipe-separated level 3/4/5 names of `reaction_code` (tiers 1-2 omitted). |
| `neural_code` / `neural_name` | the **neural-gate** prediction (always available); use as a fallback when `reaction_code` is `None`. Same name format. |
| `confidence` | neural-gate softmax confidence |
| `tier_path` | ancestor codes of `reaction_code` |

So: if `reaction_code` is populated you have a verified label; otherwise
`neural_code`/`neural_name` give the model's best (unverified) guess.

### Taxonomy and granularity examples

```python
from reactionclassifier import load_taxonomy, name_for, full_class_name, tier_path, load_granularity
load_taxonomy()["1.3.6"]      # -> single class name
name_for("2.1.2.1")           # -> single-level name
full_class_name("2.1.2.1")    # -> pipe-joined L3|L4|L5 names, e.g.
                              #    'Amidation using Carboxylic Acids | Primary Amine + Carboxylic Acid to Secondary Amide'
tier_path("1.3.6.2")          # -> ['1.3', '1.3.6', '1.3.6.2']
load_granularity()            # the two granularity-comparison tables
```

## What's included

| Component | File |
|-----------|------|
| MDP-gate MLP (full-data, 6,962 classes) | `data/gate/` |
| Exact rr0rp1_ring0 template library | `data/class_to_templates.json` |
| Full taxonomy (14,060 code→name) | `data/taxonomy.json` |
| Granularity examples (+ a small illustrative SMIRKS subset) | `data/granularity_examples*.json` |

## Full reaction database

The full labelled reaction database (≈666k reactions) is hosted externally
(Zenodo) rather than shipped in the wheel:

```python
from reactionclassifier.database import download_database
path = download_database()     # downloads + caches the parquet, returns its path
```

The released database **excludes NameRXN-derived columns** (`NAME`, `CLASS`);
NameRXN is proprietary and its labels are not redistributed.

## Detail

- The MDP fingerprint is a Morgan difference (reactant⊕product bit-unions)
  concatenated with the product fingerprint (RDKit, radius 2, 2048 bits each;
  4096-dim).
- The **full** generalised-SMIRKS library is not released; only the small subset
  embedded in the granularity examples is included.

## License

MIT (code and bundled data). See `LICENSE`.
