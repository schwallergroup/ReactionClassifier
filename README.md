# reactionclassifier

Hybrid-strict hierarchical reaction classification. Given a reaction SMILES, it
predicts a class in an LLM-derived reaction taxonomy and **confirms it
deterministically**: a Morgan difference–product (MDP) fingerprint MLP gate
proposes a class, the exact retrosynthetic templates in that class's tier-3
subtree are applied to the reaction, and a label is returned only if a template
actually reproduces the recorded product (otherwise the classifier abstains).

## Install

```bash
pip install reactionclassifier          # requires rdkit, torch, numpy
```

## Quickstart

```python
from reactionclassifier import ReactionClassifier

clf = ReactionClassifier()              # loads the bundled gate + templates + taxonomy
out = clf.classify("CC(=O)O.NCc1ccccc1>>CC(=O)NCc1ccccc1")
# {'code': '2.1.2.x', 'name': '...', 'tier_path': [...],
#  'matched': True, 'gate_code': '...', 'confidence': 0.99}
```

`matched=False` means the deterministic layer abstained (no template fired) —
the classifier does not emit a label it cannot verify.

### Taxonomy and granularity examples

```python
from reactionclassifier import load_taxonomy, name_for, tier_path, load_granularity
load_taxonomy()["1.3.6"]      # -> human-readable class name
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

## Provenance & scope

- The MDP fingerprint is a Morgan difference (reactant⊕product bit-unions)
  concatenated with the product fingerprint (RDKit, radius 2, 2048 bits each;
  4096-dim). It is DRFP-inspired but does **not** use the `drfp` package.
- The **full** generalised-SMIRKS library is not released; only the small subset
  embedded in the granularity examples is included.

## License

MIT (code and bundled data). See `LICENSE`.
