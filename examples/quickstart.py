"""Minimal usage example."""
from reactionclassifier import ReactionClassifier, name_for, tier_path

clf = ReactionClassifier()

reactions = [
    "CC(=O)O.NCc1ccccc1>>CC(=O)NCc1ccccc1",          # amide coupling
    "OCC1CNC1.COc1cnc(Cl)cc1>>COc1cnc(N2CC(CO)C2)cc1",  # SNAr / N-arylation
]
for rxn in reactions:
    out = clf.classify(rxn)
    print(rxn)
    if out["matched"]:
        print(f"  -> {out['code']}  {out['name']}  (conf {out['confidence']})")
        print(f"     tiers: {out['tier_path']}")
    else:
        print(f"  -> abstained (gate guessed {out['gate_code']}, no template confirmed)")
    print()
