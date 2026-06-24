"""Hybrid-strict reaction classifier.

Pipeline: a Morgan difference--product (MDP) fingerprint MLP gate predicts a
class; the exact rr0rp1_ring0 templates in that class's tier-3 subtree are then
applied to the query reaction, and a label is returned only if one of them
reproduces the recorded product (otherwise the classifier abstains). This
preserves the determinism guarantee: a positive label always corresponds to a
template that actually fired.
"""
from __future__ import annotations

import json
from collections import defaultdict
from importlib import resources
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ._match import match_first
from .features import FingerprintConfig, reaction_features
from .model import MLPClassifier
from .taxonomy import name_for, tier_path


def _data(name: str):
    return resources.files("reactionclassifier.data").joinpath(name)


def _prefix(code: str, depth: int) -> str:
    return ".".join(str(code).split(".")[:depth])


class ReactionClassifier:
    """Load the bundled gate + template library + taxonomy and classify reactions.

    Example
    -------
    >>> clf = ReactionClassifier()
    >>> clf.classify("CC(=O)O.NCc1ccccc1>>CC(=O)NCc1ccccc1")["name"]
    """

    def __init__(self, device: str = "cpu", subset_tier: int = 3):
        self.device = torch.device(device)
        self.subset_tier = int(subset_tier)

        meta = json.loads(_data("gate/meta.json").read_text())
        self.fp_meta = meta["fp"]
        m = meta["model"]
        label_map = json.loads(_data("gate/label_map.json").read_text())
        self.id_to_label = {int(k): str(v) for k, v in label_map.items()}

        nbits = int(self.fp_meta.get("nbits", 2048))
        in_dim = nbits * (
            int(bool(self.fp_meta.get("include_diff_fp", True)))
            + int(bool(self.fp_meta.get("include_prod_fp", True)))
            + int(bool(self.fp_meta.get("include_react_fp", False)))
        )
        self.model = MLPClassifier(
            in_dim=in_dim,
            num_classes=int(meta["num_classes"]),
            hidden_dim=int(m["hidden_dim"]),
            depth=int(m["depth"]),
            dropout=float(m["dropout"]),
            activation=str(m["activation"]),
        ).to(self.device)
        with _data("gate/model.pt").open("rb") as fh:
            self.model.load_state_dict(torch.load(fh, map_location=self.device))
        self.model.eval()

        # exact-template library, indexed by tier-N prefix for fast subsetting
        c2t: Dict[str, List[str]] = json.loads(_data("class_to_templates.json").read_text())
        self._by_prefix: Dict[str, List[Tuple[str, str, int]]] = defaultdict(list)
        for code, templates in c2t.items():
            pref = _prefix(code, self.subset_tier)
            for t in templates:
                nreact = len(t.split(">>")[0].split("."))
                self._by_prefix[pref].append((code, t, nreact))

    # -- gate ---------------------------------------------------------------
    def _featurize(self, rxn: str):
        fp = FingerprintConfig(
            radius=int(self.fp_meta.get("radius", 2)),
            nbits=int(self.fp_meta.get("nbits", 2048)),
        )
        return reaction_features(
            rxn,
            fp=fp,
            include_diff_fp=bool(self.fp_meta.get("include_diff_fp", True)),
            include_prod_fp=bool(self.fp_meta.get("include_prod_fp", True)),
            include_react_fp=bool(self.fp_meta.get("include_react_fp", False)),
        )

    def gate(self, rxn: str) -> Tuple[Optional[str], float]:
        """Return the MDP-gate's (class_code, softmax_confidence)."""
        try:
            feat = self._featurize(rxn)
        except Exception:
            return None, 0.0
        if feat is None:
            return None, 0.0
        x = torch.from_numpy(feat).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = F.softmax(self.model(x), dim=-1)
        p, idx = probs.max(dim=1)
        return self.id_to_label.get(int(idx[0].item())), float(p[0].item())

    # -- full classify ------------------------------------------------------
    def classify(self, rxn: str) -> Dict:
        """Classify a reaction SMILES (``reactants>>products`` or
        ``reactants>reagents>products``). Returns a dict with the matched class
        ``code``/``name``/``tier_path``, whether a template ``matched`` (else the
        classifier abstains), and the underlying ``gate_code``/``confidence``."""
        gate_code, conf = self.gate(rxn)
        code = None
        if gate_code:
            records = self._by_prefix.get(_prefix(gate_code, self.subset_tier), [])
            if records:
                code = match_first(rxn, records)
        return {
            "code": code,
            "name": name_for(code),
            "tier_path": tier_path(code),
            "matched": code is not None,
            "gate_code": gate_code,
            "confidence": round(conf, 4),
        }
