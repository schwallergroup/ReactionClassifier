"""Forward template matching via RDKit RunReactants (Hybrid-strict confirmation).

Ported and trimmed from the project's evaluation matcher: a template "fires" if,
applied to the reactants, it reproduces the recorded product (non-isomeric
canonical comparison). First template that fires wins.
"""
from __future__ import annotations

import functools
import itertools
from typing import Iterable, Optional, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")


@functools.lru_cache(maxsize=50000)
def _compile(smirks: str):
    try:
        return AllChem.ReactionFromSmarts(smirks)
    except Exception:
        return None


def _split_reaction(rxn: str) -> Optional[Tuple[list, set]]:
    if ">>" in rxn:
        left, right = rxn.split(">>", 1)
    else:
        parts = rxn.split(">")
        if len(parts) < 3:
            return None
        left, right = parts[0], parts[-1]
    reactants = [r for r in left.split(".") if r]
    products = [p for p in right.split(".") if p]
    if not reactants or not products or len(reactants) > 4 or len(products) > 4:
        return None
    react_mols = [Chem.MolFromSmiles(r) for r in reactants]
    if any(m is None for m in react_mols):
        return None
    expected = set()
    for p in products:
        m = Chem.MolFromSmiles(p)
        expected.add(Chem.MolToSmiles(m, isomericSmiles=False) if m is not None else p)
    return react_mols, expected


def match_first(rxn: str, records: Iterable[Tuple[str, str, int]]) -> Optional[str]:
    """``records`` = iterable of ``(class_code, smirks, nreact)``. Returns the
    class code of the first template whose product matches ``rxn``'s recorded
    product, else ``None``."""
    parsed = _split_reaction(rxn)
    if parsed is None:
        return None
    react_mols, expected = parsed
    n = len(react_mols)
    for code, smirks, nreact in records:
        if nreact > n:
            continue
        robj = _compile(smirks)
        if robj is None:
            continue
        if nreact == n:
            tuples = [tuple(react_mols)] if n == 1 else list(itertools.permutations(react_mols))
        else:
            tuples = []
            for sub in itertools.combinations(react_mols, nreact):
                tuples += [sub] if nreact == 1 else list(itertools.permutations(sub))
        for tup in tuples:
            try:
                outcomes = robj.RunReactants(tup)
            except Exception:
                continue
            for prods in outcomes:
                try:
                    smi = Chem.MolToSmiles(prods[0], isomericSmiles=False)
                except Exception:
                    continue
                if smi in expected:
                    return code
    return None
