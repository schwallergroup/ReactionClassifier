from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def _parse_rxn_smiles(rxn: str) -> Tuple[str, str]:
    s = str(rxn).strip()
    if ">>" in s:
        a, b = s.split(">>", 1)
        return a.strip(), b.strip()
    parts = s.split(">")
    if len(parts) == 3:
        return parts[0].strip(), parts[2].strip()
    raise ValueError("reaction smiles must be 'reactants>>products' or 'reactants>reagents>products'")


def _mol_or_none(smiles: str):
    from rdkit import Chem

    m = Chem.MolFromSmiles(smiles)
    return m


def _mixture_morgan_bits_or_and(smiles: str, *, radius: int, nbits: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    For a '.'-separated mixture, compute:
      - OR-combined Morgan bits (union)
      - AND-combined Morgan bits (intersection across fragments; if <=1 fragment, equals that fragment)

    Returns (or_bits, and_bits) as float32 0/1 vectors (shape [nbits]).
    """
    from rdkit.Chem import AllChem

    or_bits = np.zeros((nbits,), dtype=np.uint8)
    and_bits: Optional[np.ndarray] = None
    s = (smiles or "").strip()
    if not s:
        zeros = np.zeros((nbits,), dtype=np.uint8).astype(np.float32)
        return zeros, zeros
    for part in [p.strip() for p in s.split(".") if p.strip()]:
        m = _mol_or_none(part)
        if m is None:
            continue
        bv = AllChem.GetMorganFingerprintAsBitVect(m, int(radius), nBits=int(nbits))
        arr = np.zeros((nbits,), dtype=np.uint8)
        # RDKit fills into uint8 array.
        from rdkit import DataStructs

        DataStructs.ConvertToNumpyArray(bv, arr)
        or_bits |= arr
        and_bits = arr if and_bits is None else (and_bits & arr)

    if and_bits is None:
        and_bits = np.zeros((nbits,), dtype=np.uint8)
    return or_bits.astype(np.float32), and_bits.astype(np.float32)


@dataclass(frozen=True)
class FingerprintConfig:
    radius: int = 2
    nbits: int = 2048


def reaction_features(
    rxn_smiles: str,
    *,
    fp: FingerprintConfig,
    include_prod_fp: bool = True,
    include_diff_fp: bool = True,
    include_react_fp: bool = False,
) -> np.ndarray:
    """
    Returns concatenated feature vector:
      [diff_fp ; prod_fp ; react_fp]

    - prod_fp: OR-combined Morgan bits of products (mixture union)
    - diff_fp: OR(reactants) XOR OR(products)
    - react_fp: AND-combined Morgan bits of reactants (mixture intersection)
    """
    reactants, products = _parse_rxn_smiles(rxn_smiles)

    react_or, react_and = _mixture_morgan_bits_or_and(reactants, radius=fp.radius, nbits=fp.nbits)
    prod_or, _ = _mixture_morgan_bits_or_and(products, radius=fp.radius, nbits=fp.nbits)
    react_or_bits = react_or.astype(np.uint8)
    prod_bits = prod_or.astype(np.uint8)
    react_and_bits = react_and.astype(np.uint8)

    feats = []
    if include_diff_fp:
        feats.append((react_or_bits ^ prod_bits).astype(np.float32))
    if include_prod_fp:
        feats.append(prod_bits.astype(np.float32))
    if include_react_fp:
        feats.append(react_and_bits.astype(np.float32))
    if not feats:
        raise ValueError("At least one of include_diff_fp/include_prod_fp/include_react_fp must be True")
    return np.concatenate(feats, axis=0)
