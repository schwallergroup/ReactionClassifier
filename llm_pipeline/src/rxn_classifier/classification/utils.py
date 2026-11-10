from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from rxn_classifier.classification.generate_xx_mapping import (
    default_input_path,
    extract_hierarchy_text,
    parse_hierarchy_lines,
    build_xx_mapping,
)


def format_rxn_roles(rxn: str) -> str:
    """Format reaction SMILES into role-labeled, multi-line text for prompts.

    Input is typically "reactants>reagents>products". Missing segments are allowed.
    """
    parts = (rxn or "").split(">")
    reactants = parts[0].strip() if len(parts) > 0 else ""
    reagents = parts[1].strip() if len(parts) > 1 else ""
    product = parts[2].strip() if len(parts) > 2 else ""
    return f"reactants: {reactants}\nreagents: {reagents}\nproduct: {product}"


def _strip_mapping_species_block(block: str) -> str:
    """Remove atom-mapping numbers from a dot-delimited species block using RDKit when available.

    Falls back to removing mapping annotations like ":<digits>" within bracketed atoms via regex if RDKit is not available.
    """
    if not block:
        return block
    try:
        from rdkit import Chem  # type: ignore
    except Exception:
        import re as _re
        return _re.sub(r":\d+", "", block)

    out_species: List[str] = []
    for smi in block.split('.'):
        smi = smi.strip()
        if not smi:
            continue
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                out_species.append(smi)
                continue
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            out_species.append(Chem.MolToSmiles(mol))
        except Exception:
            out_species.append(smi)
    return '.'.join(out_species)


def strip_atom_mapping_reactants_products(rxn_smiles: str) -> str:
    """Remove atom mapping from reactants and products in a reaction SMILES (reactants>reagents>products).

    The reagents block (middle) is left unchanged.
    """
    if not rxn_smiles:
        return rxn_smiles
    parts = rxn_smiles.split('>')
    if len(parts) == 0:
        return rxn_smiles
    reactants = _strip_mapping_species_block(parts[0].strip()) if len(parts) > 0 else ""
    reagents = parts[1].strip() if len(parts) > 1 else ""
    products = _strip_mapping_species_block(parts[2].strip()) if len(parts) > 2 else ""
    return '>'.join([reactants, reagents, products])


def format_rxn_roles_with_strip(rxn: str, strip_mapping: bool = False) -> str:
    """Role-labeled formatting with optional atom-mapping removal for reactants/products."""
    if strip_mapping:
        rxn = strip_atom_mapping_reactants_products(rxn)
    return format_rxn_roles(rxn)


def load_mapping(mapping_json: Optional[Path]) -> Optional[Dict[str, List[str]]]:
    """Load a mapping for X.X -> descendants (list of "code label" strings).

    Supports multiple shapes:
    - {"2.1": ["2.1.1 Label", "2.1.1.1 Label", ...]}
    - {"2.1": {"2.1.1": "Label", "2.1.1.1": "Label", ...}}
    - {"2.1": {"descendants": ["2.1.1 Label", ...] or {"2.1.1": "Label", ...}}}
    - {"2.1": "Label", "2.1.1": "Label", ...} (flat code -> name)
    Other keys are ignored. Returns None if parsing fails.
    """
    if not mapping_json:
        return None
    p = Path(mapping_json)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

    def _flatten_value(val) -> List[str]:
        # Already a list: normalize to strings
        if isinstance(val, list):
            out: List[str] = []
            for x in val:
                if isinstance(x, str):
                    out.append(x)
                elif isinstance(x, dict):
                    code = str(x.get("code", "")).strip()
                    label = str(x.get("label", x.get("name", ""))).strip()
                    line = x.get("line")
                    if isinstance(line, str) and line.strip():
                        out.append(line.strip())
                    elif code:
                        out.append(f"{code} {label}".strip())
            return out
        # A string blob with multiple lines (like a pasted subtree)
        if isinstance(val, str):
            import re as _re
            out: List[str] = []
            for raw in val.splitlines():
                line = raw.strip()
                if not line:
                    continue
                m = _re.match(r"^(\d+(?:\.\d+)+)\s+(.*\S)\s*$", line)
                if m:
                    out.append(f"{m.group(1)} {m.group(2)}")
            return out
        # A dict: could be {code: label} or {descendants: ...}
        if isinstance(val, dict):
            if "descendants" in val:
                return _flatten_value(val["descendants"])
            out: List[str] = []
            for ck, cv in val.items():
                if not isinstance(ck, str):
                    continue
                if isinstance(cv, str):
                    out.append(f"{ck} {cv}".strip())
                elif isinstance(cv, dict):
                    label = str(cv.get("label", cv.get("name", ""))).strip()
                    line = cv.get("line")
                    if isinstance(line, str) and line.strip():
                        out.append(line.strip())
                    else:
                        out.append(f"{ck} {label}".strip())
            return out
        return []

    # Flat shape {"2.1": "Label", "2.1.1": "Label", ...}: group lines under their X.Y code
    code_re = re.compile(r"^\d+(?:\.\d+)+$")
    if isinstance(data, dict) and data and all(
        isinstance(k, str) and code_re.match(k) and isinstance(v, str) and "\n" not in v
        and not re.match(r"^\d+\.\d+", v.strip())
        for k, v in data.items()
    ):
        flat: Dict[str, List[str]] = {}
        for k, v in data.items():
            flat.setdefault(".".join(k.split(".")[:2]), []).append(f"{k} {v}".strip())
        return flat

    cleaned: Dict[str, List[str]] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            if not isinstance(k, str):
                continue
            lines = _flatten_value(v)
            if lines:
                cleaned[k] = lines
    return cleaned or None


def build_hierarchy_subtree(code_two_level: str, mapping: Optional[Dict[str, List[str]]] = None) -> str:
    """Return lines for the given two-level code through its descendants.

    If mapping is None, rebuild from the prompt file. Includes two-level header line with label if available.
    """
    code = (code_two_level or "").strip()
    # Always build a code->label lookup from the prompt file
    try:
        hierarchy_text = extract_hierarchy_text(default_input_path())
        items = parse_hierarchy_lines(hierarchy_text)
        code2label = {c: lbl for c, lbl in items}
    except Exception:
        items = []
        code2label = {}

    # If provided mapping is missing or doesn't contain the code, fall back to full prompt-derived mapping
    needs_fallback = mapping is None or not mapping.get(code)
    if needs_fallback:
        try:
            prompt_mapping = build_xx_mapping(items)
        except Exception:
            prompt_mapping = {}
        # Merge prompt_mapping into provided mapping (if any), giving priority to provided mapping
        merged: Dict[str, List[str]] = {}
        if isinstance(prompt_mapping, dict):
            merged.update(prompt_mapping)
        if isinstance(mapping, dict):
            merged.update(mapping)
        mapping = merged

    header = code
    if code in code2label:
        header = f"{code} {code2label[code]}"
    lines = [header]
    for entry in (mapping.get(code, []) if mapping else []):
        lines.append(entry)
    return "\n".join(lines)


def next_mapping_path(mapping_json: Optional[Path], output_csv: Path) -> Path:
    """Path of the next mapping version, so a run never overwrites the mapping it read.

    mapping.json -> mapping_v1.json, mapping_v3.json -> mapping_v4.json.
    With no input mapping, the first version is written beside the output CSV.
    """
    import re as _re

    if not mapping_json:
        return Path(output_csv).parent / "mapping_v1.json"
    p = Path(mapping_json)
    m = _re.match(r"^(.*)_v(\d+)$", p.stem)
    if m:
        return p.with_name(f"{m.group(1)}_v{int(m.group(2)) + 1}{p.suffix}")
    return p.with_name(f"{p.stem}_v1{p.suffix}")


def apply_lines_to_mapping_dict(lines: List[str], data: Optional[dict]) -> tuple[dict, int, Dict[str, str]]:
    """Return an updated mapping dict with the given lines appended under their two-level parents.

    Does not write to disk. Returns (updated_dict, added_count, conflicts).
    - conflicts: dict mapping conflicting codes to their existing labels (code -> existing_label)

    The dict follows the same normalized shape as load_mapping(): {"2.1": ["2.1.1 Label", ...], ...}
    This function mirrors the logic used in apply_lines_to_mapping_json so that
    in-memory dynamic_mapping behaves identically to the JSON representation.

    Conflict detection: If a new line has the same code as an existing line but a different
    label, it is considered a conflict. The new line is NOT added, and the conflict is recorded.
    """
    import re as _re

    if data is None or not isinstance(data, dict):
        data = {}
    # Work on a shallow copy to avoid mutating the caller's dict in-place
    mapping = dict(data)

    # Build a global code -> label lookup for conflict detection
    # This checks across ALL parents, not just the current one
    global_code_to_label: Dict[str, str] = {}
    for parent_key, entries in mapping.items():
        if isinstance(entries, list):
            for entry in entries:
                entry_str = (entry or "").strip()
                if not entry_str:
                    continue
                m = _re.match(r"^(\d+(?:\.\d+)+)\s+(.*)$", entry_str)
                if m:
                    global_code_to_label[m.group(1)] = m.group(2).strip()

    added = 0
    conflicts: Dict[str, str] = {}  # code -> existing_label

    for raw in (lines or []):
        line = (raw or "").strip()
        if not line:
            continue
        # Extract the leading code X.Y[.Z...] and derive two-level parent
        m = _re.match(r"^(\d+(?:\.\d+)+)\s+(.*)$", line)
        if not m:
            continue
        full_code = m.group(1)
        new_label = m.group(2).strip()
        parts = full_code.split(".")
        if len(parts) < 2:
            continue

        # Check for code conflict globally
        if full_code in global_code_to_label:
            existing_label = global_code_to_label[full_code]
            if existing_label != new_label:
                # CONFLICT: same code, different label
                try:
                    print(f"[UTILS][CONFLICT] Code {full_code} already exists with different label")
                    print(f"[UTILS][CONFLICT]   Existing: {existing_label}")
                    print(f"[UTILS][CONFLICT]   New:      {new_label}")
                except Exception:
                    pass
                conflicts[full_code] = existing_label
                continue  # Skip adding this conflicting line

        parent_two = ".".join(parts[:2])
        existing = mapping.get(parent_two)
        if isinstance(existing, list):
            if line not in existing:
                existing.append(line)
                added += 1
                global_code_to_label[full_code] = new_label  # Track for subsequent lines
            mapping[parent_two] = existing
        elif isinstance(existing, dict):
            # Convert mapping-dict to list of "code label" if needed
            vals = []
            for ck, cv in existing.items():
                if isinstance(cv, str):
                    vals.append(f"{ck} {cv}".strip())
            if line not in vals:
                vals.append(line)
                added += 1
                global_code_to_label[full_code] = new_label
            mapping[parent_two] = vals
        else:
            mapping[parent_two] = [line]
            added += 1
            global_code_to_label[full_code] = new_label

    # Stable sort per parent on code token
    for k, v in list(mapping.items()):
        if isinstance(v, list):
            try:
                mapping[k] = sorted(v, key=lambda s: s.split()[0])
            except Exception as e:
                try:
                    print(f"Raised Exception {e} during list sorting")
                except Exception:
                    pass
    try:
        print(f"[UTILS][apply_lines_to_mapping_dict] added={added} conflicts={len(conflicts)}")
    except Exception:
        pass
    return mapping, added, conflicts


def apply_lines_to_mapping_json(mapping: Dict[str, List[str]] | dict, mapping_json_path: Path) -> int:
    """Write the given dynamic mapping dict to JSON.

    This is the on-disk counterpart of ``apply_lines_to_mapping_dict``: the
    provided mapping is treated as the single source of truth and is written
    as-is (after basic normalization and sorting) to ``mapping_json_path``.
    Returns the total number of lines across all parents.
    """
    # Diagnostics for input
    try:
        print(f"[UTILS][apply_lines_to_mapping_json] path={mapping_json_path}")
        total_lines = sum(len(v) for v in (mapping or {}).values() if isinstance(v, list))
        print(f"[UTILS][apply_lines_to_mapping_json] total_lines={total_lines}")
    except Exception:
        pass
    p = Path(mapping_json_path)
    # Ensure parent directory exists
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        try:
            print(f"[UTILS][apply_lines_to_mapping_json][ERROR] mkdir failed for {p.parent}: {e}")
        except Exception:
            pass
    # Normalize and stable sort each list by code string
    data: Dict[str, List[str]] = {}
    for k, v in (mapping or {}).items():
        if isinstance(v, list):
            data[k] = list(v)
    for k, v in list(data.items()):
        if isinstance(v, list):
            try:
                data[k] = sorted(v, key=lambda s: s.split()[0])
            except Exception:
                pass

    # Robust write: temp file + fsync + replace
    try:
        import os as _os
        payload = json.dumps(data, indent=2, ensure_ascii=False)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(payload)
            try:
                f.flush()
                _os.fsync(f.fileno())
            except Exception:
                pass
        try:
            tmp.replace(p)
        except Exception:
            _os.replace(tmp, p)
    except Exception:
        return sum(len(v) for v in data.values() if isinstance(v, list))
    return sum(len(v) for v in data.values() if isinstance(v, list))


def build_code_label_lookup(mapping: Optional[Dict[str, List[str]]] = None) -> Dict[str, str]:
    """Build a code->label lookup using the provided mapping and the base prompt hierarchy.

    mapping: shape like {"2.1": ["2.1.1 Label", ...], ...}
    Returns dict like {"2.1": "N-acylation to amide", "2.1.1": "...", ...}
    If a code appears in both sources, prefer the mapping-provided label.
    """
    code2label: Dict[str, str] = {}
    # 1) Base prompt-derived labels
    try:
        hierarchy_text = extract_hierarchy_text(default_input_path())
        items = parse_hierarchy_lines(hierarchy_text)
        base = {c: lbl for c, lbl in items}
        code2label.update(base)
    except Exception:
        pass
    # 2) Overlay mapping-provided labels (priority)
    if isinstance(mapping, dict):
        import re as _re
        pat = _re.compile(r"^(\d+(?:\.\d+)+)\s+(.*\S)\s*$")
        for _, lines in mapping.items():
            if not isinstance(lines, list):
                continue
            for raw in lines:
                line = str(raw).strip()
                if not line:
                    continue
                m = pat.match(line)
                if m:
                    code2label[m.group(1)] = m.group(2)
    return code2label


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import json
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_hierarchy_codes(path: Path, smiles_order: List[str]) -> List[str]:
    """Load two-level hierarchy codes aligned to the provided SMILES order.

    Supports the following formats:
    - JSON list of codes in the same order/length as smiles_order
    - JSON dict mapping reaction_smiles -> code
    - CSV with columns: 'reaction_smiles' and 'classification' (produced by hierarchy-only mode)
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    # CSV path support
    if p.suffix.lower() == ".csv":
        import csv as _csv
        from io import StringIO as _StringIO
        reader = _csv.DictReader(_StringIO(text))
        # Prefer 'classification' as the code column; fallbacks allowed
        code_cols = ["classification", "code", "hierarchy", "class"]
        rows = list(reader)
        # Detect smiles column name (default 'reaction_smiles')
        smiles_col = None
        for candidate in ["reaction_smiles", "smiles", "rxn_smiles"]:
            if candidate in reader.fieldnames:  # type: ignore[arg-type]
                smiles_col = candidate
                break
        if not smiles_col:
            raise ValueError("CSV hierarchy codes must have a 'reaction_smiles' column")
        code_col = None
        for c in code_cols:
            if c in reader.fieldnames:  # type: ignore[arg-type]
                code_col = c
                break
        if not code_col:
            raise ValueError("CSV hierarchy codes must have a 'classification' (or code/hierarchy/class) column")
        mapping = {str(r.get(smiles_col, "")): str(r.get(code_col, "")) for r in rows}
        return [mapping.get(sm, "") for sm in smiles_order]
    # JSON path support (list or dict)
    import json
    try:
        data = json.loads(text)
    except Exception as e:
        raise ValueError(f"Failed to parse hierarchy_codes file: {p}: {e}")
    if isinstance(data, list):
        if len(data) != len(smiles_order):
            raise ValueError("hierarchy_codes_path list length does not match number of reactions")
        return [str(x) if x is not None else "" for x in data]
    if isinstance(data, dict):
        return [str(data.get(sm, "")) for sm in smiles_order]
    raise ValueError("Unsupported hierarchy_codes format; expected list, dict or CSV")


def load_detailed_results(path: Path, smiles_order: List[str]) -> List[dict]:
    """Load DetailedResult-like dicts aligned to smiles_order.

    Accepts list[dict] or dict[str, dict]. Does not construct Pydantic models
    to avoid utils<-models dependency; caller can hydrate if needed.
    """
    import json
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        # Expect same length and in the same order or contain reaction_smiles
        if len(data) == len(smiles_order):
            return [dict(x) for x in data]
        # Otherwise try to reindex by reaction_smiles
        idx = {str(x.get("reaction_smiles", "")): dict(x) for x in data}
        return [idx.get(sm, {}) for sm in smiles_order]
    if isinstance(data, dict):
        return [dict(data.get(sm, {})) for sm in smiles_order]
    raise ValueError("Unsupported detailed_results format; expected list or dict")


def load_verify_results(path: Path, smiles_order: List[str]) -> List[dict]:
    """Load VerificationResult-like dicts aligned to smiles_order.

    Accepts list[dict] or dict[str, dict].
    """
    import json
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        if len(data) == len(smiles_order):
            return [dict(x) for x in data]
        idx = {str(x.get("reaction_smiles", "")): dict(x) for x in data}
        return [idx.get(sm, {}) for sm in smiles_order]
    if isinstance(data, dict):
        return [dict(data.get(sm, {})) for sm in smiles_order]
    raise ValueError("Unsupported verify_results format; expected list or dict")
