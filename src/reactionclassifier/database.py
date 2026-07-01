"""Fetch the full public reaction database (hosted externally, not in the wheel).

The database is the ~666k-reaction labelled corpus with NameRXN-derived columns
removed. Requires the optional ``database`` extra (``pip install
reactionclassifier[database]``).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

# Filled in at release time (Zenodo record).
ZENODO_URL: Optional[str] = "https://zenodo.org/records/21097758/files/reaction_db_public.parquet"
EXPECTED_SHA256: Optional[str] = "af8482038abf24adc91e33dfd009392c61b9629dca4c2f5aff96d53295390549"
FILENAME = "reaction_db_public.parquet"


def _cache_dir() -> Path:
    d = Path(os.environ.get("REACTIONCLASSIFIER_CACHE", Path.home() / ".cache" / "reactionclassifier"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_database(dest: Optional[os.PathLike] = None, force: bool = False) -> Path:
    """Download (and cache) the public reaction database, returning its path."""
    if ZENODO_URL is None:
        raise RuntimeError(
            "The database download URL is not set in this build. Set "
            "reactionclassifier.database.ZENODO_URL to the Zenodo file URL, or "
            "download the parquet manually."
        )
    target = Path(dest) if dest is not None else _cache_dir() / FILENAME
    if target.exists() and not force:
        return target

    import requests  # optional dependency

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    with requests.get(ZENODO_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    if EXPECTED_SHA256 and _sha256(tmp) != EXPECTED_SHA256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("Downloaded database failed sha256 integrity check.")
    tmp.replace(target)
    return target


def load_database(dest: Optional[os.PathLike] = None):
    """Convenience: download if needed and return the parquet as a DataFrame."""
    import pandas as pd

    return pd.read_parquet(download_database(dest))
