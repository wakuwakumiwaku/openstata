"""Data input and output helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

_COMPRESSION_SUFFIXES = {".bz2", ".gz", ".xz", ".zip", ".zst"}
_DELIMITED_SUFFIXES = {".csv", ".tab", ".tsv"}


def _data_suffix(path: Path) -> str:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if len(suffixes) >= 2 and suffixes[-1] in _COMPRESSION_SUFFIXES:
        inner_suffix = suffixes[-2]
        if inner_suffix in _DELIMITED_SUFFIXES:
            return inner_suffix
    return suffixes[-1] if suffixes else ""


def read_data(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a clinical dataset from CSV, TSV, Stata, or Parquet."""

    source = Path(path)
    suffix = _data_suffix(source)
    if suffix == ".csv":
        return pd.read_csv(source, **kwargs)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(source, sep="\t", **kwargs)
    if suffix == ".dta":
        return pd.read_stata(source, **kwargs)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(source, **kwargs)
    raise ValueError(f"Unsupported input format: {suffix or '<none>'}")


def write_data(data: pd.DataFrame, path: str | Path, **kwargs: Any) -> None:
    """Write a dataset to CSV, TSV, Stata, or Parquet."""

    destination = Path(path)
    suffix = _data_suffix(destination)
    if suffix == ".csv":
        data.to_csv(destination, index=False, **kwargs)
        return
    if suffix in {".tsv", ".tab"}:
        data.to_csv(destination, sep="\t", index=False, **kwargs)
        return
    if suffix == ".dta":
        data.to_stata(destination, write_index=False, **kwargs)
        return
    if suffix in {".parquet", ".pq"}:
        data.to_parquet(destination, index=False, **kwargs)
        return
    raise ValueError(f"Unsupported output format: {suffix or '<none>'}")
