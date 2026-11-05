from __future__ import annotations

import importlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from openstata import StataFrame, table1


def test_table1_import_provenance() -> None:
    source = Path(importlib.import_module("openstata.table1").__file__).resolve()
    expected = os.environ.get("EXPECTED_SOURCE_ROOT")
    if expected:
        assert source.is_relative_to(Path(expected).resolve()), source


@pytest.mark.parametrize("dtype", ["float16", "float32", "Float32", "float64"])
@pytest.mark.parametrize("groups", [2, 3])
@pytest.mark.parametrize("nonnormal", [False, True])
def test_table1_matches_float64_analysis_of_stored_values(
    dtype: str, groups: int, nonnormal: bool
) -> None:
    scale = 50000 if dtype == "float16" else 1e20
    samples = [[-1, 0, 1], [-0.5, 0.25, 0.75], [-0.75, 0.5, 1]][:groups]
    values = [
        value * scale if value is not None else None
        for sample in samples
        for value in [*sample, None]
    ]
    data = pd.DataFrame(
        {
            "arm": [group for group in range(groups) for _ in range(4)],
            "lab": pd.Series(values, dtype=dtype),
        }
    )
    before = data.copy(deep=True)
    reference = data.copy(deep=True)
    reference["lab"] = reference["lab"].astype("float64")
    options = {
        "by": "arm",
        "nonnormal": ["lab"] if nonnormal else None,
        "pvalues": True,
        "standardized_differences": True,
    }

    expected = table1(reference, ["lab"], **options)
    actual = table1(data, ["lab"], **options)

    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(data, before)


def test_command_table1_uses_wide_statistics_without_changing_group_labels() -> None:
    data = pd.DataFrame(
        {
            "arm": pd.Series([0.1, 0.1, 0.1, 0.2, 0.2, 0.2], dtype="float32"),
            "lab": pd.Series([1e38] * 6, dtype="Float32"),
        }
    )
    before = data.copy(deep=True)
    result = StataFrame(data).run("table1 lab, by(arm)")
    expected_mean = float(data["lab"].iloc[0])
    assert result.loc[("lab", "Mean (SD)"), "Overall"] == f"{expected_mean:.1f} (0.0)"
    assert list(result.columns) == ["Overall", "arm=0.10000000149011612", "arm=0.20000000298023224"]
    pd.testing.assert_frame_equal(data, before)


@pytest.mark.parametrize("dtype", ["int64", "Int64", "uint64", "UInt64"])
def test_integer_rank_inputs_remain_exact(dtype: str, monkeypatch: pytest.MonkeyPatch) -> None:
    base = 2**63 if dtype.lower() == "uint64" else 2**60
    data = pd.DataFrame(
        {
            "arm": ["A"] * 3 + ["B"] * 3,
            "lab": pd.Series([base + n for n in range(6)], dtype=dtype),
        }
    )
    before = data.copy(deep=True)
    seen = []
    original = stats.mannwhitneyu

    def observed(first, second, **kwargs):
        seen.append((first.copy(), second.copy()))
        return original(first, second, **kwargs)

    monkeypatch.setattr(stats, "mannwhitneyu", observed)
    table1(data, ["lab"], by="arm", nonnormal=["lab"], pvalues=True)

    assert len(seen) == 1
    for index, actual in enumerate(seen[0]):
        expected = np.array(
            [base + n for n in range(index * 3, index * 3 + 3)], dtype=dtype.lower()
        )
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    pd.testing.assert_frame_equal(data, before)
