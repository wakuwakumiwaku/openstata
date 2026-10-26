from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openstata import OpenStata, summarize


def _moment(value: float, offset: float = 0.0) -> float:
    return np.nan if pd.isna(value) else float(value) + offset


@pytest.mark.parametrize("caller", ["function", "wrapper", "command"])
@pytest.mark.parametrize("dtype", ["Int64", "UInt64"])
@pytest.mark.parametrize("size", [3, 4])
def test_large_nullable_moments_preserve_missing_results(
    caller: str, dtype: str, size: int
) -> None:
    base = 2**60 if dtype == "Int64" else 2**63
    data = pd.DataFrame({
        "number": pd.Series([base + i for i in range(1, size + 1)] + [None], dtype=dtype)
    })
    before = data.copy(deep=True)
    values = data["number"].dropna()
    basic = summarize(data, ["number"])
    expected_skew = _moment(values.skew())
    expected_kurtosis = _moment(values.kurt(), 3.0) if size > 3 else np.nan

    if caller == "function":
        result = summarize(data, ["number"], detail=True)
    elif caller == "wrapper":
        result = OpenStata(data).summarize(["number"], detail=True)
    else:
        result = OpenStata(data).run("summarize number, detail")

    pd.testing.assert_frame_equal(result[basic.columns], basic)
    pd.testing.assert_frame_equal(data, before)
    assert result.loc["number", "Obs"] == size
    assert result.loc["number", "Missing"] == 1
    for name, expected in (("Skewness", expected_skew), ("Kurtosis", expected_kurtosis)):
        if pd.isna(expected):
            assert pd.isna(result.loc["number", name])
        else:
            assert result.loc["number", name] == pytest.approx(expected)


@pytest.mark.parametrize("method,column", [("skew", "Skewness"), ("kurt", "Kurtosis")])
def test_dependency_missing_moment_is_nan_not_an_exception(
    method: str, column: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # pandas' nullable reductions may return pd.NA rather than numpy.nan.
    # Only that dependency boundary is controlled; the public summary is real.
    data = pd.DataFrame({"number": pd.Series([1, 2, 4, 8], dtype="Int64")})
    before = data.copy(deep=True)
    monkeypatch.setattr(pd.Series, method, lambda self, *args, **kwargs: pd.NA)

    result = summarize(data, ["number"], detail=True)

    assert pd.isna(result.loc["number", column])
    assert result.loc["number", "p50"] == 3.0
    assert result.loc["number", "Variance"] == pytest.approx(data["number"].var(ddof=1))
    pd.testing.assert_frame_equal(data, before)
    assert '"' + column + '":null' in result.reset_index().to_json(orient="records")


@pytest.mark.parametrize(
    "dtype", ["int64", "Int64", "uint64", "UInt64", "float64", "Float64", "bool", "boolean"]
)
def test_finite_moments_and_pearson_offset_are_unchanged(dtype: str) -> None:
    raw = [True, False, True, True] if dtype in ("bool", "boolean") else [1, 2, 4, 8]
    data = pd.DataFrame({"number": pd.Series(raw, dtype=dtype)})
    before = data.copy(deep=True)
    values = data["number"].dropna()

    result = summarize(data, ["number"], detail=True)

    assert result.loc["number", "Skewness"] == pytest.approx(float(values.skew()))
    assert result.loc["number", "Kurtosis"] == pytest.approx(float(values.kurt()) + 3)
    pd.testing.assert_frame_equal(data, before)


@pytest.mark.parametrize("values", [[], [None, None], [1], [1, 2]])
def test_small_nullable_samples_keep_existing_undefined_moments(values: list) -> None:
    data = pd.DataFrame({"number": pd.Series(values, dtype="Int64")})
    before = data.copy(deep=True)

    result = summarize(data, ["number"], detail=True)

    assert pd.isna(result.loc["number", "Skewness"])
    assert pd.isna(result.loc["number", "Kurtosis"])
    pd.testing.assert_frame_equal(data, before)
