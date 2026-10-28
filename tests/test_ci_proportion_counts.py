from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openstata import OpenStata, ci_proportion

_METHODS = ["exact", "wald", "wilson", "agresti", "jeffreys"]


@pytest.mark.parametrize("dtype", ["float32", "Float32"])
@pytest.mark.parametrize("successes", [2**24 + 1, 2**24 + 3])
def test_ci_proportion_counts_events_beyond_float32_integer_precision(
    dtype: str, successes: int
) -> None:
    # These counts round down and up, respectively, when summed in float32.
    values = np.ones(successes + 3, dtype="float32")
    values[-3:] = [np.nan, np.inf, -np.inf]
    data = pd.DataFrame({"flag": pd.Series(values, dtype=dtype)})
    before = data.copy(deep=True)

    for method in _METHODS:
        result = ci_proportion(data, ["flag"], method=method)  # type: ignore[arg-type]
        row = result.loc["flag"]

        assert row["Obs"] == successes
        assert row["Proportion"] == 1.0
        assert row["Std. err."] == 0.0
        assert 0 <= row["CI lower"] <= row["CI upper"] <= 1
        assert result.attrs == {"confidence_level": 95.0, "method": method}
        if method == "exact":
            # With n successes in n trials, the exact lower tail solves p**n = alpha/2.
            assert row["CI lower"] == pytest.approx(0.025 ** (1 / successes), rel=0, abs=1e-12)
            assert row["CI upper"] == 1.0
        elif method == "wald":
            assert row["CI lower"] == row["CI upper"] == 1.0

    explicit = ci_proportion(data, ["flag"])
    pd.testing.assert_frame_equal(ci_proportion(data), explicit)
    pd.testing.assert_frame_equal(OpenStata(data).run("ci proportions flag"), explicit)
    pd.testing.assert_frame_equal(data, before)


@pytest.mark.parametrize(
    "dtype",
    [
        "float16", "float32", "float64", "Float32", "Float64",
        "int8", "uint8", "int64", "uint64", "Int8", "UInt8", "Int64", "UInt64",
        "bool", "boolean",
    ],
)
@pytest.mark.parametrize(
    ("method", "lower", "upper"),
    [
        ("exact", 0.0123485272, 0.3169827140),
        ("wald", -0.0314783811, 0.2314783811),
        ("wilson", 0.0278664812, 0.3010336452),
        ("agresti", 0.0156562390, 0.3132438733),
        ("jeffreys", 0.0213724880, 0.2838532563),
    ],
)
def test_ci_proportion_counting_preserves_supported_dtypes_and_intervals(
    dtype: str, method: str, lower: float, upper: float
) -> None:
    values = pd.Series([1, 1, *([0] * 18)], dtype=dtype)
    if dtype in {"Float32", "Float64", "Int8", "UInt8", "Int64", "UInt64", "boolean"}:
        values = pd.concat([values, pd.Series([None], dtype=dtype)], ignore_index=True)
    elif dtype.startswith("float"):
        values = pd.concat(
            [values, pd.Series([np.nan, np.inf, -np.inf], dtype=dtype)], ignore_index=True
        )
    data = pd.DataFrame({"flag": values})
    before = data.copy(deep=True)

    result = ci_proportion(data, ["flag"], method=method)  # type: ignore[arg-type]

    assert result.loc["flag", "Obs"] == 20
    assert result.loc["flag", "Proportion"] == 0.1
    assert result.loc["flag", "Std. err."] == pytest.approx(0.0670820393)
    assert result.loc["flag", "CI lower"] == pytest.approx(lower)
    assert result.loc["flag", "CI upper"] == pytest.approx(upper)
    assert result.attrs == {"confidence_level": 95.0, "method": method}
    pd.testing.assert_frame_equal(data, before)


@pytest.mark.parametrize("dtype", ["float32", "Float32", "Int64", "UInt64", "boolean"])
def test_ci_proportion_counting_preserves_empty_and_singleton_results(dtype: str) -> None:
    data = pd.DataFrame(
        {
            "empty": pd.Series([None, None], dtype=dtype),
            "one": pd.Series([1, None], dtype=dtype),
            "zero": pd.Series([0, None], dtype=dtype),
        }
    )
    before = data.copy(deep=True)
    result = OpenStata(data).ci_proportion(["empty", "one", "zero"])

    assert result.loc["empty", "Obs"] == 0
    assert result.loc["empty"].drop("Obs").isna().all()
    assert result.loc["one", "Obs"] == result.loc["zero", "Obs"] == 1
    assert result.loc["one", "Proportion"] == 1.0
    assert result.loc["zero", "Proportion"] == 0.0
    assert result.loc["one", "Std. err."] == result.loc["zero", "Std. err."] == 0.0
    assert result.loc["one", "CI lower"] == pytest.approx(0.025)
    assert result.loc["one", "CI upper"] == 1.0
    assert result.loc["zero", "CI lower"] == 0.0
    assert result.loc["zero", "CI upper"] == pytest.approx(0.975)
    assert list(ci_proportion(data).index) == ["one", "zero"]
    pd.testing.assert_frame_equal(data, before)
