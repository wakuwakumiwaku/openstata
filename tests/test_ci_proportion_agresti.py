from __future__ import annotations

from math import sqrt
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from openstata import OpenStata, ci_proportion
from openstata.core import ProportionMethod


def _agresti_reference(successes: int, observations: int, level: float) -> tuple[float, float]:
    # Independent scalar adjusted-normal formula, intersected with the parameter space.
    critical = NormalDist().inv_cdf(0.5 + level / 200)
    adjusted_n = observations + critical**2
    adjusted_p = (successes + critical**2 / 2) / adjusted_n
    margin = critical * sqrt(adjusted_p * (1 - adjusted_p) / adjusted_n)
    return max(0.0, adjusted_p - margin), min(1.0, adjusted_p + margin)


def _assert_agresti_result(
    result: pd.DataFrame, successes: int, observations: int, level: float = 95.0
) -> None:
    lower, upper = _agresti_reference(successes, observations, level)
    row = result.loc["event"]
    proportion = successes / observations
    assert 0.0 <= row["CI lower"] <= row["CI upper"] <= 1.0
    assert row["CI lower"] == pytest.approx(lower)
    assert row["CI upper"] == pytest.approx(upper)
    if lower == 0.0:
        assert row["CI lower"] == 0.0
    if upper == 1.0:
        assert row["CI upper"] == 1.0
    assert row["Obs"] == observations
    assert row["Proportion"] == pytest.approx(proportion)
    assert row["Std. err."] == pytest.approx(sqrt(proportion * (1 - proportion) / observations))
    assert result.attrs == {"confidence_level": level, "method": "agresti"}


@pytest.mark.parametrize(
    "dtype", ["int64", "Int64", "uint64", "UInt64", "float64", "Float64", "bool", "boolean"]
)
@pytest.mark.parametrize("outcome", [0, 1], ids=["no-events", "all-events"])
def test_agresti_limits_stay_in_probability_space(dtype: str, outcome: int) -> None:
    values: list[int | None] = [outcome] * 10
    if dtype in {"Int64", "UInt64", "Float64", "boolean"}:
        values.append(None)
    data = pd.DataFrame({"event": pd.Series(values, dtype=dtype)})
    before = data.copy(deep=True)

    result = ci_proportion(data, ["event"], method="agresti")

    _assert_agresti_result(result, outcome * 10, 10)
    pd.testing.assert_frame_equal(data, before)


@pytest.mark.parametrize("successes", [1, 99])
def test_agresti_sparse_nonboundary_limits_are_bounded(successes: int) -> None:
    data = pd.DataFrame({"event": [1] * successes + [0] * (100 - successes)})
    before = data.copy(deep=True)

    result = ci_proportion(data, ["event"], method="agresti")

    _assert_agresti_result(result, successes, 100)
    pd.testing.assert_frame_equal(data, before)


@pytest.mark.parametrize("caller", ["wrapper", "command"])
@pytest.mark.parametrize("outcome", [0, 1])
def test_agresti_bounds_through_public_callers(caller: str, outcome: int) -> None:
    data = pd.DataFrame({"event": pd.Series([outcome] * 10 + [None], dtype="Int64")})
    before = data.copy(deep=True)
    stata = OpenStata(data)

    result = (
        stata.ci_proportion(["event"], method="agresti", level=90)
        if caller == "wrapper"
        else stata.run("ci proportions event, agresti level(90)")
    )

    _assert_agresti_result(result, outcome * 10, 10, level=90)
    pd.testing.assert_frame_equal(data, before)


@pytest.mark.parametrize("dtype", ["float64", "Float64"])
@pytest.mark.parametrize("outcome", [0, 1])
def test_agresti_bounds_after_nonfinite_exclusions(dtype: str, outcome: int) -> None:
    data = pd.DataFrame(
        {"event": pd.Series([outcome] * 10 + [None, np.inf, -np.inf], dtype=dtype)}
    )
    before = data.copy(deep=True)

    # Automatic binary selection must use only finite, nonmissing observations.
    result = ci_proportion(data, method="agresti")

    _assert_agresti_result(result, outcome * 10, 10)
    pd.testing.assert_frame_equal(data, before)


@pytest.mark.parametrize("level", [90.0, 95.0, 99.0])
def test_agresti_interior_limits_are_unchanged(level: float) -> None:
    data = pd.DataFrame({"event": [1] * 20 + [0] * 30})
    before = data.copy(deep=True)
    lower, upper = _agresti_reference(20, 50, level)
    assert 0 < lower < upper < 1

    result = ci_proportion(data, ["event"], method="agresti", level=level)

    _assert_agresti_result(result, 20, 50, level)
    pd.testing.assert_frame_equal(data, before)


@pytest.mark.parametrize(
    ("values", "dtype"),
    [
        ([], "float64"),
        ([None, None], "Float64"),
        ([np.inf, -np.inf, np.nan], "float64"),
        ([], "Int64"),
    ],
    ids=["empty-native", "all-missing-nullable", "all-nonfinite", "empty-nullable"],
)
def test_agresti_empty_samples_remain_undefined(values: list, dtype: str) -> None:
    data = pd.DataFrame({"event": pd.Series(values, dtype=dtype)})
    before = data.copy(deep=True)

    result = ci_proportion(data, ["event"], method="agresti")

    assert result.loc["event", "Obs"] == 0
    assert result.loc["event", ["Proportion", "Std. err.", "CI lower", "CI upper"]].isna().all()
    assert result.attrs == {"confidence_level": 95.0, "method": "agresti"}
    pd.testing.assert_frame_equal(data, before)


@pytest.mark.parametrize("method", ["exact", "wald", "wilson", "jeffreys"])
@pytest.mark.parametrize("successes", [0, 2, 10])
def test_other_interval_methods_are_unchanged(method: ProportionMethod, successes: int) -> None:
    data = pd.DataFrame({"event": [1] * successes + [0] * (10 - successes)})
    before = data.copy(deep=True)
    proportion = successes / 10
    if method in {"exact", "wilson"}:
        expected = stats.binomtest(successes, 10).proportion_ci(method=method)
    elif method == "wald":
        margin = NormalDist().inv_cdf(0.975) * sqrt(proportion * (1 - proportion) / 10)
        expected = (proportion - margin, proportion + margin)
    else:
        expected = stats.beta.ppf([0.025, 0.975], successes + 0.5, 10 - successes + 0.5)

    result = ci_proportion(data, ["event"], method=method)

    assert result.loc["event", "CI lower"] == pytest.approx(expected[0], abs=1e-14)
    assert result.loc["event", "CI upper"] == pytest.approx(expected[1], abs=1e-14)
    assert result.loc["event", "Obs"] == 10
    assert result.loc["event", "Proportion"] == pytest.approx(proportion)
    assert result.loc["event", "Std. err."] == pytest.approx(
        sqrt(proportion * (1 - proportion) / 10)
    )
    assert result.attrs == {"confidence_level": 95.0, "method": method}
    pd.testing.assert_frame_equal(data, before)


@pytest.mark.parametrize("dtype", ["int64", "Int64", "uint64", "UInt64"])
def test_agresti_rejects_large_nonbinary_integers_without_mutation(dtype: str) -> None:
    values = (
        [-(2**63) + 1, 2**60 + 1, 2**60 + 2, 2**63 - 1]
        if dtype.lower() == "int64"
        else [2**63 + 1, 2**63 + 2, 2**64 - 1]
    )
    data = pd.DataFrame({"event": pd.Series(values, dtype=dtype)})
    before = data.copy(deep=True)
    assert data["event"].tolist() == values
    assert data["event"].nunique() == len(values)

    with pytest.raises(ValueError, match="coded as 0 and 1"):
        ci_proportion(data, ["event"], method="agresti")

    pd.testing.assert_frame_equal(data, before)
    assert data["event"].tolist() == values
