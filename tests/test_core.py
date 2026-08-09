from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from openstata import ci_mean, summarize, tabulate


@pytest.fixture
def patients() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "arm": ["Control", "Control", "Control", "Treatment", "Treatment", "Treatment"],
            "age": [50, 60, 55, 70, 80, 75],
            "crp": [2.0, 5.0, np.nan, 3.0, 7.0, 10.0],
            "female": [1, 0, 1, 1, 1, 0],
        }
    )


def test_summarize_reports_stata_like_statistics(patients: pd.DataFrame) -> None:
    result = summarize(patients, ["age", "crp"])

    assert result.loc["age", "Obs"] == 6
    assert result.loc["crp", "Missing"] == 1
    assert result.loc["age", "Mean"] == pytest.approx(65.0)
    assert result.loc["age", "Std. dev."] == pytest.approx(11.832159566)


def test_summarize_detail_adds_percentiles_and_shape(patients: pd.DataFrame) -> None:
    result = summarize(patients, ["age"], detail=True)

    assert result.loc["age", "p50"] == pytest.approx(65.0)
    assert result.loc["age", "Variance"] == pytest.approx(140.0)
    assert "Skewness" in result.columns
    assert "Kurtosis" in result.columns


def test_summarize_rejects_nonnumeric_columns(patients: pd.DataFrame) -> None:
    with pytest.raises(TypeError, match="numeric"):
        summarize(patients, ["arm"])


def test_ci_mean_uses_student_t_interval() -> None:
    data = pd.DataFrame({"age": [10.0, 12.0, 14.0, 16.0, 18.0]})

    result = ci_mean(data, ["age"])

    standard_error = data["age"].std(ddof=1) / np.sqrt(5)
    critical_value = stats.t.ppf(0.975, df=4)
    assert result.loc["age", "Obs"] == 5
    assert result.loc["age", "Mean"] == pytest.approx(14.0)
    assert result.loc["age", "Std. err."] == pytest.approx(standard_error)
    assert result.loc["age", "CI lower"] == pytest.approx(14.0 - critical_value * standard_error)
    assert result.loc["age", "CI upper"] == pytest.approx(14.0 + critical_value * standard_error)
    assert result.attrs["confidence_level"] == 95.0


def test_ci_mean_excludes_missing_and_nonfinite_values() -> None:
    data = pd.DataFrame({"ratio": [1.0, 2.0, np.nan, np.inf, -np.inf]})

    result = ci_mean(data, ["ratio"], level=90)

    assert result.loc["ratio", "Obs"] == 2
    assert result.loc["ratio", "Mean"] == pytest.approx(1.5)
    assert np.isfinite(result.loc["ratio", "CI lower"])
    assert np.isfinite(result.loc["ratio", "CI upper"])
    assert result.attrs["confidence_level"] == 90.0


def test_ci_mean_reports_undefined_interval_for_one_observation() -> None:
    result = ci_mean(pd.DataFrame({"age": [42.0, np.nan]}), ["age"])

    assert result.loc["age", "Obs"] == 1
    assert result.loc["age", "Mean"] == pytest.approx(42.0)
    assert pd.isna(result.loc["age", "Std. err."])
    assert pd.isna(result.loc["age", "CI lower"])
    assert pd.isna(result.loc["age", "CI upper"])


@pytest.mark.parametrize("level", [0, 100, -1, 101, np.nan])
def test_ci_mean_rejects_invalid_confidence_levels(level: float) -> None:
    with pytest.raises(ValueError, match="level"):
        ci_mean(pd.DataFrame({"age": [10.0, 12.0]}), level=level)


def test_one_way_tabulate_reports_percent_and_cumulative_percent(
    patients: pd.DataFrame,
) -> None:
    result = tabulate(patients, "arm")

    assert result.loc["Control", "Freq."] == 3
    assert result.loc["Control", "Percent"] == pytest.approx(50.0)
    assert result.loc["Treatment", "Cum."] == pytest.approx(100.0)


def test_two_way_tabulate_counts_and_totals(patients: pd.DataFrame) -> None:
    result = tabulate(patients, "arm", "female")

    assert result.loc["Control", 1] == 2
    assert result.loc["Treatment", 0] == 1
    assert result.loc["Total", "Total"] == 6


def test_two_way_tabulate_row_percentages(patients: pd.DataFrame) -> None:
    result = tabulate(patients, "arm", "female", percent="row")

    assert result.loc["Control"].sum() == pytest.approx(100.0)
    assert result.loc["Treatment", 1] == pytest.approx(66.6666667)


def test_missing_is_an_explicit_tabulation_level(patients: pd.DataFrame) -> None:
    result = tabulate(patients, "crp", missing=True)

    assert result.loc["<missing>", "Freq."] == 1
