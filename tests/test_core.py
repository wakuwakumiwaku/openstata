from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from openstata import ci_mean, ci_proportion, summarize, tabulate


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


def test_summarize_treats_nonfinite_values_as_missing() -> None:
    data = pd.DataFrame({"lab": [1.0, 3.0, np.nan, np.inf, -np.inf]})

    result = summarize(data, ["lab"])

    assert result.loc["lab", "Obs"] == 2
    assert result.loc["lab", "Missing"] == 3
    assert result.loc["lab", "Mean"] == pytest.approx(2.0)
    assert result.loc["lab", "Min"] == pytest.approx(1.0)
    assert result.loc["lab", "Max"] == pytest.approx(3.0)


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


def test_ci_proportion_matches_stata_exact_example() -> None:
    data = pd.DataFrame({"promoted": [1, 1, *([0] * 18)]})

    result = ci_proportion(data, ["promoted"])

    assert result.loc["promoted", "Obs"] == 20
    assert result.loc["promoted", "Proportion"] == pytest.approx(0.1)
    assert result.loc["promoted", "Std. err."] == pytest.approx(0.0670820393)
    assert result.loc["promoted", "CI lower"] == pytest.approx(0.0123485272)
    assert result.loc["promoted", "CI upper"] == pytest.approx(0.3169827140)
    assert result.attrs == {"confidence_level": 95.0, "method": "exact"}


@pytest.mark.parametrize(
    ("method", "lower", "upper"),
    [
        ("wald", -0.0314783811, 0.2314783811),
        ("wilson", 0.0278664812, 0.3010336452),
        ("agresti", 0.0156562390, 0.3132438733),
        ("jeffreys", 0.0213724880, 0.2838532563),
    ],
)
def test_ci_proportion_supports_common_binomial_intervals(
    method: str, lower: float, upper: float
) -> None:
    data = pd.DataFrame({"promoted": [1, 1, *([0] * 18)]})

    result = ci_proportion(data, ["promoted"], method=method)  # type: ignore[arg-type]

    assert result.loc["promoted", "CI lower"] == pytest.approx(lower)
    assert result.loc["promoted", "CI upper"] == pytest.approx(upper)
    assert result.attrs["method"] == method


def test_ci_proportion_finds_binary_variables_and_excludes_nonfinite_values() -> None:
    data = pd.DataFrame(
        {
            "age": [20.0, 30.0, 40.0, 50.0],
            "female": [1.0, 0.0, np.nan, np.inf],
            "consented": [True, False, True, True],
        }
    )

    result = ci_proportion(data)

    assert list(result.index) == ["female", "consented"]
    assert result.loc["female", "Obs"] == 2
    assert result.loc["female", "Proportion"] == pytest.approx(0.5)
    assert result.loc["consented", "Proportion"] == pytest.approx(0.75)


def test_ci_proportion_rejects_nonbinary_data() -> None:
    with pytest.raises(ValueError, match="0 and 1"):
        ci_proportion(pd.DataFrame({"response": [0, 1, 2]}), ["response"])


def test_ci_proportion_rejects_unknown_interval_method() -> None:
    with pytest.raises(ValueError, match="method"):
        ci_proportion(
            pd.DataFrame({"response": [0, 1]}),
            ["response"],
            method="approximate",  # type: ignore[arg-type]
        )


def test_numeric_analyses_reject_complex_data() -> None:
    data = pd.DataFrame({"signal": [0 + 1j, 1 + 2j]})

    for analysis in (summarize, ci_mean, ci_proportion):
        with pytest.raises(TypeError, match="real-valued"):
            analysis(data, ["signal"])


def test_one_way_tabulate_reports_percent_and_cumulative_percent(
    patients: pd.DataFrame,
) -> None:
    result = tabulate(patients, "arm")

    assert result.loc["Control", "Freq."] == 3
    assert result.loc["Control", "Percent"] == pytest.approx(50.0)
    assert result.loc["Treatment", "Cum."] == pytest.approx(100.0)


def test_one_way_tabulate_rejects_two_way_percentage_modes(
    patients: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="column variable"):
        tabulate(patients, "arm", percent="row")


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


def test_missing_label_cannot_collide_with_observed_value() -> None:
    data = pd.DataFrame({"arm": ["Control", "<missing>", None]})

    with pytest.raises(ValueError, match="reserved"):
        tabulate(data, "arm", missing=True)
