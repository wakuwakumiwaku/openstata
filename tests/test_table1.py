from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from openstata import table1


def test_table1_builds_grouped_clinical_summary() -> None:
    patients = pd.DataFrame(
        {
            "arm": ["Control", "Control", "Control", "Treatment", "Treatment", "Treatment"],
            "age": [50, 60, 55, 70, 80, 75],
            "crp": [2.0, 5.0, np.nan, 3.0, 7.0, 10.0],
            "female": [1, 0, 1, 1, 1, 0],
        }
    )

    result = table1(
        patients,
        ["age", "crp", "female"],
        by="arm",
        categorical=["female"],
        nonnormal=["crp"],
        pvalues=True,
        standardized_differences=True,
    )

    assert result.loc[("age", "Mean (SD)"), "arm=Control"] == "55.0 (5.0)"
    assert result.loc[("age", "Mean (SD)"), "arm=Treatment"] == "75.0 (5.0)"
    assert result.loc[("crp", "Median [IQR]"), "Overall"] == "5.0 [3.0, 7.0]"
    assert result.loc[("crp", "Missing"), "arm=Control"] == "1 (33.3%)"
    assert result.loc[("female", "1"), "Overall"] == "4 (66.7%)"
    assert result.loc[("age", "Mean (SD)"), "P-value"]
    assert result.loc[("age", "Mean (SD)"), "SMD"] == "4.00"
    assert result.attrs["openstata_group_sizes"] == {
        "Overall": 6,
        "arm=Control": 3,
        "arm=Treatment": 3,
    }


def test_table1_automatically_treats_text_as_categorical() -> None:
    patients = pd.DataFrame({"site": ["A", "B", "A"], "age": [20, 30, 40]})

    result = table1(patients, ["site", "age"])

    assert result.loc[("site", "A"), "Overall"] == "2 (66.7%)"
    assert result.loc[("age", "Mean (SD)"), "Overall"] == "30.0 (10.0)"


def test_table1_labels_variables() -> None:
    patients = pd.DataFrame({"age": [20, 30, 40]})

    result = table1(patients, labels={"age": "Age, years"})

    assert ("Age, years", "Mean (SD)") in result.index


@pytest.mark.parametrize(
    ("pvalues", "standardized_differences", "option"),
    [
        (True, False, "pvalues"),
        (False, True, "standardized_differences"),
    ],
)
def test_table1_requires_groups_for_comparative_statistics(
    pvalues: bool, standardized_differences: bool, option: str
) -> None:
    patients = pd.DataFrame({"age": [20, 30, 40]})

    with pytest.raises(ValueError, match=rf"{option}.*by"):
        table1(
            patients,
            ["age"],
            pvalues=pvalues,
            standardized_differences=standardized_differences,
        )


def test_table1_treats_nonfinite_numeric_values_as_missing() -> None:
    patients = pd.DataFrame(
        {
            "arm": ["A", "A", "B", "B"],
            "lab": [1.0, np.inf, 3.0, np.nan],
        }
    )

    result = table1(patients, ["lab"], by="arm")

    assert result.loc[("lab", "Mean (SD)"), "Overall"] == "2.0 (1.4)"
    assert result.loc[("lab", "Missing"), "Overall"] == "2 (50.0%)"
    assert result.loc[("lab", "Missing"), "arm=A"] == "1 (50.0%)"


@pytest.mark.parametrize("dtype", ["float64", "Int64", "Float64"])
@pytest.mark.parametrize("group_count", [2, 3])
@pytest.mark.parametrize("nonnormal", [False, True])
def test_table1_continuous_pvalues_for_nullable_numeric_columns(
    dtype: str, group_count: int, nonnormal: bool
) -> None:
    samples = [[1.0, 2.0, 5.0], [2.0, 4.0, 6.0, 8.0], [3.0, 7.0, 9.0]][:group_count]
    patients = pd.DataFrame(
        {
            "arm": [group for group, sample in enumerate(samples) for _ in range(len(sample) + 1)],
            "lab": pd.Series(
                [value for sample in samples for value in [*sample, None]], dtype=dtype
            ),
        }
    )
    before = patients.copy(deep=True)

    if group_count == 2:
        reference = (
            stats.mannwhitneyu(*samples, alternative="two-sided")
            if nonnormal
            else stats.ttest_ind(*samples, equal_var=False)
        )
    else:
        reference = stats.kruskal(*samples) if nonnormal else stats.f_oneway(*samples)
    expected_pvalue = f"{reference.pvalue:.3f}"
    assert 0.001 < reference.pvalue < 1

    result = table1(
        patients,
        ["lab"],
        by="arm",
        nonnormal=["lab"] if nonnormal else None,
        pvalues=True,
    )

    summary = "Median [IQR]" if nonnormal else "Mean (SD)"
    assert result.loc[("lab", summary), "P-value"] == expected_pvalue
    assert result.loc[("lab", "Missing"), "P-value"] == ""
    assert result.loc[("lab", "Missing"), "Overall"].startswith(f"{group_count} (")
    pd.testing.assert_frame_equal(patients, before)


@pytest.mark.parametrize("dtype", ["int64", "uint64", "Int64", "UInt64"])
@pytest.mark.parametrize("group_count", [2, 3])
def test_table1_nonnormal_pvalues_preserve_large_integer_ranks(
    dtype: str, group_count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    numpy_dtype = dtype.lower()
    base = 2**63 if numpy_dtype == "uint64" else 2**60
    group_size = 6 // group_count
    samples = [
        np.array(
            [base + group * group_size + offset for offset in range(1, group_size + 1)],
            dtype=numpy_dtype,
        )
        for group in range(group_count)
    ]
    # These exact integers become artificial ties if normalized through float64.
    assert np.unique(np.concatenate(samples)).size == 6
    assert np.unique(np.concatenate(samples).astype(float)).size == 1
    patients = pd.DataFrame(
        {
            "arm": [group for group, sample in enumerate(samples) for _ in sample],
            "lab": pd.Series(np.concatenate(samples), dtype=dtype),
        }
    )
    before = patients.copy(deep=True)
    reference = (
        stats.mannwhitneyu(*samples, alternative="two-sided")
        if group_count == 2
        else stats.kruskal(*samples)
    )
    # Preserve the direct integer SciPy result, including versions returning NaN.
    expected_pvalue = "" if np.isnan(reference.pvalue) else f"{reference.pvalue:.3f}"
    test_name = "mannwhitneyu" if group_count == 2 else "kruskal"
    rank_test = getattr(stats, test_name)
    observed_samples = []

    def record_samples(*actual_samples: np.ndarray, **options: object) -> object:
        observed_samples.append(actual_samples)
        return rank_test(*actual_samples, **options)

    monkeypatch.setattr(stats, test_name, record_samples)

    result = table1(patients, ["lab"], by="arm", nonnormal=["lab"], pvalues=True)

    assert result.loc[("lab", "Median [IQR]"), "P-value"] == expected_pvalue
    assert len(observed_samples) == 1
    for actual, expected in zip(observed_samples[0], samples, strict=True):
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    pd.testing.assert_frame_equal(patients, before)


def test_table1_rejects_complex_data() -> None:
    with pytest.raises(TypeError, match="real-valued"):
        table1(pd.DataFrame({"signal": [1 + 2j, 3 + 4j]}), ["signal"])


def test_table1_rejects_duplicate_variables() -> None:
    patients = pd.DataFrame({"age": [20, 30, 40]})

    with pytest.raises(ValueError, match="Duplicate variables: age"):
        table1(patients, ["age", "age"])


@pytest.mark.parametrize("dtype", ["object", "category"])
def test_table1_rejects_ambiguous_group_labels(dtype: str) -> None:
    patients = pd.DataFrame(
        {
            "arm": pd.Series([1, 1, "1", "1", "1"], dtype=dtype),
            "age": [20, 30, 60, 70, 80],
        }
    )
    before = patients.copy(deep=True)

    with pytest.raises(ValueError, match="Group labels must be unique.*arm=1"):
        table1(patients, ["age"], by="arm")

    pd.testing.assert_frame_equal(patients, before)


def test_table1_ignores_unobserved_ambiguous_group_categories() -> None:
    patients = pd.DataFrame(
        {
            "arm": pd.Categorical([1, 1, 2, 2], categories=[1, "1", 2]),
            "age": [20, 30, 60, 80],
        }
    )

    result = table1(patients, ["age"], by="arm")

    assert list(result.columns) == ["Overall", "arm=1", "arm=2"]
    assert result.loc[("age", "Mean (SD)"), "arm=1"] == "25.0 (7.1)"
    assert result.loc[("age", "Mean (SD)"), "arm=2"] == "70.0 (14.1)"
    assert result.attrs["openstata_group_sizes"] == {"Overall": 4, "arm=1": 2, "arm=2": 2}
