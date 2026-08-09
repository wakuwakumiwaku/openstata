from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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


def test_table1_rejects_complex_data() -> None:
    with pytest.raises(TypeError, match="real-valued"):
        table1(pd.DataFrame({"signal": [1 + 2j, 3 + 4j]}), ["signal"])
