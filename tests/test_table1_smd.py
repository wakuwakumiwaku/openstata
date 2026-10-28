from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from openstata import OpenStata, table1


@pytest.mark.parametrize("dtype", ["float64", "Float64", "Int64", "UInt64"])
@pytest.mark.parametrize(
    "values",
    [[1, None, 3, 5], [1, 2, 3, None], [1, None, 3, None]],
    ids=["singleton-first", "singleton-second", "singletons-both"],
)
def test_table1_smd_is_undefined_for_singleton_groups(
    dtype: str, values: list[int | None]
) -> None:
    patients = pd.DataFrame(
        {"arm": ["A", "A", "B", "B"], "lab": pd.Series(values, dtype=dtype)}
    )
    before = patients.copy(deep=True)
    without_smd = table1(patients, ["lab"], by="arm")

    try:
        result = table1(patients, ["lab"], by="arm", standardized_differences=True)
    except TypeError as error:
        pytest.fail(f"An undefined singleton SMD must not abort the table: {error}")

    assert result.loc[("lab", "Mean (SD)"), "SMD"] == ""
    assert result.loc[("lab", "Missing"), "SMD"] == ""
    pd.testing.assert_frame_equal(result.drop(columns="SMD"), without_smd)
    assert result.attrs == without_smd.attrs
    pd.testing.assert_frame_equal(patients, before)


@pytest.mark.parametrize("dtype", ["float64", "Float64", "Int64", "UInt64"])
@pytest.mark.parametrize(
    "values",
    [
        [None, None, 3, 5],
        [1, 2, None, None],
        [None, None, None, None],
        [1, 2, 3, 5],
        [2, 2, 2, 2],
        [2, 2, 5, 5],
    ],
    ids=["empty-first", "empty-second", "empty-both", "estimable", "equal", "separated"],
)
def test_table1_smd_preserves_empty_constant_and_estimable_results(
    dtype: str, values: list[int | None]
) -> None:
    patients = pd.DataFrame(
        {"arm": ["A", "A", "B", "B"], "lab": pd.Series(values, dtype=dtype)}
    )
    before = patients.copy(deep=True)
    first = patients.loc[patients.arm == "A", "lab"].dropna()
    second = patients.loc[patients.arm == "B", "lab"].dropna()
    expected = ""
    if len(first) > 1 and len(second) > 1:
        # Independent pandas reference for the documented sample-variance estimator.
        variance = (first.var(ddof=1) + second.var(ddof=1)) / 2
        difference = abs(second.mean() - first.mean())
        if variance > 0:
            expected = f"{difference / np.sqrt(variance):.2f}"
        elif difference == 0:
            expected = "0.00"

    result = table1(patients, ["lab"], by="arm", standardized_differences=True)

    assert result.loc[("lab", "Mean (SD)"), "SMD"] == expected
    if ("lab", "Missing") in result.index:
        assert result.loc[("lab", "Missing"), "SMD"] == ""
    pd.testing.assert_frame_equal(patients, before)


@pytest.mark.parametrize("invalid", [np.inf, -np.inf])
def test_table1_smd_counts_observations_after_excluding_nonfinite_values(invalid: float) -> None:
    patients = pd.DataFrame(
        {
            "arm": ["A", "A", "B", "B"],
            "lab": pd.Series([1, invalid, 3, 5], dtype="Float64"),
        }
    )
    before = patients.copy(deep=True)

    try:
        result = table1(patients, ["lab"], by="arm", standardized_differences=True)
    except TypeError as error:
        pytest.fail(f"A nonfinite exclusion leaving a singleton must not abort the table: {error}")

    assert result.loc[("lab", "Mean (SD)"), "arm=A"] == "1.0 (NA)"
    assert result.loc[("lab", "Mean (SD)"), "SMD"] == ""
    assert result.loc[("lab", "Missing"), "arm=A"] == "1 (50.0%)"
    pd.testing.assert_frame_equal(patients, before)


@pytest.mark.parametrize("command", [False, True], ids=["wrapper", "command"])
def test_table1_singleton_smd_through_public_callers(command: bool) -> None:
    patients = pd.DataFrame(
        {
            "arm": ["A", "A", "B", "B"],
            "lab": pd.Series([1, None, 3, 5], dtype="Int64"),
        }
    )
    before = patients.copy(deep=True)
    stata = OpenStata(patients)

    try:
        result = (
            stata.run("table1 lab, by(arm) nonnormal(lab) missing smd")
            if command
            else stata.table1(["lab"], by="arm", standardized_differences=True)
        )
    except TypeError as error:
        pytest.fail(f"Public Table 1 callers must preserve undefined singleton SMDs: {error}")

    summary = "Median [IQR]" if command else "Mean (SD)"
    assert result.loc[("lab", summary), "SMD"] == ""
    assert result.loc[("lab", "Missing"), "Overall"] == "1 (25.0%)"
    assert result.attrs["openstata_group_sizes"] == {"Overall": 4, "arm=A": 2, "arm=B": 2}
    pd.testing.assert_frame_equal(patients, before)


def test_table1_singleton_smd_does_not_prevent_export(tmp_path: Path) -> None:
    patients = pd.DataFrame(
        {
            "arm": ["A", "A", "B", "B"],
            "lab": pd.Series([1, None, 3, 5], dtype="UInt64"),
        }
    )
    before = patients.copy(deep=True)
    destination = tmp_path / "singleton.html"

    try:
        result = OpenStata(patients).export_table1(
            destination, ["lab"], by="arm", standardized_differences=True
        )
    except TypeError as error:
        pytest.fail(f"An undefined singleton SMD must not prevent export: {error}")

    assert result == destination.resolve()
    document = destination.read_text(encoding="utf-8")
    assert ">SMD<" in document
    assert "1.0 (NA)" in document
    assert "4.0 (1.4)" in document
    pd.testing.assert_frame_equal(patients, before)


def test_table1_smd_remains_undefined_with_three_groups() -> None:
    patients = pd.DataFrame(
        {"arm": ["A", "B", "C"], "lab": pd.Series([1, 3, 5], dtype="Int64")}
    )

    result = table1(patients, ["lab"], by="arm", standardized_differences=True)

    assert result.loc[("lab", "Mean (SD)"), "SMD"] == ""
    assert list(result.columns) == ["Overall", "arm=A", "arm=B", "arm=C", "SMD"]
