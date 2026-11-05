from __future__ import annotations

import pandas as pd
import pytest

from openstata import StataFrame, table1


@pytest.mark.parametrize("values", [[1, "1", None], ["Missing", "yes", None]])
def test_rejects_ambiguous_categorical_row_labels(values: list[object]) -> None:
    data = pd.DataFrame({"status": values})
    original = data.copy(deep=True)
    with pytest.raises(ValueError, match="[Ll]evel labels.*unique"):
        table1(data)
    pd.testing.assert_frame_equal(data, original)


def test_missing_is_an_ordinary_category_when_no_missing_row_is_emitted() -> None:
    result = table1(pd.DataFrame({"status": ["Missing", None]}), include_missing=False)
    assert result.index.tolist() == [("status", "Missing")]
    assert result.iloc[0, 0] == "1 (100.0%)"


def test_rejects_display_labels_that_merge_distinct_variables_in_exports(tmp_path) -> None:
    data = pd.DataFrame({"age": [30, 40], "weight": [60, 70]})
    destination = tmp_path / "table.html"
    with pytest.raises(ValueError, match="[Vv]ariable labels.*unique"):
        StataFrame(data).export_table1(
            destination, labels={"age": "Measurement", "weight": "Measurement"}
        )
    assert not destination.exists()


def test_unique_display_and_category_labels_preserve_counts() -> None:
    data = pd.DataFrame({"status": ["yes", "no", None], "age": [20, 30, 40]})
    result = table1(data, labels={"status": "Status", "age": "Age"})
    assert result.index.is_unique
    assert result.loc[("Status", "yes"), "Overall"] == "1 (50.0%)"
    assert result.loc[("Status", "Missing"), "Overall"] == "1 (33.3%)"
    assert result.loc[("Age", "Mean (SD)"), "Overall"] == "30.0 (10.0)"
