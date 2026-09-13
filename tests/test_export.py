from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from docx import Document
from openpyxl import load_workbook

from openstata import OpenStata, export_table1, table1


@pytest.fixture
def baseline() -> pd.DataFrame:
    patients = pd.DataFrame(
        {
            "arm": ["Control", "Control", "Control", "Treatment", "Treatment", "Treatment"],
            "age": [50, 60, 55, 70, 80, 75],
            "crp": [2.0, 5.0, None, 3.0, 7.0, 10.0],
            "sex": ["Female", "Male", "Female", "Female", "Female", "Male"],
        }
    )
    return table1(
        patients,
        ["age", "crp", "sex"],
        by="arm",
        nonnormal=["crp"],
        pvalues=True,
        standardized_differences=True,
    )


def test_html_export_is_standalone_and_styled(
    baseline: pd.DataFrame, tmp_path: Path
) -> None:
    destination = tmp_path / "table1.html"

    result = export_table1(
        baseline,
        destination,
        title="Table 1. Trial population",
        subtitle="Intention-to-treat cohort",
        footnotes=["Synthetic data only.", "A < B & C."],
    )

    document = destination.read_text(encoding="utf-8")
    assert result == destination.resolve()
    assert "<!doctype html>" in document
    assert "Table 1. Trial population" in document
    assert "Intention-to-treat cohort" in document
    assert "A &lt; B &amp; C." in document
    assert "@media print" in document
    assert "rowspan=\"2\"" in document
    assert ">Control (n=3)<" in document
    assert "arm=Control" not in document
    assert "https://" not in document


def test_export_protects_existing_files(baseline: pd.DataFrame, tmp_path: Path) -> None:
    destination = tmp_path / "table1.html"
    destination.write_text("keep me", encoding="utf-8")

    with pytest.raises(FileExistsError):
        export_table1(baseline, destination)

    export_table1(baseline, destination, overwrite=True)
    assert destination.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_export_rejects_bad_style_and_format(
    baseline: pd.DataFrame, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="style"):
        export_table1(baseline, tmp_path / "table.html", style="neon")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unsupported"):
        export_table1(baseline, tmp_path / "table.pdf")


def test_excel_export_has_professional_workbook_features(
    baseline: pd.DataFrame, tmp_path: Path
) -> None:
    destination = tmp_path / "table1.xlsx"

    export_table1(
        baseline,
        destination,
        title="Table 1. Trial population",
        subtitle="Intention-to-treat cohort",
        style="minimal",
    )

    workbook = load_workbook(destination)
    sheet = workbook["Table 1"]
    assert sheet["A1"].value == "Table 1. Trial population"
    assert sheet.freeze_panes == "C5"
    assert sheet.sheet_view.showGridLines is False
    assert any(str(cell_range).startswith("A") for cell_range in sheet.merged_cells.ranges)
    assert any(str(cell_range).startswith("F") for cell_range in sheet.merged_cells.ranges)
    assert sheet["A4"].value == "Variable"
    assert sheet["C4"].value == "Overall (n=6)"
    assert sheet["D4"].value == "Control (n=3)"
    assert sheet.oddFooter.center.text == "OpenStata"


@pytest.mark.parametrize(
    "text",
    ["=1+1", "#NULL!", "#DIV/0!", "#VALUE!", "#REF!", "#NAME?", "#NUM!", "#N/A"],
)
@pytest.mark.parametrize("row_count", [1, 2], ids=["single-row", "merged-rows"])
@pytest.mark.parametrize("data_only", [False, True], ids=["formulas", "cached-values"])
def test_excel_export_preserves_literal_text(
    tmp_path: Path, text: str, row_count: int, data_only: bool
) -> None:
    table = pd.DataFrame(
        [[text, 12.5, None], [None, 0.0, None]][:row_count],
        index=pd.MultiIndex.from_tuples([(text, text), (text, "Other")][:row_count]),
        # Group headings drop the prefix before the first equals sign.
        columns=[f"arm={text}", "Numeric", "Missing"],
    )
    destination = tmp_path / "literal-text.xlsx"

    export_table1(table, destination, title=text, subtitle=text, footnotes=[text])

    workbook = load_workbook(destination, data_only=data_only)
    try:
        sheet = workbook["Table 1"]
        expected = {
            "A1": text,
            "A2": text,
            "C4": text,
            "A5": text,
            "B5": text,
            "C5": text,
            "D5": "12.5",
            f"A{row_count + 7}": f"1. {text}",
        }
        actual = {
            address: (sheet[address].value, sheet[address].data_type) for address in expected
        }
        assert actual == {address: (value, "s") for address, value in expected.items()}
        assert sheet["E5"].value is None
        assert all(cell.data_type not in {"f", "e"} for row in sheet for cell in row)

        merged_ranges = {str(cell_range) for cell_range in sheet.merged_cells.ranges}
        assert "A1:E1" in merged_ranges
        assert "A2:E2" in merged_ranges
        assert sheet["A1"].font.bold is True
        assert sheet["A2"].font.italic is True
        assert sheet["C4"].font.bold is True
        assert sheet["A5"].font.bold is True
        assert sheet["C5"].alignment.horizontal == "right"
        assert sheet.freeze_panes == "C5"
        if row_count == 2:
            assert "A5:A6" in merged_ranges
            assert "C5:C6" in merged_ranges
            assert sheet["A5"].alignment.vertical == "center"
            assert sheet["C5"].alignment.vertical == "center"
            assert sheet["A6"].value is None
            assert sheet["C6"].value is None
            assert sheet["D6"].value == "0.0"
        else:
            assert "A5:A6" not in merged_ranges
            assert "C5:C6" not in merged_ranges
    finally:
        workbook.close()


def test_word_export_is_editable_and_structured(
    baseline: pd.DataFrame, tmp_path: Path
) -> None:
    destination = tmp_path / "table1.docx"

    export_table1(
        baseline,
        destination,
        title="Table 1. Trial population",
        subtitle="Intention-to-treat cohort",
        style="journal",
    )

    document = Document(destination)
    assert document.paragraphs[0].text == "Table 1. Trial population"
    assert document.paragraphs[1].text == "Intention-to-treat cohort"
    assert len(document.tables) == 1
    headers = [cell.text for cell in document.tables[0].rows[0].cells]
    assert headers[:4] == [
        "Variable",
        "Level / statistic",
        "Overall (n=6)",
        "Control (n=3)",
    ]
    assert any(paragraph.text == "Notes" for paragraph in document.paragraphs)


def test_wrapper_builds_and_exports_in_one_step(tmp_path: Path) -> None:
    patients = pd.DataFrame(
        {
            "arm": ["A", "A", "B", "B"],
            "age": [50, 60, 55, 65],
            "sex": ["F", "M", "F", "F"],
        }
    )
    destination = tmp_path / "wrapper.html"

    result = OpenStata(patients).export_table1(
        destination,
        ["age", "sex"],
        by="arm",
        pvalues=True,
        title="Participant characteristics",
    )

    assert result == destination.resolve()
    assert "Participant characteristics" in destination.read_text(encoding="utf-8")
