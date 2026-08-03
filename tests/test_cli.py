from __future__ import annotations

from pathlib import Path

import pandas as pd

from openstata.cli import main


def test_cli_prints_summary_table(tmp_path: Path, capsys: object) -> None:
    source = tmp_path / "patients.csv"
    pd.DataFrame({"age": [50, 60, 70]}).to_csv(source, index=False)

    exit_code = main([str(source), "summarize age"])

    assert exit_code == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "Mean" in captured.out
    assert "60.0" in captured.out


def test_cli_supports_json_output(tmp_path: Path, capsys: object) -> None:
    source = tmp_path / "patients.csv"
    pd.DataFrame({"arm": ["A", "B"]}).to_csv(source, index=False)

    exit_code = main([str(source), "tabulate arm", "--format", "json"])

    assert exit_code == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert '"arm": "A"' in captured.out


def test_cli_exports_a_styled_baseline_table(tmp_path: Path, capsys: object) -> None:
    source = tmp_path / "patients.csv"
    destination = tmp_path / "baseline.html"
    pd.DataFrame(
        {
            "arm": ["Control", "Control", "Treatment", "Treatment"],
            "age": [50, 60, 55, 65],
            "sex": ["F", "M", "F", "F"],
        }
    ).to_csv(source, index=False)

    exit_code = main(
        [
            str(source),
            "table1 age sex, by(arm) pvalues smd",
            "--output",
            str(destination),
            "--title",
            "Table 1. Trial cohort",
            "--style",
            "clinical",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "Exported baseline table" in captured.out
    assert "Table 1. Trial cohort" in destination.read_text(encoding="utf-8")
