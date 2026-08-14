from __future__ import annotations

import json
from importlib.metadata import version as package_version
from pathlib import Path

import pandas as pd
import pytest

from openstata.cli import main


def test_cli_prints_version(capsys: object) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])

    assert exit_info.value.code == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert captured.out.strip() == f"openstata {package_version('openstata')}"


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


def test_cli_json_uses_null_for_undefined_statistics(tmp_path: Path, capsys: object) -> None:
    source = tmp_path / "one_patient.csv"
    pd.DataFrame({"age": [50]}).to_csv(source, index=False)

    exit_code = main([str(source), "ci means age", "--format", "json"])

    assert exit_code == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "NaN" not in captured.out
    row = json.loads(captured.out)[0]
    assert row["Std. err."] is None
    assert row["CI lower"] is None
    assert row["CI upper"] is None


def test_cli_runs_ci_means_command(tmp_path: Path, capsys: object) -> None:
    source = tmp_path / "patients.csv"
    pd.DataFrame({"age": [50, 60, 70, 80]}).to_csv(source, index=False)

    exit_code = main([str(source), "ci means age, level(90)", "--format", "json"])

    assert exit_code == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert '"Mean": 65.0' in captured.out
    assert '"CI lower"' in captured.out
    assert '"CI upper"' in captured.out


def test_cli_runs_ci_proportions_command(tmp_path: Path, capsys: object) -> None:
    source = tmp_path / "patients.csv"
    pd.DataFrame({"female": [1, 0, 1, 1]}).to_csv(source, index=False)

    exit_code = main(
        [str(source), "ci proportions female, wilson level(90)", "--format", "json"]
    )

    assert exit_code == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert '"Proportion": 0.75' in captured.out
    assert '"CI lower"' in captured.out
    assert '"CI upper"' in captured.out


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
