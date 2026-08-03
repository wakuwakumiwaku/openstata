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
