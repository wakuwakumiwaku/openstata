from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from openstata import OpenStata, summarize
from openstata.cli import main


@pytest.mark.parametrize("caller", ["function", "wrapper", "command"])
@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        ("bool", [True, False, True, False]),
        ("boolean", [True, False, True, False, None]),
        ("boolean", [True, True, None]),
        ("bool", [False]),
        ("boolean", [None, None]),
    ],
)
def test_boolean_detail_matches_zero_one_summary(
    caller: str, dtype: str, values: list[bool | None]
) -> None:
    data = pd.DataFrame({"flag": pd.Series(values, dtype=dtype)})
    before = data.copy(deep=True)
    numeric = data.astype({"flag": "Int64"})
    expected = summarize(numeric, ["flag"], detail=True)

    if caller == "function":
        result = summarize(data, ["flag"], detail=True)
    elif caller == "wrapper":
        result = OpenStata(data).summarize(["flag"], detail=True)
    else:
        result = OpenStata(data).run("summarize flag, detail")

    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(data, before)
    assert result.loc["flag", "Obs"] == data["flag"].notna().sum()
    assert result.loc["flag", "Missing"] == data["flag"].isna().sum()
    if values[:4] == [True, False, True, False]:
        # Linear interpolation of sorted 0, 0, 1, 1, not Boolean subtraction.
        assert result.loc["flag", "p25"] == 0.0
        assert result.loc["flag", "p50"] == 0.5
        assert result.loc["flag", "p75"] == 1.0
        assert result.loc["flag", "Variance"] == pytest.approx(1 / 3)


def test_cli_summarizes_boolean_csv_with_detail(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    source = tmp_path / "flags.csv"
    pd.DataFrame({"flag": [True, False, True, False]}).to_csv(source, index=False)

    assert main([str(source), "summarize flag, detail", "--format", "json"]) == 0

    records = json.loads(capsys.readouterr().out)
    assert len(records) == 1
    assert records[0]["Obs"] == 4
    assert records[0]["Missing"] == 0
    assert records[0]["Mean"] == 0.5
    assert records[0]["p25"] == 0.0
    assert records[0]["p50"] == 0.5
    assert records[0]["p75"] == 1.0
