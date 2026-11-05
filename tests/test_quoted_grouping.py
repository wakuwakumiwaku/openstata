from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

import openstata.commands as commands
from openstata import StataFrame, table1


def test_import_provenance() -> None:
    if root := os.environ.get("EXPECTED_SOURCE_ROOT"):
        assert Path(commands.__file__).resolve().is_relative_to(Path(root).resolve())


@pytest.mark.parametrize(("name", "quote"), [
    ("treatment", ""), ("treatment", "'"), ("treatment", '"'),
    ("treatment arm", "'"), ("treatment arm", '"'),
])
def test_command_grouping_uses_shell_quoted_column_name(name, quote):
    data = pd.DataFrame({"age": [20, 30, 40, 50], name: ["A", "A", "B", "B"]})
    expected = table1(data, ["age"], by=name)
    try:
        actual = StataFrame(data).run(f"table1 age, by({quote}{name}{quote})")
    except KeyError as exc:
        pytest.fail(f"quoted grouping variable was not unquoted: {exc}")
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("option", ["by", "by()", "by('')", "by(treatment arm)"])
def test_grouping_requires_one_nonempty_name(option):
    data = pd.DataFrame({"age": [20, 30], "treatment": ["A", "B"]})
    error = None
    try:
        StataFrame(data).run(f"table1 age, {option}")
    except (ValueError, KeyError) as exc:
        error = exc
    assert isinstance(error, ValueError)
    assert "exactly one grouping variable" in str(error)
