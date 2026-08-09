from __future__ import annotations

import pandas as pd
import pytest

from openstata import OpenStata


@pytest.fixture
def stata() -> OpenStata:
    return OpenStata(
        pd.DataFrame(
            {
                "arm": ["A", "A", "B", "B"],
                "age": [40, 50, 60, 70],
                "female": [0, 1, 1, 1],
            }
        )
    )


def test_run_summarize_command(stata: OpenStata) -> None:
    result = stata.run("summarize age, detail")

    assert result.loc["age", "Mean"] == pytest.approx(55.0)
    assert "p50" in result.columns


def test_run_ci_means_command(stata: OpenStata) -> None:
    result = stata.run("ci means age, level(90)")

    assert result.loc["age", "Obs"] == 4
    assert result.loc["age", "Mean"] == pytest.approx(55.0)
    assert result.attrs["confidence_level"] == 90.0


def test_ci_mean_wrapper_method(stata: OpenStata) -> None:
    result = stata.ci_mean(["age"], level=99)

    assert result.loc["age", "Mean"] == pytest.approx(55.0)
    assert result.attrs["confidence_level"] == 99.0


def test_run_tabulate_command(stata: OpenStata) -> None:
    result = stata.run("tabulate arm female, row")

    assert result.loc["A", 0] == pytest.approx(50.0)
    assert result.loc["B", 1] == pytest.approx(100.0)


def test_run_baseline_command(stata: OpenStata) -> None:
    result = stata.run(
        "table1 age female, by(arm) categorical(female) missing pvalues smd"
    )

    assert "arm=A" in result.columns
    assert "P-value" in result.columns
    assert "SMD" in result.columns


def test_run_rejects_unknown_commands_and_options(stata: OpenStata) -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        stata.run("regress age female")
    with pytest.raises(ValueError, match="Unknown options"):
        stata.run("summarize age, typo")
    with pytest.raises(ValueError, match="ci subcommand"):
        stata.run("ci proportions female")
    with pytest.raises(ValueError, match="level"):
        stata.run("ci means age, level(ninety)")
