"""Stata-inspired descriptive statistics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

PercentMode = Literal["count", "row", "column", "cell"]


def _check_columns(data: pd.DataFrame, variables: Sequence[str]) -> list[str]:
    columns = list(variables)
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise KeyError(f"Columns not found: {', '.join(missing)}")
    return columns


def _numeric_variables(
    data: pd.DataFrame, variables: Sequence[str] | None
) -> list[str]:
    if variables is None:
        columns = list(data.select_dtypes(include="number").columns)
    else:
        columns = _check_columns(data, variables)
        nonnumeric = [column for column in columns if not is_numeric_dtype(data[column])]
        if nonnumeric:
            raise TypeError(f"summarize requires numeric columns: {', '.join(nonnumeric)}")
    if not columns:
        raise ValueError("No numeric variables were selected")
    return columns


def summarize(
    data: pd.DataFrame,
    variables: Sequence[str] | None = None,
    *,
    detail: bool = False,
) -> pd.DataFrame:
    """Return Stata-like summary statistics for numeric variables.

    Parameters
    ----------
    data:
        Source data.
    variables:
        Variables to summarize. If omitted, all numeric variables are used.
    detail:
        Add percentiles, variance, skewness, and Pearson kurtosis.
    """

    columns = _numeric_variables(data, variables)
    records: list[dict[str, float | int | str]] = []

    for column in columns:
        values = data[column].dropna()
        record: dict[str, float | int | str] = {
            "Variable": column,
            "Obs": int(values.size),
            "Missing": int(data[column].isna().sum()),
            "Mean": float(values.mean()) if not values.empty else np.nan,
            "Std. dev.": float(values.std(ddof=1)) if values.size > 1 else np.nan,
            "Min": float(values.min()) if not values.empty else np.nan,
            "Max": float(values.max()) if not values.empty else np.nan,
        }
        if detail:
            quantiles = values.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
            for percentile, label in [
                (0.01, "p1"),
                (0.05, "p5"),
                (0.10, "p10"),
                (0.25, "p25"),
                (0.50, "p50"),
                (0.75, "p75"),
                (0.90, "p90"),
                (0.95, "p95"),
                (0.99, "p99"),
            ]:
                record[label] = float(quantiles.loc[percentile]) if not values.empty else np.nan
            record["Variance"] = float(values.var(ddof=1)) if values.size > 1 else np.nan
            record["Skewness"] = float(values.skew()) if values.size > 2 else np.nan
            record["Kurtosis"] = float(values.kurt() + 3) if values.size > 3 else np.nan
        records.append(record)

    return pd.DataFrame.from_records(records).set_index("Variable")


def _with_missing(series: pd.Series, include_missing: bool) -> pd.Series:
    if include_missing:
        return series.astype("object").where(series.notna(), "<missing>")
    return series.dropna()


def tabulate(
    data: pd.DataFrame,
    row: str,
    column: str | None = None,
    *,
    missing: bool = False,
    percent: PercentMode = "count",
) -> pd.DataFrame:
    """Create a one-way or two-way Stata-like frequency table.

    One-way tables contain frequency, percent, and cumulative percent. Two-way
    tables contain counts by default. Set ``percent`` to ``row``, ``column``, or
    ``cell`` to return the corresponding percentages instead.
    """

    names = [row] if column is None else [row, column]
    _check_columns(data, names)
    if percent not in {"count", "row", "column", "cell"}:
        raise ValueError("percent must be one of: count, row, column, cell")

    if column is None:
        values = _with_missing(data[row], missing)
        counts = values.value_counts(sort=False, dropna=False)
        total = int(counts.sum())
        percentages = counts.astype(float) * 100 / total if total else counts.astype(float)
        result = pd.DataFrame(
            {
                "Freq.": counts.astype(int),
                "Percent": percentages,
                "Cum.": percentages.cumsum(),
            }
        )
        result.index.name = row
        return result

    subset = data[[row, column]].copy()
    if missing:
        subset[row] = subset[row].astype("object").where(subset[row].notna(), "<missing>")
        subset[column] = subset[column].astype("object").where(
            subset[column].notna(), "<missing>"
        )
    else:
        subset = subset.dropna()

    counts = pd.crosstab(subset[row], subset[column], dropna=False)
    counts.index.name = row
    counts.columns.name = column

    if percent == "count":
        result = counts.copy()
        result["Total"] = result.sum(axis=1)
        total_row = result.sum(axis=0).to_frame().T
        total_row.index = pd.Index(["Total"], name=row)
        return pd.concat([result, total_row]).astype(int)

    numeric = counts.astype(float)
    if percent == "row":
        result = numeric.div(numeric.sum(axis=1).replace(0, np.nan), axis=0) * 100
    elif percent == "column":
        result = numeric.div(numeric.sum(axis=0).replace(0, np.nan), axis=1) * 100
    else:
        grand_total = numeric.to_numpy().sum()
        result = numeric * 100 / grand_total if grand_total else numeric
    result.index.name = row
    result.columns.name = column
    return result
