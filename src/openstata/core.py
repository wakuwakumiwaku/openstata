"""Stata-inspired descriptive statistics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype
from scipy import stats

PercentMode = Literal["count", "row", "column", "cell"]
ProportionMethod = Literal["exact", "wald", "wilson", "agresti", "jeffreys"]
_PROPORTION_METHODS = {"exact", "wald", "wilson", "agresti", "jeffreys"}
_MISSING_LABEL = "<missing>"


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
        columns = [
            column
            for column in data.select_dtypes(include="number").columns
            if not is_complex_dtype(data[column])
        ]
    else:
        columns = _check_columns(data, variables)
        nonnumeric = [
            column
            for column in columns
            if not is_numeric_dtype(data[column]) or is_complex_dtype(data[column])
        ]
        if nonnumeric:
            raise TypeError(
                "Analysis requires real-valued numeric columns: " + ", ".join(nonnumeric)
            )
    if not columns:
        raise ValueError("No numeric variables were selected")
    return columns


def _confidence_level(level: float) -> float:
    try:
        confidence_level = float(level)
    except (TypeError, ValueError) as error:
        raise ValueError("level must be a number between 0 and 100") from error
    if not np.isfinite(confidence_level) or not 0 < confidence_level < 100:
        raise ValueError("level must be a number between 0 and 100")
    return confidence_level


def _finite_numeric(series: pd.Series) -> pd.Series:
    numeric = cast(pd.Series, pd.to_numeric(series, errors="coerce"))
    return numeric.replace([np.inf, -np.inf], np.nan).dropna()


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
        series = cast(pd.Series, data[column])
        values = _finite_numeric(series)
        record: dict[str, float | int | str] = {
            "Variable": column,
            "Obs": int(values.size),
            "Missing": int(len(series) - values.size),
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


def ci_mean(
    data: pd.DataFrame,
    variables: Sequence[str] | None = None,
    *,
    level: float = 95.0,
) -> pd.DataFrame:
    """Estimate means and Student's t confidence intervals.

    Missing and non-finite values are excluded independently for each variable.
    Intervals and standard errors are undefined when fewer than two observations
    remain. ``level`` is expressed as a percentage to match Stata's ``level()``
    option.
    """

    confidence_level = _confidence_level(level)
    columns = _numeric_variables(data, variables)
    records: list[dict[str, float | int | str]] = []
    probability = 0.5 + confidence_level / 200

    for column in columns:
        values = _finite_numeric(data[column])
        observations = int(values.size)
        mean = float(values.mean()) if observations else np.nan
        standard_error = np.nan
        lower = np.nan
        upper = np.nan

        if observations > 1:
            standard_error = float(values.std(ddof=1) / np.sqrt(observations))
            critical_value = float(stats.t.ppf(probability, df=observations - 1))
            margin = critical_value * standard_error
            lower = mean - margin
            upper = mean + margin

        records.append(
            {
                "Variable": column,
                "Obs": observations,
                "Mean": mean,
                "Std. err.": standard_error,
                "CI lower": lower,
                "CI upper": upper,
            }
        )

    result = pd.DataFrame.from_records(records).set_index("Variable")
    result.attrs["confidence_level"] = confidence_level
    return result


def ci_proportion(
    data: pd.DataFrame,
    variables: Sequence[str] | None = None,
    *,
    level: float = 95.0,
    method: ProportionMethod = "exact",
) -> pd.DataFrame:
    """Estimate binary proportions and binomial confidence intervals.

    Variables must contain only 0 and 1 after missing and non-finite values are
    excluded. If ``variables`` is omitted, all binary numeric and Boolean columns
    are selected. ``method`` may be ``exact`` (Clopper-Pearson), ``wald``,
    ``wilson``, ``agresti`` (Agresti-Coull), or ``jeffreys``.
    """

    confidence_level = _confidence_level(level)
    if method not in _PROPORTION_METHODS:
        choices = ", ".join(sorted(_PROPORTION_METHODS))
        raise ValueError(f"method must be one of: {choices}")

    if variables is None:
        columns = []
        for column in data.columns:
            series = data[column]
            if is_complex_dtype(series) or not (
                is_numeric_dtype(series) or is_bool_dtype(series)
            ):
                continue
            observed = _finite_numeric(series)
            if not observed.empty and observed.isin([0, 1]).all():
                columns.append(column)
        if not columns:
            raise ValueError("No binary 0/1 variables were found")
    else:
        columns = _check_columns(data, variables)
        nonnumeric = [
            column
            for column in columns
            if is_complex_dtype(data[column])
            or not (is_numeric_dtype(data[column]) or is_bool_dtype(data[column]))
        ]
        if nonnumeric:
            raise TypeError(
                "ci_proportion requires real-valued numeric or Boolean columns: "
                + ", ".join(nonnumeric)
            )
        nonbinary = [
            column
            for column in columns
            if not _finite_numeric(data[column]).isin([0, 1]).all()
        ]
        if nonbinary:
            raise ValueError(
                "ci_proportion requires values coded as 0 and 1: " + ", ".join(nonbinary)
            )
        if not columns:
            raise ValueError("No variables were selected")

    alpha = 1 - confidence_level / 100
    critical_value = float(stats.norm.ppf(1 - alpha / 2))
    records: list[dict[str, float | int | str]] = []

    for column in columns:
        values = _finite_numeric(data[column])
        observations = int(values.size)
        successes = int(values.sum()) if observations else 0
        proportion = successes / observations if observations else np.nan
        standard_error = (
            float(np.sqrt(proportion * (1 - proportion) / observations))
            if observations
            else np.nan
        )
        lower = np.nan
        upper = np.nan

        if observations:
            failures = observations - successes
            if method == "exact":
                lower = (
                    float(stats.beta.ppf(alpha / 2, successes, failures + 1))
                    if successes
                    else 0.0
                )
                upper = (
                    float(stats.beta.ppf(1 - alpha / 2, successes + 1, failures))
                    if failures
                    else 1.0
                )
            elif method == "wald":
                margin = critical_value * standard_error
                lower = proportion - margin
                upper = proportion + margin
            elif method == "wilson":
                denominator = 1 + critical_value**2 / observations
                center = (
                    proportion + critical_value**2 / (2 * observations)
                ) / denominator
                margin = (
                    critical_value
                    * np.sqrt(
                        proportion * (1 - proportion) / observations
                        + critical_value**2 / (4 * observations**2)
                    )
                    / denominator
                )
                lower = center - margin
                upper = center + margin
            elif method == "agresti":
                adjusted_n = observations + critical_value**2
                adjusted_p = (successes + critical_value**2 / 2) / adjusted_n
                margin = critical_value * np.sqrt(adjusted_p * (1 - adjusted_p) / adjusted_n)
                lower = adjusted_p - margin
                upper = adjusted_p + margin
            else:
                lower = float(stats.beta.ppf(alpha / 2, successes + 0.5, failures + 0.5))
                upper = float(
                    stats.beta.ppf(1 - alpha / 2, successes + 0.5, failures + 0.5)
                )

        records.append(
            {
                "Variable": column,
                "Obs": observations,
                "Proportion": proportion,
                "Std. err.": standard_error,
                "CI lower": lower,
                "CI upper": upper,
            }
        )

    result = pd.DataFrame.from_records(records).set_index("Variable")
    result.attrs["confidence_level"] = confidence_level
    result.attrs["method"] = method
    return result


def _with_missing(series: pd.Series, include_missing: bool) -> pd.Series:
    if include_missing:
        return series.astype("object").where(series.notna(), _MISSING_LABEL)
    return series.dropna()


def _reject_missing_label_collision(series: pd.Series, name: str) -> None:
    if any(isinstance(value, str) and value == _MISSING_LABEL for value in series.dropna()):
        raise ValueError(f"{name} contains reserved missing label {_MISSING_LABEL!r}")


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
        if missing:
            _reject_missing_label_collision(data[row], row)
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
        _reject_missing_label_collision(subset[row], row)
        _reject_missing_label_collision(subset[column], column)
        subset[row] = subset[row].astype("object").where(subset[row].notna(), _MISSING_LABEL)
        subset[column] = subset[column].astype("object").where(
            subset[column].notna(), _MISSING_LABEL
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
