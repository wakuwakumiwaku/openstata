"""Publication-ready baseline characteristics tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from scipy import stats


def _check_columns(data: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise KeyError(f"Columns not found: {', '.join(missing)}")


def _levels(series: pd.Series) -> list[object]:
    observed = series.dropna()
    if isinstance(series.dtype, pd.CategoricalDtype):
        present = set(observed.tolist())
        return [value for value in series.cat.categories if value in present]
    return list(pd.unique(observed))


def _number(value: float, digits: int) -> str:
    return "" if pd.isna(value) else f"{value:.{digits}f}"


def _mean_sd(series: pd.Series, digits: int) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return ""
    mean = _number(float(values.mean()), digits)
    sd = _number(float(values.std(ddof=1)), digits) if len(values) > 1 else "NA"
    return f"{mean} ({sd})"


def _median_iqr(series: pd.Series, digits: int) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return ""
    q1, median, q3 = values.quantile([0.25, 0.50, 0.75]).tolist()
    return f"{_number(median, digits)} [{_number(q1, digits)}, {_number(q3, digits)}]"


def _n_percent(count: int, denominator: int, digits: int) -> str:
    percent = count * 100 / denominator if denominator else np.nan
    return f"{count} ({_number(percent, digits)}%)" if denominator else f"{count} (NA%)"


def _continuous_pvalue(frame: pd.DataFrame, variable: str, by: str, nonnormal: bool) -> float:
    samples = [
        pd.to_numeric(frame.loc[frame[by] == level, variable], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy()
        for level in _levels(frame[by])
    ]
    samples = [sample for sample in samples if len(sample)]
    if len(samples) < 2:
        return np.nan
    try:
        if len(samples) == 2:
            if nonnormal:
                test = stats.mannwhitneyu(samples[0], samples[1], alternative="two-sided")
                return float(test.pvalue)
            return float(stats.ttest_ind(samples[0], samples[1], equal_var=False).pvalue)
        if nonnormal:
            return float(stats.kruskal(*samples).pvalue)
        return float(stats.f_oneway(*samples).pvalue)
    except ValueError:
        return np.nan


def _categorical_pvalue(frame: pd.DataFrame, variable: str, by: str) -> float:
    contingency = pd.crosstab(frame[variable], frame[by])
    if min(contingency.shape, default=0) < 2:
        return np.nan
    try:
        _, pvalue, _, expected = stats.chi2_contingency(contingency, correction=False)
        if contingency.shape == (2, 2) and (expected < 5).any():
            return float(stats.fisher_exact(contingency.to_numpy()).pvalue)
        return float(pvalue)
    except ValueError:
        return np.nan


def _continuous_smd(frame: pd.DataFrame, variable: str, by: str) -> float:
    levels = _levels(frame[by])
    if len(levels) != 2:
        return np.nan
    first = pd.to_numeric(frame.loc[frame[by] == levels[0], variable], errors="coerce").dropna()
    second = pd.to_numeric(frame.loc[frame[by] == levels[1], variable], errors="coerce").dropna()
    if first.empty or second.empty:
        return np.nan
    denominator = np.sqrt((first.var(ddof=1) + second.var(ddof=1)) / 2)
    difference = abs(float(second.mean() - first.mean()))
    if denominator == 0:
        return 0.0 if difference == 0 else np.nan
    return float(difference / denominator)


def _categorical_smd(frame: pd.DataFrame, variable: str, by: str, level: object) -> float:
    groups = _levels(frame[by])
    if len(groups) != 2:
        return np.nan
    probabilities: list[float] = []
    for group in groups:
        values = frame.loc[frame[by] == group, variable].dropna()
        if values.empty:
            return np.nan
        probabilities.append(float((values == level).mean()))
    denominator = np.sqrt(
        (probabilities[0] * (1 - probabilities[0]) + probabilities[1] * (1 - probabilities[1]))
        / 2
    )
    difference = abs(probabilities[1] - probabilities[0])
    if denominator == 0:
        return 0.0 if difference == 0 else np.nan
    return float(difference / denominator)


def _format_pvalue(value: float) -> str:
    if pd.isna(value):
        return ""
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _format_smd(value: float) -> str:
    return "" if pd.isna(value) else f"{value:.2f}"


def table1(
    data: pd.DataFrame,
    variables: Sequence[str] | None = None,
    *,
    by: str | None = None,
    categorical: Sequence[str] | None = None,
    nonnormal: Sequence[str] | None = None,
    labels: Mapping[str, str] | None = None,
    include_missing: bool = True,
    pvalues: bool = False,
    standardized_differences: bool = False,
    digits: int = 1,
) -> pd.DataFrame:
    """Build a baseline characteristics table for clinical research.

    Numeric variables use mean (SD), except variables listed in ``nonnormal``,
    which use median [IQR]. Object, categorical, and Boolean variables are
    categorical automatically. Numeric codes can be declared categorical.

    P-values use Welch's t test, one-way ANOVA, Mann-Whitney U, Kruskal-Wallis,
    chi-square, or Fisher's exact test as appropriate. Absolute standardized
    mean differences are reported only when ``by`` has exactly two groups.
    """

    if digits < 0:
        raise ValueError("digits must be non-negative")

    chosen = list(variables) if variables is not None else list(data.columns)
    if by is not None and by in chosen:
        chosen.remove(by)
    if not chosen:
        raise ValueError("No variables were selected")

    categorical_set = set(categorical or [])
    nonnormal_set = set(nonnormal or [])
    required = chosen + ([by] if by is not None else [])
    _check_columns(data, required)
    undeclared = (categorical_set | nonnormal_set) - set(chosen)
    if undeclared:
        raise ValueError(f"Options reference unselected variables: {', '.join(sorted(undeclared))}")

    frame = data.copy()
    if by is not None:
        frame = frame.loc[frame[by].notna()].copy()
    if frame.empty:
        raise ValueError("No observations remain in the analysis population")

    group_frames: list[tuple[str, pd.DataFrame]] = [("Overall", frame)]
    if by is not None:
        group_frames.extend(
            (f"{by}={level}", frame.loc[frame[by] == level]) for level in _levels(frame[by])
        )

    label_map = dict(labels or {})
    rows: list[dict[str, str]] = []
    index: list[tuple[str, str]] = []

    for variable in chosen:
        series = frame[variable]
        is_categorical = (
            variable in categorical_set
            or not is_numeric_dtype(series)
            or is_bool_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
        )
        display_name = label_map.get(variable, variable)
        first_variable_row = len(rows)

        if is_categorical:
            levels = _levels(series)
            for level in levels:
                row = {
                    group_name: _n_percent(
                        int((group[variable] == level).sum()),
                        int(group[variable].notna().sum()),
                        digits,
                    )
                    for group_name, group in group_frames
                }
                if standardized_differences and by is not None:
                    row["SMD"] = _format_smd(_categorical_smd(frame, variable, by, level))
                rows.append(row)
                index.append((display_name, str(level)))
        else:
            summary_label = "Median [IQR]" if variable in nonnormal_set else "Mean (SD)"
            formatter = _median_iqr if variable in nonnormal_set else _mean_sd
            row = {
                group_name: formatter(group[variable], digits)
                for group_name, group in group_frames
            }
            if standardized_differences and by is not None:
                row["SMD"] = _format_smd(_continuous_smd(frame, variable, by))
            rows.append(row)
            index.append((display_name, summary_label))

        missing_count = int(series.isna().sum())
        if include_missing and missing_count:
            missing_row = {
                group_name: _n_percent(
                    int(group[variable].isna().sum()),
                    len(group),
                    digits,
                )
                for group_name, group in group_frames
            }
            if standardized_differences and by is not None:
                missing_row["SMD"] = ""
            rows.append(missing_row)
            index.append((display_name, "Missing"))

        if len(rows) == first_variable_row:
            rows.append({group_name: "" for group_name, _ in group_frames})
            index.append((display_name, "No observed values"))

        if pvalues and by is not None:
            pvalue = (
                _categorical_pvalue(frame, variable, by)
                if is_categorical
                else _continuous_pvalue(frame, variable, by, variable in nonnormal_set)
            )
            rows[first_variable_row]["P-value"] = _format_pvalue(pvalue)
            for position in range(first_variable_row + 1, len(rows)):
                rows[position]["P-value"] = ""

    result = pd.DataFrame(rows)
    ordered_columns = [name for name, _ in group_frames]
    if pvalues and by is not None:
        ordered_columns.append("P-value")
    if standardized_differences and by is not None:
        ordered_columns.append("SMD")
    result = result.reindex(columns=ordered_columns, fill_value="")
    result.index = pd.MultiIndex.from_tuples(index, names=["Variable", "Level"])
    result.attrs["openstata_group_sizes"] = {
        group_name: len(group) for group_name, group in group_frames
    }
    result.attrs["openstata_group_variable"] = by
    return result
