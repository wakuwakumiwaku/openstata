"""A small command layer for familiar Stata-style workflows."""

from __future__ import annotations

import re
import shlex
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd

from openstata.core import ProportionMethod, ci_mean, ci_proportion, summarize, tabulate
from openstata.export import ExportStyle
from openstata.export import export_table1 as write_table1
from openstata.table1 import table1

_OPTION_PATTERN = re.compile(r"([A-Za-z_]\w*)(?:\(([^)]*)\))?")


def _parse_options(text: str) -> dict[str, str | None]:
    options: dict[str, str | None] = {}
    position = 0
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position == len(text):
            break
        match = _OPTION_PATTERN.match(text, position)
        if match is None:
            raise ValueError(f"Could not parse option near: {text[position:]}")
        options[match.group(1).lower()] = match.group(2)
        position = match.end()
    return options


def _option_variables(options: dict[str, str | None], name: str) -> list[str]:
    value = options.get(name)
    return shlex.split(value) if value else []


def _reject_unknown(options: dict[str, str | None], allowed: set[str]) -> None:
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"Unknown options: {', '.join(sorted(unknown))}")


class StataFrame:
    """Wrap a DataFrame with functions and a compact Stata-like command runner."""

    def __init__(self, data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        self.data = data

    def summarize(
        self, variables: Sequence[str] | None = None, *, detail: bool = False
    ) -> pd.DataFrame:
        return summarize(self.data, variables, detail=detail)

    def ci_mean(
        self,
        variables: Sequence[str] | None = None,
        *,
        level: float = 95.0,
    ) -> pd.DataFrame:
        return ci_mean(self.data, variables, level=level)

    def ci_proportion(
        self,
        variables: Sequence[str] | None = None,
        *,
        level: float = 95.0,
        method: ProportionMethod = "exact",
    ) -> pd.DataFrame:
        return ci_proportion(self.data, variables, level=level, method=method)

    def tabulate(
        self,
        row: str,
        column: str | None = None,
        *,
        missing: bool = False,
        percent: str = "count",
    ) -> pd.DataFrame:
        return tabulate(
            self.data,
            row,
            column,
            missing=missing,
            percent=percent,  # type: ignore[arg-type]
        )

    def table1(self, variables: Sequence[str] | None = None, **kwargs: object) -> pd.DataFrame:
        return table1(self.data, variables, **kwargs)  # type: ignore[arg-type]

    def export_table1(
        self,
        destination: str | Path,
        variables: Sequence[str] | None = None,
        *,
        title: str = "Table 1. Baseline characteristics",
        subtitle: str | None = None,
        footnotes: Sequence[str] | None = None,
        style: ExportStyle = "clinical",
        overwrite: bool = False,
        **table_options: Any,
    ) -> Path:
        """Build and export a professional baseline table in one step."""

        result = table1(self.data, variables, **table_options)
        return write_table1(
            result,
            destination,
            title=title,
            subtitle=subtitle,
            footnotes=footnotes,
            style=style,
            overwrite=overwrite,
        )

    def run(self, command: str) -> pd.DataFrame:
        """Run supported descriptive, confidence-interval, and table commands."""

        body, separator, option_text = command.partition(",")
        tokens = shlex.split(body)
        if not tokens:
            raise ValueError("Command is empty")
        name = tokens[0].lower()
        variables = tokens[1:]
        options = _parse_options(option_text) if separator else {}

        if name == "ci":
            if not variables:
                raise ValueError("ci requires a subcommand")
            subcommand = variables[0].lower()
            raw_level = options.get("level", "95")
            if raw_level is None:
                raise ValueError("level() requires a numeric confidence percentage")
            try:
                level = float(raw_level)
            except ValueError as error:
                raise ValueError("level() requires a numeric confidence percentage") from error

            if subcommand in {"mean", "means"}:
                _reject_unknown(options, {"level"})
                return ci_mean(self.data, variables[1:] or None, level=level)

            if subcommand in {"prop", "proportion", "proportions"}:
                method_names = ("exact", "wald", "wilson", "agresti", "jeffreys")
                _reject_unknown(options, {"level", *method_names})
                methods = [method for method in method_names if method in options]
                if len(methods) > 1:
                    raise ValueError("Choose only one proportion interval method")
                method = cast(ProportionMethod, methods[0] if methods else "exact")
                return ci_proportion(
                    self.data,
                    variables[1:] or None,
                    level=level,
                    method=method,
                )

            raise ValueError("ci subcommand must be means or proportions")

        if name in {"summarize", "sum"}:
            _reject_unknown(options, {"detail"})
            return summarize(self.data, variables or None, detail="detail" in options)

        if name in {"tabulate", "tab"}:
            _reject_unknown(options, {"missing", "row", "column", "cell"})
            if len(variables) not in {1, 2}:
                raise ValueError("tabulate requires one or two variables")
            modes = [mode for mode in ("row", "column", "cell") if mode in options]
            if len(modes) > 1:
                raise ValueError("Choose only one percentage mode: row, column, or cell")
            return tabulate(
                self.data,
                variables[0],
                variables[1] if len(variables) == 2 else None,
                missing="missing" in options,
                percent=modes[0] if modes else "count",  # type: ignore[arg-type]
            )

        if name in {"table1", "baseline"}:
            allowed = {"by", "categorical", "nonnormal", "missing", "pvalues", "smd"}
            _reject_unknown(options, allowed)
            by = options.get("by")
            if by is not None:
                by = by.strip()
                if not by or len(shlex.split(by)) != 1:
                    raise ValueError("by() must contain exactly one grouping variable")
            return table1(
                self.data,
                variables or None,
                by=by,
                categorical=_option_variables(options, "categorical"),
                nonnormal=_option_variables(options, "nonnormal"),
                include_missing="missing" in options,
                pvalues="pvalues" in options,
                standardized_differences="smd" in options,
            )

        raise ValueError(f"Unsupported command: {name}")


OpenStata = StataFrame
