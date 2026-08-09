"""Command-line interface for OpenStata."""

from __future__ import annotations

import argparse
import json
import shlex
from collections.abc import Sequence

from openstata.commands import OpenStata
from openstata.export import export_table1
from openstata.io import read_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openstata",
        description="Run Stata-style clinical statistics and export professional Table 1 files.",
    )
    parser.add_argument("data", help="Input CSV, TSV, DTA, or Parquet dataset")
    parser.add_argument(
        "command",
        help='Command, for example: "summarize age bmi, detail" or "ci means age bmi"',
    )
    parser.add_argument(
        "--format",
        choices=("table", "csv", "json"),
        default="table",
        help="Console output format",
    )
    parser.add_argument(
        "--output",
        help="Export a table1/baseline command to a styled .html, .xlsx, or .docx file",
    )
    parser.add_argument(
        "--title",
        default="Table 1. Baseline characteristics",
        help="Title used in an exported baseline table",
    )
    parser.add_argument("--subtitle", help="Optional subtitle used in an exported table")
    parser.add_argument(
        "--footnote",
        action="append",
        help="Custom footnote; repeat the option to add multiple notes",
    )
    parser.add_argument(
        "--style",
        choices=("clinical", "journal", "minimal"),
        default="clinical",
        help="Visual export theme",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing export file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = OpenStata(read_data(args.data)).run(args.command)

    if args.output:
        command_tokens = shlex.split(args.command.partition(",")[0])
        command_name = command_tokens[0].lower() if command_tokens else ""
        if command_name not in {"table1", "baseline"}:
            parser.error("--output supports table1 or baseline commands")
        if args.format != "table":
            parser.error("--format cannot be combined with --output")
        destination = export_table1(
            result,
            args.output,
            title=args.title,
            subtitle=args.subtitle,
            footnotes=args.footnote,
            style=args.style,
            overwrite=args.overwrite,
        )
        print(f"Exported baseline table to {destination}")
        return 0

    if args.format == "csv":
        print(result.to_csv())
    elif args.format == "json":
        print(json.dumps(result.reset_index().to_dict(orient="records"), default=str))
    else:
        print(result.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
