"""Command-line interface for OpenStata."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from openstata.commands import OpenStata
from openstata.io import read_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openstata",
        description="Run a Stata-style descriptive command on CSV, TSV, DTA, or Parquet data.",
    )
    parser.add_argument("data", help="Input dataset")
    parser.add_argument("command", help='Command, for example: "summarize age bmi, detail"')
    parser.add_argument(
        "--format",
        choices=("table", "csv", "json"),
        default="table",
        help="Output format",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = OpenStata(read_data(args.data)).run(args.command)
    if args.format == "csv":
        print(result.to_csv())
    elif args.format == "json":
        print(json.dumps(result.reset_index().to_dict(orient="records"), default=str))
    else:
        print(result.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
