"""OpenStata: Stata-inspired clinical statistics for Python."""

from importlib.metadata import PackageNotFoundError, version

from openstata.commands import OpenStata, StataFrame
from openstata.core import ci_mean, ci_proportion, summarize, tabulate
from openstata.export import export_table1
from openstata.io import read_data, write_data
from openstata.table1 import table1

__all__ = [
    "OpenStata",
    "StataFrame",
    "__version__",
    "ci_mean",
    "ci_proportion",
    "export_table1",
    "read_data",
    "summarize",
    "table1",
    "tabulate",
    "write_data",
]

try:
    __version__ = version("openstata")
except PackageNotFoundError:
    __version__ = "0+unknown"
