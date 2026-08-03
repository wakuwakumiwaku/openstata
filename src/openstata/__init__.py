"""OpenStata: Stata-inspired clinical statistics for Python."""

from openstata.commands import OpenStata, StataFrame
from openstata.core import summarize, tabulate
from openstata.io import read_data, write_data
from openstata.table1 import table1

__all__ = [
    "OpenStata",
    "StataFrame",
    "read_data",
    "summarize",
    "table1",
    "tabulate",
    "write_data",
]

__version__ = "0.1.0"
