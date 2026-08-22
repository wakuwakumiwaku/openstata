from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from openstata import read_data, write_data


def test_csv_round_trip(tmp_path: Path) -> None:
    destination = tmp_path / "patients.csv"
    source = pd.DataFrame({"patient_id": [1, 2], "age": [61, 74]})

    write_data(source, destination)
    restored = read_data(destination)

    pd.testing.assert_frame_equal(restored, source)


def test_write_data_creates_missing_parent_directories(tmp_path: Path) -> None:
    destination = tmp_path / "exports" / "nested" / "patients.csv"
    source = pd.DataFrame({"patient_id": [1], "age": [61]})

    write_data(source, destination)

    pd.testing.assert_frame_equal(read_data(destination), source)


def test_data_paths_expand_user_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    source = pd.DataFrame({"patient_id": [1], "age": [61]})

    write_data(source, "~/patients.csv")

    pd.testing.assert_frame_equal(read_data("~/patients.csv"), source)


@pytest.mark.parametrize("suffix", [".csv.gz", ".tsv.gz"])
def test_compressed_delimited_round_trip(tmp_path: Path, suffix: str) -> None:
    destination = tmp_path / f"patients{suffix}"
    source = pd.DataFrame({"patient_id": [1, 2], "age": [61, 74]})

    write_data(source, destination)
    restored = read_data(destination)

    pd.testing.assert_frame_equal(restored, source)


def test_stata_round_trip(tmp_path: Path) -> None:
    destination = tmp_path / "patients.dta"
    source = pd.DataFrame({"patient_id": [1, 2], "age": [61.0, 74.0]})

    write_data(source, destination, version=118)
    restored = read_data(destination)

    pd.testing.assert_frame_equal(restored, source, check_dtype=False)


def test_unsupported_format_is_explicit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        read_data(tmp_path / "patients.xlsx")
