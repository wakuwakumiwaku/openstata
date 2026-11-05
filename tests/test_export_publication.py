from __future__ import annotations

import os
import stat
from pathlib import Path

import pandas as pd
import pytest

import openstata.export as exports
from openstata import table1


def test_import_provenance() -> None:
    if root := os.environ.get("EXPECTED_SOURCE_ROOT"):
        assert Path(exports.__file__).resolve().is_relative_to(Path(root).resolve())


def sample_table() -> pd.DataFrame:
    return table1(pd.DataFrame({"age": [20, 40]}))


@pytest.mark.parametrize(("suffix", "writer"), [
    ("html", "_write_html"), ("xlsx", "_write_xlsx"), ("docx", "_write_docx"),
])
@pytest.mark.parametrize("existing", [False, True])
def test_failed_export_does_not_publish_partial_report(
    tmp_path, monkeypatch, suffix, writer, existing
):
    destination = tmp_path / f"table.{suffix}"
    previous = b"previous complete report"
    if existing:
        destination.write_bytes(previous)

    def fail_after_write(table, path, **kwargs):
        path.write_bytes(b"partial report")
        raise OSError("simulated report writer failure")

    monkeypatch.setattr(exports, writer, fail_after_write)
    with pytest.raises(OSError, match="simulated report writer failure"):
        exports.export_table1(sample_table(), destination, overwrite=True)
    if existing:
        assert destination.read_bytes() == previous
    else:
        assert not destination.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ([destination.name] if existing else [])


@pytest.mark.parametrize("suffix", ["html", "xlsx", "docx"])
def test_successful_export_replaces_report_and_preserves_mode(tmp_path, suffix):
    destination = tmp_path / f"table.{suffix}"
    destination.write_bytes(b"old report")
    destination.chmod(0o640)
    result = exports.export_table1(sample_table(), destination, overwrite=True)
    assert result == destination.resolve()
    assert destination.read_bytes() != b"old report"
    assert stat.S_IMODE(destination.stat().st_mode) == 0o640
    if suffix == "html":
        assert "30.0" in destination.read_text()
    else:
        import zipfile
        with zipfile.ZipFile(destination) as archive:
            assert archive.testzip() is None
    assert sorted(p.name for p in tmp_path.iterdir()) == [destination.name]


def test_symlink_destination_remains_a_link(tmp_path):
    target = tmp_path / "actual.html"
    target.write_text("previous")
    link = tmp_path / "report.html"
    link.symlink_to(target.name)
    exports.export_table1(sample_table(), link, overwrite=True)
    assert link.is_symlink()
    assert "30.0" in target.read_text()


def test_default_overwrite_refusal_preserves_existing_report(tmp_path):
    destination = tmp_path / "table.html"
    destination.write_bytes(b"old report")
    with pytest.raises(FileExistsError):
        exports.export_table1(sample_table(), destination)
    assert destination.read_bytes() == b"old report"


@pytest.mark.parametrize("existing", [False, True])
def test_atomic_replacement_failure_keeps_destination(tmp_path, monkeypatch, existing):
    destination = tmp_path / "table.html"
    if existing:
        destination.write_bytes(b"previous report")

    def fail_replace(source, target):
        assert Path(source).is_file()
        raise OSError("simulated publication failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated publication failure"):
        exports.export_table1(sample_table(), destination, overwrite=True)
    if existing:
        assert destination.read_bytes() == b"previous report"
    else:
        assert not destination.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ([destination.name] if existing else [])
