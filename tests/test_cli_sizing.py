"""The `solarest size` command, and the messy files it has to read."""

from __future__ import annotations

import numpy as np
import pytest

from solarest.cli import _parse_monthly, _read_interval_file, main


# ------------------------------------------------------------- bill parsing


def test_twelve_monthly_totals_are_parsed():
    parsed = _parse_monthly("420,380,340,280,240,210,205,220,260,310,380,430")
    assert parsed == [420, 380, 340, 280, 240, 210, 205, 220, 260, 310, 380, 430]


def test_whitespace_between_totals_is_tolerated():
    assert _parse_monthly(" 100, 200 ,300,400,500,600,700,800,900,1000,1100,1200")[0] == 100


def test_the_wrong_number_of_months_is_rejected():
    with pytest.raises(ValueError, match="needs 12 values"):
        _parse_monthly("100,200,300")


def test_non_numeric_bills_are_rejected():
    with pytest.raises(ValueError, match="must be numeric"):
        _parse_monthly("100,200,three,400,500,600,700,800,900,1000,1100,1200")


def test_no_bills_means_no_bills():
    assert _parse_monthly(None) is None


# ----------------------------------------------------------- meter parsing


def write(tmp_path, name: str, text: str):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_one_reading_per_line_is_read(tmp_path):
    path = write(tmp_path, "plain.csv", "0.4\n0.5\n0.6\n")
    assert _read_interval_file(path) == [0.4, 0.5, 0.6]


def test_a_timestamp_column_is_ignored(tmp_path):
    path = write(
        tmp_path,
        "meter.csv",
        "timestamp,kwh\n2024-01-01T00:00,0.4\n2024-01-01T00:30,0.5\n",
    )
    assert _read_interval_file(path) == [0.4, 0.5]


def test_headers_blank_lines_and_comments_are_skipped(tmp_path):
    path = write(
        tmp_path,
        "messy.csv",
        "# exported 2024-01-01\n\nMPAN,Reading (kWh)\n\n1200001,0.4\n1200001,0.5\n\n",
    )
    # The MPAN row's last numeric field is the reading, as intended.
    assert _read_interval_file(path) == [0.4, 0.5]


def test_a_byte_order_mark_does_not_break_the_first_row(tmp_path):
    """Spreadsheet exports on Windows routinely start with one."""
    path = tmp_path / "bom.csv"
    path.write_bytes(b"\xef\xbb\xbftimestamp,kwh\n2024-01-01,0.4\n")
    assert _read_interval_file(str(path)) == [0.4]


def test_a_file_with_no_readings_is_rejected(tmp_path):
    path = write(tmp_path, "empty.csv", "# nothing here\n\n")
    with pytest.raises(ValueError, match="no numeric readings"):
        _read_interval_file(path)


def test_a_missing_file_raises(tmp_path):
    with pytest.raises(OSError):
        _read_interval_file(str(tmp_path / "absent.csv"))


def test_no_meter_file_means_none():
    assert _read_interval_file(None) is None


# ------------------------------------------------------------ the command


def test_the_size_command_runs_offline_and_reports(capsys):
    code = main([
        "size", "51.5", "-0.13", "--synthetic", "--years", "1",
        "--annual-kwh", "3500", "--battery-kwh", "5",
    ])
    out = capsys.readouterr().out

    assert code == 0
    assert "Not bought from the grid" in out
    assert "Suggested capacity" in out
    assert "assumption" in out  # the honesty note


def test_the_size_command_accepts_a_meter_file(tmp_path, capsys):
    readings = np.full(8760, 0.4)
    path = write(tmp_path, "year.csv", "\n".join(f"{v}" for v in readings))

    code = main([
        "size", "51.5", "-0.13", "--synthetic", "--years", "1",
        "--hourly-csv", path,
    ])
    out = capsys.readouterr().out

    assert code == 0
    assert "Metered interval data" in out


def test_the_size_command_emits_json_on_request(capsys):
    import json

    code = main([
        "size", "51.5", "-0.13", "--synthetic", "--years", "1",
        "--annual-kwh", "3500", "--json",
    ])
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["balance"]["avoided_import_kwh"] > 0
    assert payload["sizing_curve"]


def test_a_bad_meter_file_exits_non_zero(capsys):
    code = main([
        "size", "51.5", "-0.13", "--synthetic",
        "--hourly-csv", "/nonexistent/meter.csv",
    ])
    assert code == 1
    assert "error:" in capsys.readouterr().err


def test_demand_sources_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        main([
            "size", "51.5", "-0.13", "--synthetic",
            "--annual-kwh", "3500", "--monthly-kwh", "1,2,3,4,5,6,7,8,9,10,11,12",
        ])


def test_some_demand_figure_is_required():
    with pytest.raises(SystemExit):
        main(["size", "51.5", "-0.13", "--synthetic"])
