"""Tests for occforecast.load using a small hand-made CSV."""

import pandas as pd
import pytest

from occforecast.load import audit, load_cached, load_occupancy, parse_timestamps

CSV = """id,deviceid,floorspaceid,occupancystatus,headcount,collecteddate
3,d,A,NotOccupied,0,2024-03-01 09:10:00+11
1,d,A,CurrentlyOccupied,2,2024-03-01 09:00:00+11
2,d,A,CurrentlyOccupied,0,2024-03-01 09:05:00+11
4,d,B,CurrentlyOccupied,1,not-a-date
5,d,B,CurrentlyOccupied,-1,2024-03-01 09:00:00+11
6,d,B,Unknown,1,2024-03-01 09:00:00+11
7,d,B,CurrentlyOccupied,3,2024-03-01 09:00:00+11
"""


@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "occ.csv"
    p.write_text(CSV)
    return p


def test_invalid_rows_dropped(csv_path):
    df = load_occupancy(csv_path)
    assert df.attrs["n_raw"] == 7
    assert df.attrs["n_dropped"] == 3          # bad date, negative count, bad status
    assert len(df) == 4


def test_sorted_and_utc(csv_path):
    df = load_occupancy(csv_path)
    a = df[df["space"] == "A"]
    assert a["time"].is_monotonic_increasing
    assert str(a["time"].dt.tz) == "UTC"
    assert a["time"].iloc[0] == pd.Timestamp("2024-02-29 22:00", tz="UTC")  # 09:00 +11


def test_missing_column_raises(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("floorspaceid,headcount\nA,1\n")
    with pytest.raises(ValueError, match="collecteddate"):
        load_occupancy(p)


def test_audit_flags_status_mismatch(csv_path):
    report = audit(load_occupancy(csv_path))
    # Row 2: CurrentlyOccupied with headcount 0 is inconsistent.
    assert report.loc["A", "status_mismatch"] == 1
    assert report.loc["B", "status_mismatch"] == 0
    assert report.loc["A", "max_headcount"] == 2


def test_load_cached_reuses_saved_copy(csv_path, tmp_path):
    cache = tmp_path / "occ.clean.pkl"
    first = load_cached(csv_path, cache=cache)
    second = load_cached(csv_path, cache=cache)
    assert first.attrs["from_cache"] is False and second.attrs["from_cache"] is True
    pd.testing.assert_frame_equal(first, second)
    assert second.attrs["n_raw"] == 7          # audit counts survive the cache


def test_parse_timestamps_matches_pandas():
    text = pd.Series(["2023-11-24 00:36:52+11", "2024-06-01 09:00:00+10",
                      "2024-03-01 10:00:00+05:30", "2024-03-01 10:00:00+0000",
                      "not-a-date", None])
    fast = parse_timestamps(text)
    slow = pd.to_datetime(text, utc=True, errors="coerce", format="mixed")
    assert (fast.isna() == slow.isna()).all()
    ok = slow.notna()
    assert (fast[ok] == slow[ok]).all()
    assert fast.iloc[0] == pd.Timestamp("2023-11-23 13:36:52", tz="UTC")
