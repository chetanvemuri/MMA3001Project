"""Tests for occforecast.resample against hand-calculated answers."""

import numpy as np
import pandas as pd
import pytest

from occforecast.resample import to_regular


def events(*pairs):
    """Build an events table from ("HH:MM", headcount) pairs on one UTC day."""
    return pd.DataFrame({
        "time": pd.to_datetime([f"2024-03-01 {t}" for t, _ in pairs]).tz_localize("UTC"),
        "headcount": [h for _, h in pairs],
    })


def test_time_weighted_mean():
    # 2 people for 5 min then 0 for 10 min -> mean = 2*5/15 = 2/3.
    out = to_regular(events(("00:00", 2), ("00:05", 0), ("00:30", 4)), local_tz=None)
    assert out["headcount_mean"].tolist() == pytest.approx([2 / 3, 0.0, 4.0])
    assert out["headcount_max"].tolist() == [2, 0, 4]
    assert out["occupied_frac"].tolist() == pytest.approx([1 / 3, 0.0, 1.0])
    assert (out["coverage"] == 1).all()


def test_mean_not_biased_by_event_rate():
    # Many events at 1 person, one event at 10 people: the long 10-person
    # period must dominate even though it has only one event.
    ev = events(*[(f"00:0{i}", 1) for i in range(5)], ("00:05", 10))
    out = to_regular(ev, local_tz=None)
    assert out["headcount_mean"].iloc[0] == pytest.approx((1 * 5 + 10 * 10) / 15)


def test_unsorted_input_gives_same_result():
    ev = events(("00:00", 2), ("00:05", 0), ("00:30", 4))
    a = to_regular(ev, local_tz=None)
    b = to_regular(ev.iloc[::-1], local_tz=None)
    pd.testing.assert_frame_equal(a, b)


def test_max_hold_marks_outage_unknown():
    # Reading of 3 at 00:00, then silence until 02:00. With a 30-min hold,
    # 00:00-00:30 is known, 00:30-02:00 is unknown.
    out = to_regular(events(("00:00", 3), ("02:00", 0)), max_hold="30min", local_tz=None)
    assert out["headcount_mean"].iloc[:2].tolist() == [3.0, 3.0]
    assert out["headcount_mean"].iloc[2:8].isna().all()
    assert (out["coverage"].iloc[2:8] == 0).all()
    assert out["headcount_mean"].iloc[8] == 0.0


def test_partial_coverage():
    # Hold of 5 min inside a 15-min bin -> coverage 1/3, mean over known part.
    out = to_regular(events(("00:00", 6), ("00:20", 0)), max_hold="5min", local_tz=None)
    assert out["coverage"].iloc[0] == pytest.approx(1 / 3)
    assert out["headcount_mean"].iloc[0] == 6.0


def test_local_time_index_across_dst_end():
    # Melbourne DST ended 03:00 local on 7 Apr 2024 (clocks went back to 02:00).
    ev = pd.DataFrame({
        "time": pd.to_datetime(["2024-04-06 15:00", "2024-04-06 17:00"]).tz_localize("UTC"),
        "headcount": [1, 0],
    })
    out = to_regular(ev)
    assert str(out.index.tz) == "Australia/Melbourne"
    assert len(out) == 9 and out.index.is_monotonic_increasing


def test_rejects_naive_and_empty():
    ev = events(("00:00", 1))
    with pytest.raises(ValueError):
        to_regular(ev.assign(time=ev["time"].dt.tz_localize(None)))
    with pytest.raises(ValueError):
        to_regular(ev.iloc[0:0])
