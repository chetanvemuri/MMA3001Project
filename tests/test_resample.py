"""Tests for occforecast.resample against hand-calculated answers."""

import numpy as np
import pandas as pd
import pytest

from occforecast.resample import resample_all, to_regular, to_regular_naive


def events(*pairs):
    """Build an events table from ("HH:MM", headcount) pairs on one UTC day."""
    return pd.DataFrame({
        "time": pd.to_datetime([f"2024-03-01 {t}" for t, _ in pairs]).tz_localize("UTC"),
        "headcount": [h for _, h in pairs],
    })


def test_time_weighted_mean():
    """Interval means are time-weighted (hand-calculated)."""
    # 2 people for 5 min then 0 for 10 min -> mean = 2*5/15 = 2/3.
    out = to_regular(events(("00:00", 2), ("00:05", 0), ("00:30", 4)), local_tz=None)
    assert out["headcount_mean"].tolist() == pytest.approx([2 / 3, 0.0, 4.0])
    assert out["headcount_max"].tolist() == [2, 0, 4]
    assert out["occupied_frac"].tolist() == pytest.approx([1 / 3, 0.0, 1.0])
    assert (out["coverage"] == 1).all()


def test_mean_not_biased_by_event_rate():
    """Many short events do not outweigh one long steady period."""
    # Many events at 1 person, one event at 10 people: the long 10-person
    # period must dominate even though it has only one event.
    ev = events(*[(f"00:0{i}", 1) for i in range(5)], ("00:05", 10))
    out = to_regular(ev, local_tz=None)
    assert out["headcount_mean"].iloc[0] == pytest.approx((1 * 5 + 10 * 10) / 15)


def test_unsorted_input_gives_same_result():
    """Input order does not change the result."""
    ev = events(("00:00", 2), ("00:05", 0), ("00:30", 4))
    a = to_regular(ev, local_tz=None)
    b = to_regular(ev.iloc[::-1], local_tz=None)
    pd.testing.assert_frame_equal(a, b)


def test_max_hold_marks_outage_unknown():
    """Silences longer than max_hold become unknown (NaN)."""
    # Reading of 3 at 00:00, then silence until 02:00. With a 30-min hold,
    # 00:00-00:30 is known, 00:30-02:00 is unknown.
    out = to_regular(events(("00:00", 3), ("02:00", 0)), max_hold="30min", local_tz=None)
    assert out["headcount_mean"].iloc[:2].tolist() == [3.0, 3.0]
    assert out["headcount_mean"].iloc[2:8].isna().all()
    assert (out["coverage"].iloc[2:8] == 0).all()
    assert out["headcount_mean"].iloc[8] == 0.0


def test_partial_coverage():
    """Coverage and mean are correct when only part of an interval is known."""
    # Hold of 5 min inside a 15-min bin -> coverage 1/3, mean over known part.
    out = to_regular(events(("00:00", 6), ("00:20", 0)), max_hold="5min", local_tz=None)
    assert out["coverage"].iloc[0] == pytest.approx(1 / 3)
    assert out["headcount_mean"].iloc[0] == 6.0


def test_local_time_index_across_dst_end():
    """Output is in Melbourne time and stays ordered across the DST change."""
    # Melbourne DST ended 03:00 local on 7 Apr 2024 (clocks went back to 02:00).
    ev = pd.DataFrame({
        "time": pd.to_datetime(["2024-04-06 15:00", "2024-04-06 17:00"]).tz_localize("UTC"),
        "headcount": [1, 0],
    })
    out = to_regular(ev)
    assert str(out.index.tz) == "Australia/Melbourne"
    assert len(out) == 9 and out.index.is_monotonic_increasing


def test_rejects_naive_and_empty():
    """Timezone-naive or empty input raises ValueError."""
    ev = events(("00:00", 1))
    with pytest.raises(ValueError):
        to_regular(ev.assign(time=ev["time"].dt.tz_localize(None)))
    with pytest.raises(ValueError):
        to_regular(ev.iloc[0:0])


# --- Independent reference implementation ------------------------------------


@pytest.mark.parametrize("max_hold", [None, "20min", "2h"])
def test_vectorised_matches_naive_reference(max_hold):
    """The vectorised resampler matches the independent loop version."""
    # Random events (irregular gaps from seconds to hours, duplicates included):
    # two independently written implementations must agree.
    rng = np.random.default_rng(42)
    gaps = rng.choice([2, 4, 30, 600, 3600, 4 * 3600], size=400, p=[.3, .2, .2, .15, .1, .05])
    times = pd.Timestamp("2024-03-01", tz="UTC") + pd.to_timedelta(np.cumsum(gaps), unit="s")
    ev = pd.DataFrame({"time": times, "headcount": rng.integers(0, 6, len(times))})
    ev = pd.concat([ev, ev.iloc[[5, 50]].assign(headcount=9)])       # duplicate timestamps
    fast = to_regular(ev, max_hold=max_hold, local_tz=None)
    slow = to_regular_naive(ev, max_hold=max_hold, local_tz=None)
    pd.testing.assert_frame_equal(fast, slow, check_freq=False, rtol=1e-9)


def test_naive_reference_local_time_and_bad_input():
    """The loop version also returns Melbourne time and rejects bad input."""
    ev = events(("00:00", 2), ("00:30", 0))
    assert str(to_regular_naive(ev).index.tz) == "Australia/Melbourne"
    with pytest.raises(ValueError):
        to_regular_naive(ev.iloc[0:0])
    with pytest.raises(ValueError):
        to_regular_naive(ev.assign(time=ev["time"].dt.tz_localize(None)))


def test_resample_all_labels_each_space():
    """resample_all resamples every space and tags rows with their space ID."""
    ev = pd.concat([events(("00:00", 2), ("00:30", 0)).assign(space="A"),
                    events(("00:00", 5)).assign(space="B")])
    out = resample_all(ev, local_tz=None)
    assert set(out["space"]) == {"A", "B"}
    assert out.loc[out["space"] == "B", "headcount_mean"].tolist() == [5.0]
    assert out.loc[out["space"] == "A", "headcount_mean"].tolist() == [2.0, 2.0, 0.0]
