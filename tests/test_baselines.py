"""Tests for occforecast.baselines against hand-calculated answers."""

import numpy as np
import pandas as pd
import pytest

from occforecast.baselines import minute_of_week, persistence, weekly_profile

TZ = "Australia/Melbourne"


def grid(start, periods):
    """15-minute local-time index (dates chosen to avoid DST changes)."""
    return pd.date_range(start, periods=periods, freq="15min", tz=TZ)


def test_minute_of_week():
    """minute_of_week gives minutes since Monday 00:00 (hand-calculated)."""
    idx = pd.DatetimeIndex(["2024-03-18 00:00", "2024-03-19 10:15", "2024-03-24 23:45"], tz=TZ)
    # Mon 00:00, Tue 10:15, Sun 23:45
    assert list(minute_of_week(idx)) == [0, 1440 + 615, 6 * 1440 + 23 * 60 + 45]


def test_weekly_profile_averages_same_slot():
    """The forecast for a slot is the mean of past values in the same weekday/time slot."""
    # Two Mondays at 10:00 with values 2 and 4 -> forecast 3 for the next Monday 10:00.
    train = pd.Series([2.0, 4.0], index=pd.DatetimeIndex(["2024-03-18 10:00", "2024-03-25 10:00"], tz=TZ))
    target = pd.DatetimeIndex(["2024-04-01 10:00"], tz=TZ)
    assert weekly_profile(train, target).iloc[0] == pytest.approx(3.0)


def test_weekly_profile_ignores_nan_and_unseen_slot_is_nan():
    """Missing training values are skipped; a never-seen slot gives NaN."""
    train = pd.Series([2.0, np.nan], index=pd.DatetimeIndex(["2024-03-18 10:00", "2024-03-25 10:00"], tz=TZ))
    target = pd.DatetimeIndex(["2024-04-01 10:00", "2024-04-01 11:00"], tz=TZ)
    out = weekly_profile(train, target)
    assert out.iloc[0] == 2.0          # NaN not averaged in
    assert np.isnan(out.iloc[1])       # 11:00 Monday never observed


def test_weekly_profile_recovers_exact_weekly_pattern():
    """A pattern that repeats exactly every week is forecast perfectly."""
    # A pattern that repeats exactly every week must be forecast perfectly.
    idx = grid("2024-05-06", 4 * 96 * 7)                     # 4 weeks (Mondays, no DST)
    pattern = pd.Series(np.tile(np.arange(96 * 7) % 13, 4).astype(float), index=idx)
    future = grid("2024-06-03", 96 * 7)
    out = weekly_profile(pattern, future)
    np.testing.assert_allclose(out.to_numpy(), np.arange(96 * 7) % 13)


def test_persistence_copies_value_one_week_earlier():
    """Persistence returns the observed value exactly one lag earlier."""
    hist = pd.Series(np.arange(96 * 14, dtype=float), index=grid("2024-03-04", 96 * 14))
    target = grid("2024-03-11 10:00", 4)
    out = persistence(hist, target, lag="7D")
    expected = hist.loc[target - pd.Timedelta("7D")].to_numpy()
    np.testing.assert_array_equal(out.to_numpy(), expected)


def test_persistence_missing_history_is_nan():
    """Persistence gives NaN when no value exists one lag earlier."""
    hist = pd.Series([1.0], index=grid("2024-03-04", 1))
    out = persistence(hist, grid("2024-03-04", 1), lag="1D")   # nothing 1 day earlier
    assert np.isnan(out.iloc[0])
