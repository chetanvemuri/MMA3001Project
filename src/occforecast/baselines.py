"""Reference forecasting methods that the main model must beat.

Both baselines need no training beyond simple averaging, so they set a fair,
easily explained bar for the ML model.
"""

from __future__ import annotations

import pandas as pd


def minute_of_week(index: pd.DatetimeIndex) -> pd.Index:
    """Minutes since Monday 00:00 (local time) for each timestamp.

    Example: Tuesday 10:15 -> 1 * 1440 + 10 * 60 + 15 = 2055.

    Args:
        index: Timestamps in local time (e.g. Australia/Melbourne).

    Returns:
        Integer index in the range 0-10079, one value per timestamp.
    """
    return pd.Index(index.day_of_week * 1440 + index.hour * 60 + index.minute)


def weekly_profile(train: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Forecast using the average for each weekday and time-of-day.

    The forecast for, say, Tuesday 10:15 is the mean of every Tuesday 10:15
    value in ``train``. Missing (NaN) training values are ignored.

    Args:
        train: Historical headcount series indexed by local interval start.
        index: Interval starts to forecast (local time).

    Returns:
        Forecast headcount indexed by ``index``. A slot never seen in
        ``train`` gives NaN.
    """
    train = train.dropna()
    profile = train.groupby(minute_of_week(train.index)).mean()
    values = profile.reindex(minute_of_week(index)).to_numpy()
    return pd.Series(values, index=index, name="weekly_profile")


def persistence(history: pd.Series, index: pd.DatetimeIndex, lag: str = "7D") -> pd.Series:
    """Forecast that each interval equals the observed value ``lag`` earlier.

    ``history`` may include the test period, because the forecast for time t
    only uses the value at t - lag, which is already known at forecast time
    as long as ``lag`` is at least the forecast horizon (e.g. 24 h).

    Note: ``lag`` is an absolute duration, so in the week after a
    daylight-saving change the copied value is one local hour out of step.

    Args:
        history: Observed headcount series indexed by interval start.
        index: Interval starts to forecast.
        lag: How far back to copy from, e.g. ``"1D"`` or ``"7D"``.

    Returns:
        Forecast headcount indexed by ``index`` (NaN where the lagged value
        is unavailable).
    """
    values = history.reindex(index - pd.Timedelta(lag)).to_numpy()
    return pd.Series(values, index=index, name=f"persistence_{lag}")
