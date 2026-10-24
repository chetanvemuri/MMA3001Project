"""Reference forecasting methods that the main model must beat.

TODO (you): implement and test both functions. Keep them simple - their job is
to set the bar for the ML model, and to be easy to explain in the Q&A.
"""

from __future__ import annotations

import pandas as pd


def weekly_profile(train: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Forecast using the average for each weekday and time-of-day.

    Example: the forecast for a Tuesday 10:15 interval is the mean of all
    Tuesday 10:15 intervals in ``train``.

    Args:
        train: Historical headcount series indexed by local interval start.
        index: Interval starts to forecast.

    Returns:
        Forecast headcount indexed by ``index``.
    """
    raise NotImplementedError


def persistence(train: pd.Series, index: pd.DatetimeIndex, lag: str = "7D") -> pd.Series:
    """Forecast that each interval equals the value ``lag`` earlier.

    Args:
        train: Historical headcount series indexed by local interval start.
        index: Interval starts to forecast.
        lag: How far back to copy from, e.g. ``"1D"`` or ``"7D"``.

    Returns:
        Forecast headcount indexed by ``index``.
    """
    raise NotImplementedError
