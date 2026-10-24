"""Machine-learning occupancy forecaster.

TODO (you): build features and fit the model chosen in your proposal.
Only use information available at forecast time (no future values in
features), or validation results will be optimistic.
"""

from __future__ import annotations

import pandas as pd


def make_features(series: pd.Series) -> pd.DataFrame:
    """Build model inputs for each interval.

    Suggested features: hour, minute, weekday, weekend/holiday flag,
    semester-week flag, and lagged headcounts (e.g. 1 day and 7 days earlier).

    Args:
        series: Headcount series indexed by local interval start.

    Returns:
        Feature table with the same index as ``series``.
    """
    raise NotImplementedError
