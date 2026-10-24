"""Error metrics and chronological validation.

TODO (you): implement metrics and a rolling-origin split. Never shuffle -
training data must always come before test data in time.
"""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error, ignoring positions where either value is NaN.

    Args:
        y_true: Observed values.
        y_pred: Forecast values, same shape as ``y_true``.

    Returns:
        Mean of ``|y_true - y_pred|`` over valid positions.
    """
    raise NotImplementedError
