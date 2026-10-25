"""Error metrics for headcount forecasts.

All metrics ignore positions where either the observed or the forecast value
is NaN (e.g. intervals inside a sensor outage), and report how many positions
were actually scored. Training and test data must be split by date, never
shuffled, so the test period is always later than the training period.
"""

from __future__ import annotations

import numpy as np


def _valid(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Return the two arrays restricted to positions where both are finite.

    Args:
        y_true: Observed values.
        y_pred: Forecast values, same length as ``y_true``.

    Returns:
        ``(y_true, y_pred)`` as float arrays without NaN/inf positions.

    Raises:
        ValueError: If the inputs differ in length.
    """
    t = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if t.shape != p.shape:
        raise ValueError(f"Length mismatch: {t.shape} vs {p.shape}")
    ok = np.isfinite(t) & np.isfinite(p)
    return t[ok], p[ok]


def mae(y_true, y_pred) -> float:
    """Mean absolute error (people).

    Args:
        y_true: Observed headcount.
        y_pred: Forecast headcount.

    Returns:
        Mean of ``|y_true - y_pred|`` over valid positions, or NaN if none.
    """
    t, p = _valid(y_true, y_pred)
    return float(np.mean(np.abs(t - p))) if t.size else float("nan")


def rmse(y_true, y_pred) -> float:
    """Root-mean-square error (people). Penalises large misses more than MAE.

    Args:
        y_true: Observed headcount.
        y_pred: Forecast headcount.

    Returns:
        ``sqrt(mean((y_true - y_pred)**2))`` over valid positions, or NaN.
    """
    t, p = _valid(y_true, y_pred)
    return float(np.sqrt(np.mean((t - p) ** 2))) if t.size else float("nan")


def occupied_f1(y_true, y_pred, threshold: float = 0.5) -> float:
    """F1 score for the occupied/unoccupied decision.

    An interval counts as *occupied* when its mean headcount is at least
    ``threshold``. F1 balances precision (of the intervals forecast as
    occupied, how many were) and recall (of the truly occupied intervals,
    how many were caught). Low recall means under-ventilating occupied rooms.

    Args:
        y_true: Observed mean headcount per interval.
        y_pred: Forecast mean headcount per interval.
        threshold: Headcount at or above which an interval is occupied.

    Returns:
        F1 in [0, 1], or NaN if there are no occupied intervals in either
        series.
    """
    t, p = _valid(y_true, y_pred)
    actual, forecast = t >= threshold, p >= threshold
    tp = np.sum(actual & forecast)
    fp = np.sum(~actual & forecast)
    fn = np.sum(actual & ~forecast)
    if tp + fp + fn == 0:
        return float("nan")
    return float(2 * tp / (2 * tp + fp + fn))


def score(y_true, y_pred, threshold: float = 0.5) -> dict:
    """All metrics for one forecast.

    Args:
        y_true: Observed headcount.
        y_pred: Forecast headcount.
        threshold: Occupancy threshold passed to :func:`occupied_f1`.

    Returns:
        Dict with ``MAE``, ``RMSE``, ``F1`` and ``n`` (positions scored).
    """
    t, _ = _valid(y_true, y_pred)
    return {"MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred),
            "F1": occupied_f1(y_true, y_pred, threshold), "n": int(t.size)}
