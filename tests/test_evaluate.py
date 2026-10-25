"""Tests for occforecast.evaluate against hand-calculated answers."""

import numpy as np
import pandas as pd
import pytest

from occforecast.evaluate import mae, occupied_f1, rmse, rolling_origin, score


def test_mae_and_rmse_hand_calculated():
    """MAE and RMSE match hand calculations."""
    t, p = [0, 2, 4], [1, 2, 1]          # errors 1, 0, 3
    assert mae(t, p) == pytest.approx(4 / 3)
    assert rmse(t, p) == pytest.approx(np.sqrt(10 / 3))


def test_perfect_forecast():
    """A perfect forecast scores MAE = RMSE = 0 and F1 = 1."""
    t = [0.0, 1.5, 3.0]
    assert mae(t, t) == 0 and rmse(t, t) == 0 and occupied_f1(t, t) == 1


def test_nan_positions_ignored():
    """Positions with NaN in either series are excluded from scoring."""
    t, p = [1, np.nan, 3], [2, 5, np.nan]
    assert mae(t, p) == 1.0
    assert score(t, p)["n"] == 1


def test_all_nan_gives_nan():
    """Metrics are NaN when there is nothing to score."""
    assert np.isnan(mae([np.nan], [1.0]))


def test_f1_hand_calculated():
    """Occupancy F1 matches a hand-calculated confusion matrix."""
    # actual occupied:   T T F F ; forecast occupied: T F T F
    # tp=1, fp=1, fn=1 -> F1 = 2/(2+1+1) = 0.5
    assert occupied_f1([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.5)


def test_f1_threshold():
    """The occupancy threshold changes which intervals count as occupied."""
    # 0.3 people is unoccupied at threshold 0.5 but occupied at 0.2.
    assert occupied_f1([0.3], [0.3], threshold=0.2) == 1.0
    assert np.isnan(occupied_f1([0.3], [0.3], threshold=0.5))


def test_length_mismatch_raises():
    """Series of different lengths raise ValueError."""
    with pytest.raises(ValueError):
        mae([1, 2], [1])


# --- Rolling origin ----------------------------------------------------------


def _series(days=20):
    """Increasing 15-min test series starting 1 May 2024."""
    idx = pd.date_range("2024-05-01", periods=days * 96, freq="15min", tz="Australia/Melbourne")
    return pd.Series(np.arange(len(idx), dtype=float), index=idx)


def test_rolling_origin_scores_each_window():
    """Each origin is scored on exactly its own window."""
    y = _series()
    origins = [y.index[0] + pd.Timedelta(days=d) for d in (5, 10)]
    perfect = lambda s, o: s[s.index >= o]                       # knows the answer
    off_by_one = lambda s, o: s[s.index >= o] + 1
    out = rolling_origin(y, origins, "2D", {"perfect": perfect, "off": off_by_one})
    assert len(out) == 4
    assert (out.loc[out.method == "perfect", "MAE"] == 0).all()
    assert (out.loc[out.method == "off", "MAE"] == 1).all()
    assert (out["n"] == 2 * 96).all()                            # exactly the 2-day window


def test_rolling_origin_common_intervals_and_skips():
    """Methods share scored intervals; origins without history are skipped."""
    y = _series()
    origin = y.index[0] + pd.Timedelta(days=5)
    gappy = lambda s, o: s[s.index >= o].iloc[10:]               # misses 10 intervals
    full = lambda s, o: s[s.index >= o]
    out = rolling_origin(y, [origin, y.index[0]], "1D", {"gappy": gappy, "full": full})
    assert set(out["origin"]) == {origin}                        # first origin has no history
    assert (out["n"] == 96 - 10).all()                           # both scored on same intervals
