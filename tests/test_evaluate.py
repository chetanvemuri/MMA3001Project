"""Tests for occforecast.evaluate against hand-calculated answers."""

import numpy as np
import pytest

from occforecast.evaluate import mae, occupied_f1, rmse, score


def test_mae_and_rmse_hand_calculated():
    t, p = [0, 2, 4], [1, 2, 1]          # errors 1, 0, 3
    assert mae(t, p) == pytest.approx(4 / 3)
    assert rmse(t, p) == pytest.approx(np.sqrt(10 / 3))


def test_perfect_forecast():
    t = [0.0, 1.5, 3.0]
    assert mae(t, t) == 0 and rmse(t, t) == 0 and occupied_f1(t, t) == 1


def test_nan_positions_ignored():
    t, p = [1, np.nan, 3], [2, 5, np.nan]
    assert mae(t, p) == 1.0
    assert score(t, p)["n"] == 1


def test_all_nan_gives_nan():
    assert np.isnan(mae([np.nan], [1.0]))


def test_f1_hand_calculated():
    # actual occupied:   T T F F ; forecast occupied: T F T F
    # tp=1, fp=1, fn=1 -> F1 = 2/(2+1+1) = 0.5
    assert occupied_f1([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.5)


def test_f1_threshold():
    # 0.3 people is unoccupied at threshold 0.5 but occupied at 0.2.
    assert occupied_f1([0.3], [0.3], threshold=0.2) == 1.0
    assert np.isnan(occupied_f1([0.3], [0.3], threshold=0.5))


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        mae([1, 2], [1])
