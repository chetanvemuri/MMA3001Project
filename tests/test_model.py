"""Tests for occforecast.model: leakage, feature values and model behaviour."""

import numpy as np
import pandas as pd
import pytest

from occforecast.model import (FEATURES, feature_importance, fit_predict,
                               holiday_flags, make_features, make_model)

TZ = "Australia/Melbourne"


def weekly_series(start="2024-05-06", weeks=8, noise=0.0, seed=0):
    """15-min series with a repeating weekly pattern (busy weekday daytimes)."""
    idx = pd.date_range(start, periods=weeks * 7 * 96, freq="15min", tz=TZ)
    hour = idx.hour + idx.minute / 60
    base = np.where((idx.day_of_week < 5) & (hour >= 9) & (hour < 17), 5.0, 0.0)
    rng = np.random.default_rng(seed)
    return pd.Series(base + noise * rng.standard_normal(len(idx)), index=idx)


# --- Features ----------------------------------------------------------------

def test_feature_columns_and_index():
    y = weekly_series(weeks=2)
    X = make_features(y)
    assert list(X.columns) == FEATURES
    assert X.index.equals(y.index)


def test_no_leakage_from_last_24_hours():
    # Changing values in the 24 h before time t must not change t's features.
    y = weekly_series(weeks=6)
    t = y.index[-1]
    y2 = y.copy()
    y2[(y2.index > t - pd.Timedelta("1D")) & (y2.index <= t)] = 999.0
    pd.testing.assert_series_equal(make_features(y).loc[t], make_features(y2).loc[t])


def test_lag_values():
    y = pd.Series(np.arange(96 * 30, dtype=float),
                  index=pd.date_range("2024-05-01", periods=96 * 30, freq="15min", tz=TZ))
    X = make_features(y)
    t = y.index[-1]
    assert X.loc[t, "lag_1d"] == y[t - pd.Timedelta("1D")]
    assert X.loc[t, "lag_7d"] == y[t - pd.Timedelta("7D")]
    expected_profile = np.mean([y[t - pd.Timedelta(days=7 * k)] for k in (1, 2, 3, 4)])
    assert X.loc[t, "profile_4w"] == pytest.approx(expected_profile)
    assert np.isnan(X.iloc[0]["lag_1d"])          # no history at the start


def test_calendar_features():
    idx = pd.DatetimeIndex(["2024-05-11 10:15"], tz=TZ)   # Saturday
    X = make_features(pd.Series([1.0], index=idx))
    assert X["hour"].iloc[0] == 10.25
    assert X["day_of_week"].iloc[0] == 5 and X["is_weekend"].iloc[0] == 1


def test_holiday_flags():
    idx = pd.DatetimeIndex(["2024-04-25 12:00",    # Anzac Day
                            "2024-12-27 12:00",    # shutdown window
                            "2025-01-07 12:00",    # last shutdown day
                            "2025-01-08 12:00",    # back at work
                            "2024-05-14 12:00"],   # ordinary Tuesday
                           tz=TZ)
    assert list(holiday_flags(idx)) == [1, 1, 1, 0, 0]


def test_unsorted_raises():
    y = weekly_series(weeks=1)
    with pytest.raises(ValueError):
        make_features(y.iloc[::-1])


# --- Models ------------------------------------------------------------------

@pytest.mark.parametrize("name", ["linear", "tree", "mlp"])
def test_models_forecast_whole_test_period_non_negative(name):
    y = weekly_series(weeks=8, noise=0.3)
    split = y.index[0] + pd.Timedelta(weeks=6)
    pred = fit_predict(y, split, make_model(name))
    assert pred.index.equals(y.index[y.index >= split])
    assert (pred >= 0).all() and pred.notna().all()


def test_tree_learns_exact_weekly_pattern():
    # A noise-free repeating pattern should be forecast almost perfectly.
    y = weekly_series(weeks=8)
    split = y.index[0] + pd.Timedelta(weeks=6)
    pred = fit_predict(y, split, make_model("tree"))
    assert np.abs(pred - y[pred.index]).mean() < 0.05


def test_missing_training_targets_are_skipped():
    y = weekly_series(weeks=8)
    y.iloc[:96 * 7] = np.nan                       # first week unknown
    split = y.index[0] + pd.Timedelta(weeks=6)
    pred = fit_predict(y, split, make_model("tree"))
    assert pred.notna().all()


def test_feature_importance_sums_to_one():
    y = weekly_series(weeks=8, noise=0.3)
    model = make_model("tree")
    fit_predict(y, y.index[0] + pd.Timedelta(weeks=6), model)
    imp = feature_importance(model)
    assert imp.sum() == pytest.approx(1.0)
    assert set(imp.index) == set(FEATURES)


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        make_model("forest")
