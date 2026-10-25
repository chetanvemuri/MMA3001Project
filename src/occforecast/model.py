"""Machine-learning occupancy forecasters (day-ahead, 15-minute resolution).

Three regression methods taught in MMA3001 (Week 5) are compared:

* **Linear regression** - a weighted sum of the features.
* **Decision tree regression** - splits the feature space into boxes and
  predicts the training mean in each box; captures non-linear calendar
  effects and reports which features matter.
* **Neural network regression** (multi-layer perceptron) - a flexible
  non-linear fit; needs scaled inputs.

Forecasting setup
-----------------
The forecast for interval *t* is assumed to be issued 24 hours earlier, so
every feature must be computable from data up to *t - 24 h*. Lagged features
therefore use lags of at least one day. :func:`make_features` enforces this;
``tests/test_model.py`` checks it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import holidays
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

#: Minimum lag allowed in any feature (the forecast horizon).
HORIZON = pd.Timedelta(days=1)

#: University shutdown window (month, day), inclusive, wrapping the new year.
#: Chosen from the data audit: the longest silences ran 23 Dec - 8 Jan.
SHUTDOWN = ((12, 22), (1, 7))

#: Feature columns, in the order the models receive them.
FEATURES = [
    "hour", "day_of_week", "is_weekend", "is_holiday",
    "tod_sin", "tod_cos", "tod_sin2", "tod_cos2",
    "lag_1d", "lag_7d", "profile_4w", "recent_mean_7d",
]


def _lag(y: pd.Series, delta: pd.Timedelta) -> np.ndarray:
    """Value of ``y`` exactly ``delta`` before each timestamp.

    Args:
        y: Series indexed by timestamp.
        delta: How far back to look.

    Returns:
        Array aligned with ``y.index``; NaN where no value exists ``delta`` earlier.
    """
    return y.reindex(y.index - delta).to_numpy()


def holiday_flags(index: pd.DatetimeIndex, subdiv: str = "VIC") -> np.ndarray:
    """Flag Victorian public holidays and the university shutdown window.

    Args:
        index: Local timestamps.
        subdiv: Australian state for public holidays.

    Returns:
        Integer array, 1 on a holiday or shutdown day, else 0.
    """
    years = range(index.year.min() - 1, index.year.max() + 2)
    public = holidays.Australia(subdiv=subdiv, years=years)
    dates = index.normalize().tz_localize(None).date
    is_public = np.array([d in public for d in dates])
    (m0, d0), (m1, d1) = SHUTDOWN
    md = index.month * 100 + index.day
    in_shutdown = (md >= m0 * 100 + d0) | (md <= m1 * 100 + d1)
    return (is_public | in_shutdown).astype(int)


def make_features(y: pd.Series) -> pd.DataFrame:
    """Build day-ahead model inputs for every interval of one space.

    Features (all known 24 h before the interval):

    * ``hour``: local time of day in hours (e.g. 10.25 for 10:15).
    * ``day_of_week``: 0 = Monday ... 6 = Sunday; ``is_weekend``: 1 on Sat/Sun.
    * ``is_holiday``: public holiday or university shutdown (:func:`holiday_flags`).
    * ``tod_sin``, ``tod_cos`` (and 2nd harmonics): time of day on a circle,
      so 23:45 and 00:00 are close - needed by the linear and MLP models.
    * ``lag_1d``, ``lag_7d``: observed headcount 1 day and 7 days earlier.
    * ``profile_4w``: mean of the same slot 7, 14, 21 and 28 days earlier
      (an adaptive weekly profile).
    * ``recent_mean_7d``: mean headcount over the 7 days ending 24 h earlier;
      tracks term vs. break activity levels without a calendar.

    Args:
        y: Regular (e.g. 15-min) headcount series for one space, indexed by
            local interval start. NaN marks unknown intervals.

    Returns:
        DataFrame with columns :data:`FEATURES`, indexed like ``y``.
        Lag features are NaN where history is missing.

    Raises:
        ValueError: If the index is not sorted.
    """
    if not y.index.is_monotonic_increasing:
        raise ValueError("y must be sorted by time")
    idx = y.index
    tod = (idx.hour + idx.minute / 60) / 24 * 2 * np.pi
    day = pd.Timedelta(days=1)
    weekly = [_lag(y, k * 7 * day) for k in (1, 2, 3, 4)]
    rolling = y.rolling("7D", min_periods=1).mean()

    return pd.DataFrame({
        "hour": idx.hour + idx.minute / 60,
        "day_of_week": idx.day_of_week,
        "is_weekend": (idx.day_of_week >= 5).astype(int),
        "is_holiday": holiday_flags(idx),
        "tod_sin": np.sin(tod), "tod_cos": np.cos(tod),
        "tod_sin2": np.sin(2 * tod), "tod_cos2": np.cos(2 * tod),
        "lag_1d": _lag(y, day),
        "lag_7d": weekly[0],
        "profile_4w": pd.DataFrame(weekly).mean(axis=0, skipna=True).to_numpy(),
        "recent_mean_7d": rolling.reindex(idx - HORIZON).to_numpy(),
    }, index=idx)[FEATURES]


def make_model(name: str, **params) -> Pipeline:
    """Create an untrained model pipeline.

    Every pipeline first fills missing lag values with the training median
    (so all models score the same intervals). Linear and MLP models also
    standardise features to zero mean and unit variance.

    Args:
        name: ``"linear"``, ``"tree"`` or ``"mlp"``.
        **params: Overrides for the regressor, e.g. ``max_depth=8``.

    Returns:
        scikit-learn Pipeline with a final step named ``"model"``.

    Raises:
        ValueError: If ``name`` is unknown.
    """
    impute = ("impute", SimpleImputer(strategy="median"))
    if name == "linear":
        return Pipeline([impute, ("scale", StandardScaler()),
                         ("model", LinearRegression(**params))])
    if name == "tree":
        cfg = {"max_depth": 10, "min_samples_leaf": 20, "random_state": 0, **params}
        return Pipeline([impute, ("model", DecisionTreeRegressor(**cfg))])
    if name == "mlp":
        cfg = {"hidden_layer_sizes": (32, 16), "early_stopping": True,
               "max_iter": 200, "random_state": 0, **params}
        return Pipeline([impute, ("scale", StandardScaler()),
                         ("model", MLPRegressor(**cfg))])
    raise ValueError(f"Unknown model {name!r}; use 'linear', 'tree' or 'mlp'")


def fit_predict(y: pd.Series, split: pd.Timestamp, model: Pipeline,
                features: pd.DataFrame | None = None) -> pd.Series:
    """Train on intervals before ``split`` and forecast intervals from ``split`` on.

    Only training intervals with a known target are used for fitting.
    Forecasts are clipped at zero because headcount cannot be negative.

    Args:
        y: Headcount series for one space (see :func:`make_features`).
        split: First timestamp of the test period.
        model: Untrained pipeline from :func:`make_model` (fitted in place).
        features: Precomputed ``make_features(y)`` to save time; built if None.

    Returns:
        Forecast headcount for every interval from ``split`` onwards.
    """
    X = make_features(y) if features is None else features
    train = (y.index < split) & y.notna().to_numpy()
    test = y.index >= split
    model.fit(X[train], y[train])
    pred = np.clip(model.predict(X[test]), 0, None)
    return pd.Series(pred, index=y.index[test], name="forecast")


def feature_importance(model: Pipeline) -> pd.Series:
    """Feature importances of a fitted decision tree, largest first.

    Args:
        model: Fitted pipeline from ``make_model("tree")``.

    Returns:
        Importance per feature (sums to 1).
    """
    return (pd.Series(model.named_steps["model"].feature_importances_, index=FEATURES)
            .sort_values(ascending=False))
