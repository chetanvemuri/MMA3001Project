"""Tests for occforecast.ventilation against hand-calculated answers."""

import numpy as np
import pandas as pd
import pytest

from occforecast.ventilation import (compare_strategies, conditioning_energy_kwh,
                                     fixed_schedule, outdoor_air_flow)

TZ = "Australia/Melbourne"


def test_outdoor_air_flow():
    hc = pd.Series([0.0, 2.5, np.nan])
    out = outdoor_air_flow(hc, 10.0)
    assert out.iloc[0] == 0 and out.iloc[1] == 25.0 and np.isnan(out.iloc[2])


def test_outdoor_air_flow_rejects_bad_input():
    with pytest.raises(ValueError):
        outdoor_air_flow(pd.Series([1.0]), 0)
    with pytest.raises(ValueError):
        outdoor_air_flow(pd.Series([-1.0]))


def test_fixed_schedule_hours_and_weekend():
    idx = pd.DatetimeIndex(["2024-05-13 06:45", "2024-05-13 07:00", "2024-05-13 18:45",
                            "2024-05-13 19:00", "2024-05-18 10:00"], tz=TZ)   # Mon x4, Sat
    out = fixed_schedule(idx, design_occupancy=20, rate_l_s_per_person=10)
    assert list(out) == [0, 200, 200, 0, 0]


def test_compare_strategies_hand_calculated():
    idx = pd.date_range("2024-05-13 09:00", periods=4, freq="15min", tz=TZ)
    required = pd.Series([0.0, 20.0, 40.0, np.nan], index=idx)        # 0, 2, 4 people, unknown
    supplied = {"exact": required.fillna(0),
                "flat30": pd.Series(30.0, index=idx)}
    out = compare_strategies(required, supplied, rate_l_s_per_person=10)
    # Required volume over 3 known intervals: (0+20+40) L/s x 900 s / 1000 = 54 m3
    assert out.loc["exact", "air_m3"] == pytest.approx(54.0)
    assert out.loc["exact", "under_vent_intervals_pct"] == 0
    assert out.loc["exact", "shortfall_person_h"] == 0
    # flat30: 30 x 900 x 3 / 1000 = 81 m3 (NaN interval ignored); short only in last (10 L/s)
    assert out.loc["flat30", "air_m3"] == pytest.approx(81.0)
    assert out.loc["flat30", "air_vs_required"] == pytest.approx(81 / 54)
    assert out.loc["flat30", "under_vent_intervals_pct"] == pytest.approx(50.0)   # 1 of 2 occupied
    # 10 L/s short = 1 person, for 0.25 h -> 0.25 person-hours
    assert out.loc["flat30", "shortfall_person_h"] == pytest.approx(0.25)


def test_conditioning_energy():
    # 1000 m3 x 1.2 kg/m3 x 1.005 kJ/kgK x 10 K = 12,060 kJ = 3.35 kWh
    assert conditioning_energy_kwh(1000, 10) == pytest.approx(3.35)
