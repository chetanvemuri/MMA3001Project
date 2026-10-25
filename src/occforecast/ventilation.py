"""Convert headcount into outdoor-air requirements and compare ventilation strategies.

AS 1668.2 (Table A1) requires at least **10 L/s of outdoor air per person** for
offices (including meeting rooms), with a floor-area minimum of 0.35 L/s/m^2
that applies even under demand control. The per-area minimum is not applied
here because the floor area of each space is unknown (the room mapping is
unverified); this is stated as a limitation.

Three kinds of strategy are compared:

* **Fixed schedule** - ventilate for the design occupancy during set hours,
  as a conventional building management system does.
* **Forecast-based** - ventilate for the day-ahead headcount forecast
  (optionally with a safety margin).
* **Ideal** - ventilate for the actual headcount (not achievable in practice;
  the lower bound on air volume).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: AS 1668.2 Table A1 outdoor-air rate for offices (L/s per person).
RATE_OFFICE = 10.0


def outdoor_air_flow(headcount: pd.Series, rate_l_s_per_person: float = RATE_OFFICE) -> pd.Series:
    """Required outdoor-air flow for each interval.

    Args:
        headcount: Forecast or measured headcount (people), >= 0.
        rate_l_s_per_person: Outdoor-air rate per person (L/s/person).

    Returns:
        Required flow (L/s), same index as ``headcount``. NaN stays NaN.

    Raises:
        ValueError: If the rate is not positive or any headcount is negative.
    """
    if rate_l_s_per_person <= 0:
        raise ValueError("rate_l_s_per_person must be positive")
    if (headcount < 0).any():
        raise ValueError("headcount must be non-negative")
    return headcount * rate_l_s_per_person


def fixed_schedule(index: pd.DatetimeIndex, design_occupancy: float,
                   rate_l_s_per_person: float = RATE_OFFICE,
                   start_hour: int = 7, end_hour: int = 19,
                   weekdays_only: bool = True) -> pd.Series:
    """Conventional fixed ventilation schedule.

    Supplies outdoor air for ``design_occupancy`` people from ``start_hour``
    to ``end_hour`` (local time), and none outside those hours.

    Args:
        index: Interval starts (local time).
        design_occupancy: Number of people the space is ventilated for.
        rate_l_s_per_person: Outdoor-air rate per person (L/s/person).
        start_hour: First hour of operation (inclusive).
        end_hour: Hour operation stops (exclusive).
        weekdays_only: If True, no ventilation on Saturday and Sunday.

    Returns:
        Supplied flow (L/s) for each interval.
    """
    on = (index.hour >= start_hour) & (index.hour < end_hour)
    if weekdays_only:
        on &= index.day_of_week < 5
    return pd.Series(np.where(on, design_occupancy * rate_l_s_per_person, 0.0), index=index)


def compare_strategies(required: pd.Series, supplied: dict[str, pd.Series],
                       rate_l_s_per_person: float = RATE_OFFICE,
                       interval: str = "15min") -> pd.DataFrame:
    """Compare ventilation strategies against the actual requirement.

    Args:
        required: Required flow from the actual headcount (L/s).
        supplied: Strategy name -> supplied flow (L/s), same index as ``required``.
        rate_l_s_per_person: Rate used to convert a flow shortfall back to people.
        interval: Length of each interval.

    Returns:
        One row per strategy with:

        * ``air_m3``: total outdoor air supplied (m^3).
        * ``air_vs_required``: supplied volume / required volume.
        * ``under_vent_intervals_pct``: % of occupied intervals (required > 0)
          where supplied < required.
        * ``shortfall_person_h``: person-hours for which the required
          10 L/s/person was not supplied (shortfall flow / rate x hours).

        Intervals where ``required`` is NaN are ignored.
    """
    dt_s = pd.Timedelta(interval).total_seconds()
    ok = required.notna()
    req = required[ok]
    occupied = req > 0
    req_m3 = req.sum() * dt_s / 1000
    rows = []
    for name, s in supplied.items():
        sup = s.reindex(req.index).fillna(0.0)
        short = (req - sup).clip(lower=0)
        rows.append({
            "strategy": name,
            "air_m3": sup.sum() * dt_s / 1000,
            "air_vs_required": sup.sum() * dt_s / 1000 / req_m3 if req_m3 > 0 else np.nan,
            "under_vent_intervals_pct": 100 * (short[occupied] > 1e-9).mean() if occupied.any() else np.nan,
            "shortfall_person_h": short.sum() / rate_l_s_per_person * dt_s / 3600,
        })
    return pd.DataFrame(rows).set_index("strategy")


def conditioning_energy_kwh(air_m3: float, delta_t_k: float,
                            density: float = 1.2, cp_kj_per_kg_k: float = 1.005) -> float:
    """Sensible energy to bring outdoor air to indoor temperature.

    ``E = rho * V * cp * dT`` (indicative only: ignores humidity, heat
    recovery and equipment efficiency).

    Args:
        air_m3: Volume of outdoor air (m^3).
        delta_t_k: Average indoor-outdoor temperature difference (K).
        density: Air density (kg/m^3).
        cp_kj_per_kg_k: Specific heat of air (kJ/kg/K).

    Returns:
        Energy in kWh.
    """
    return density * air_m3 * cp_kj_per_kg_k * delta_t_k / 3600
