"""Convert irregular occupancy events into a regular time series.

Each event reports the headcount at one instant. Between events the headcount
is assumed to stay constant (a *zero-order hold*), so the series is a step
function. Each output interval gets the **time-weighted mean** of that step
function, which is exact for a step function and does not over-weight busy
periods that happen to produce more events.

Long silences may be sensor outages rather than a genuinely unchanged room, so
``max_hold`` can limit how long a value is carried forward; time beyond it is
treated as unknown (NaN). Choose ``max_hold`` from the gap distribution found
in the data audit and justify it in the report.

Binning is done in UTC (Melbourne's offsets are whole hours, so 15-min bins
line up identically) and converted to local time at the end, which avoids
errors at daylight-saving changeovers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LOCAL_TZ = "Australia/Melbourne"


def to_regular(
    events: pd.DataFrame,
    freq: str = "15min",
    max_hold: str | pd.Timedelta | None = None,
    local_tz: str | None = LOCAL_TZ,
) -> pd.DataFrame:
    """Resample one space's events onto a regular time grid.

    Args:
        events: Events for a **single** space with columns ``time``
            (tz-aware) and ``headcount``. Need not be sorted.
        freq: Output interval as a pandas offset string, e.g. ``"15min"``.
        max_hold: Longest time a reading is carried forward after its event.
            ``None`` carries it forward until the next event.
        local_tz: Time zone for the output index, or ``None`` to keep UTC.

    Returns:
        DataFrame indexed by interval start, with columns:

        * ``headcount_mean``: time-weighted mean headcount over known time.
        * ``headcount_max``: highest headcount held at any point in the interval.
        * ``occupied_frac``: fraction of known time with headcount > 0.
        * ``coverage``: fraction of the interval with a known value (0-1).

        Intervals with ``coverage == 0`` have NaN in the first three columns.
        The last event's value is held to the end of its interval.

    Raises:
        ValueError: If ``events`` is empty or ``time`` is not tz-aware.
    """
    if events.empty:
        raise ValueError("events is empty")
    if events["time"].dt.tz is None:
        raise ValueError("events['time'] must be timezone-aware")

    step = pd.Timedelta(freq)
    s = (events[["time", "headcount"]]
         .assign(time=events["time"].dt.tz_convert("UTC"))
         .sort_values("time")
         .drop_duplicates("time", keep="last")   # same instant: keep latest row
         .set_index("time")["headcount"]
         .astype(float))

    start = s.index[0].floor(freq)
    end = s.index[-1].floor(freq) + step
    edges = pd.date_range(start, end, freq=freq)

    # Breakpoints: every event, every bin edge, and (optionally) the moment
    # each reading expires. Values are held between consecutive breakpoints.
    points = s.index.union(edges)
    if max_hold is not None:
        expiry = s.index + pd.Timedelta(max_hold)
        nxt = s.index[1:].append(pd.DatetimeIndex([end]))  # next event (or end)
        expiry = expiry[expiry < nxt]                      # only real silences
        points = points.union(expiry)

    held = s.reindex(points).ffill()
    if max_hold is not None:
        last_event = pd.Series(s.index, index=s.index).reindex(points).ffill()
        stale = (points - pd.DatetimeIndex(last_event)) >= pd.Timedelta(max_hold)
        held[stale] = np.nan

    seg = pd.DataFrame({"value": held.values}, index=points)
    # Seconds until the next breakpoint (unit-safe: pandas may store ns or us).
    seg["dur"] = np.append((points[1:] - points[:-1]).total_seconds(), 0.0)
    seg = seg[points < end]
    seg["bin"] = seg.index.floor(freq)
    known = seg["value"].notna()
    seg["known_dur"] = seg["dur"].where(known, 0.0)
    seg["weighted"] = (seg["value"] * seg["dur"]).where(known, 0.0)
    seg["occ_dur"] = seg["dur"].where(seg["value"] > 0, 0.0)

    g = seg.groupby("bin")
    known_t = g["known_dur"].sum()
    out = pd.DataFrame({
        "headcount_mean": g["weighted"].sum() / known_t,
        "headcount_max": g["value"].max(),
        "occupied_frac": g["occ_dur"].sum() / known_t,
        "coverage": known_t / step.total_seconds(),
    }).reindex(edges[:-1])
    out.loc[out["coverage"] == 0, ["headcount_mean", "headcount_max", "occupied_frac"]] = np.nan
    out.index.name = "interval_start"
    if local_tz is not None:
        out.index = out.index.tz_convert(local_tz)
    return out


def resample_all(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Apply :func:`to_regular` to every space.

    Args:
        df: Output of :func:`occforecast.load.load_occupancy`.
        **kwargs: Passed to :func:`to_regular` (``freq``, ``max_hold``, ...).

    Returns:
        Long DataFrame with a ``space`` column plus the columns of
        :func:`to_regular`, indexed by interval start.
    """
    parts = [to_regular(g, **kwargs).assign(space=space)
             for space, g in df.groupby("space", observed=True)]
    return pd.concat(parts)
