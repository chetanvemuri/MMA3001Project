"""Read, validate and audit the raw occupancy event CSV.

The raw file holds one row per sensor event (roughly every 2-60 s while a space
is active), with these columns used here:

==========================  ========  =============================================
Column                      Type      Meaning
==========================  ========  =============================================
``floorspaceid``            str       Space identifier (UUID; room mapping unverified)
``collecteddate``           datetime  Event time with UTC offset, e.g. ``+11``
``occupancystatus``         str       ``CurrentlyOccupied``, ``RecentlyOccupied``
                                      or ``NotOccupied``
``headcount``               int       People counted in the space (>= 0)
==========================  ========  =============================================

Rows are not stored in time order, so :func:`load_occupancy` always sorts them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

#: Columns that must be present in the raw CSV.
REQUIRED_COLUMNS = ["floorspaceid", "collecteddate", "occupancystatus", "headcount"]

#: Allowed values of ``occupancystatus``.
VALID_STATUSES = {"CurrentlyOccupied", "RecentlyOccupied", "NotOccupied"}


def load_occupancy(path: str, chunksize: int = 1_000_000) -> pd.DataFrame:
    """Load the occupancy CSV into a clean, time-sorted DataFrame.

    The file is read in chunks so that only the four required columns are held
    in memory. Rows with an unparseable timestamp, a missing space ID, a
    negative or missing headcount, or an unknown status are dropped.

    Args:
        path: Path to the raw occupancy CSV.
        chunksize: Number of rows read per chunk. Larger is faster but uses
            more memory.

    Returns:
        DataFrame with columns ``time`` (tz-aware, UTC), ``space`` (category),
        ``status`` (category) and ``headcount`` (int16), sorted by space then
        time. ``df.attrs["n_raw"]`` and ``df.attrs["n_dropped"]`` record how
        many rows were read and removed.

    Raises:
        ValueError: If any column in :data:`REQUIRED_COLUMNS` is missing.
    """
    header = pd.read_csv(path, nrows=0).columns
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    chunks, n_raw = [], 0
    for chunk in pd.read_csv(path, usecols=REQUIRED_COLUMNS, chunksize=chunksize):
        n_raw += len(chunk)
        chunks.append(_clean_chunk(chunk))

    df = pd.concat(chunks, ignore_index=True)
    df["space"] = df["space"].astype("category")
    df["status"] = df["status"].astype("category")
    df = df.sort_values(["space", "time"], kind="stable", ignore_index=True)
    df.attrs["n_raw"] = n_raw
    df.attrs["n_dropped"] = n_raw - len(df)
    return df


def load_cached(path: str, cache: str | None = None, **kwargs) -> pd.DataFrame:
    """Load the occupancy CSV, reusing a saved copy of the cleaned data if possible.

    The first call runs :func:`load_occupancy` and saves the result as a
    pickle (a fast binary format). Later calls read the pickle instead of
    re-parsing 9 million CSV rows, as long as the pickle is newer than the CSV.
    Only use pickles you created yourself: loading a pickle can run code.

    Args:
        path: Path to the raw occupancy CSV.
        cache: Where to store the cleaned copy. Defaults to the CSV path with
            the suffix ``.clean.pkl`` (ignored by git).
        **kwargs: Passed to :func:`load_occupancy`.

    Returns:
        Same as :func:`load_occupancy`. ``df.attrs["from_cache"]`` is True if
        the saved copy was used.
    """
    src = Path(path)
    cache_path = Path(cache) if cache else src.with_suffix(".clean.pkl")
    if cache_path.exists() and cache_path.stat().st_mtime >= src.stat().st_mtime:
        df = pd.read_pickle(cache_path)
        df.attrs["from_cache"] = True
        return df
    df = load_occupancy(path, **kwargs)
    df.to_pickle(cache_path)
    df.attrs["from_cache"] = False
    return df


def parse_timestamps(text: pd.Series) -> pd.Series:
    """Alternative timestamp parser, kept for the benchmark in notebook 05.

    Parses the fixed-format date-time part and the UTC offset (``+10``/``+11``)
    separately and combines them. It gives the same instants as
    ``pd.to_datetime(text, utc=True, errors="coerce")`` (checked in
    ``tests/test_load.py``).

    It was written as a proposed optimisation, but on the reference machine
    (pandas on Windows, real data) it was **about 3x slower** than
    ``pd.to_datetime``, so :func:`load_occupancy` does not use it. The
    measurement is reported in ``examples/05_performance.ipynb``.

    Args:
        text: Timestamps as ``YYYY-MM-DD HH:MM:SS`` followed by an offset
            ``+HH``, ``+HHMM`` or ``+HH:MM``.

    Returns:
        Timezone-aware UTC timestamps; NaT where the text is not valid.
    """
    s = text.astype("string")
    local = pd.to_datetime(s.str[:19], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    off = (pd.to_numeric(s.str[19:].str.replace(":", "", regex=False), errors="coerce")
           .to_numpy(dtype=float, na_value=np.nan))
    # "+11" -> 11 h; "+1030" -> 10.5 h
    hours = np.where(np.abs(off) >= 100,
                     np.trunc(off / 100) + np.sign(off) * (np.abs(off) % 100) / 60, off)
    return (local - pd.to_timedelta(hours, unit="h")).dt.tz_localize("UTC")


def _clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Rename, type-convert and filter one chunk of raw rows.

    Args:
        chunk: Raw rows containing :data:`REQUIRED_COLUMNS`.

    Returns:
        Valid rows only, with columns ``time``, ``space``, ``status``,
        ``headcount``.
    """
    out = pd.DataFrame({
        # utc=True handles the mixed +10/+11 offsets (daylight saving).
        # (A hand-written parser was tried and was slower; see parse_timestamps.)
        "time": pd.to_datetime(chunk["collecteddate"], utc=True, errors="coerce"),
        "space": chunk["floorspaceid"],
        "status": chunk["occupancystatus"],
        "headcount": pd.to_numeric(chunk["headcount"], errors="coerce"),
    })
    valid = (
        out["time"].notna()
        & out["space"].notna()
        & out["headcount"].ge(0)          # also False for NaN
        & out["status"].isin(VALID_STATUSES)
    )
    out = out[valid].copy()
    out["headcount"] = out["headcount"].astype("int16")
    return out


def audit(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise data quality for each space.

    Args:
        df: Output of :func:`load_occupancy`.

    Returns:
        One row per space with: number of events, first and last event time,
        maximum headcount, longest gap between events (hours), number of
        duplicated timestamps, and number of rows where status and headcount
        disagree (``CurrentlyOccupied`` with headcount 0, or another status
        with headcount > 0).
    """
    def _per_space(g: pd.DataFrame) -> pd.Series:
        occupied = g["status"].eq("CurrentlyOccupied")
        inconsistent = (occupied & g["headcount"].eq(0)) | (~occupied & g["headcount"].gt(0))
        return pd.Series({
            "events": len(g),
            "start": g["time"].min(),
            "end": g["time"].max(),
            "max_headcount": int(g["headcount"].max()),
            "longest_gap_h": g["time"].diff().max().total_seconds() / 3600,
            "duplicate_times": int(g["time"].duplicated().sum()),
            "status_mismatch": int(inconsistent.sum()),
        })

    return df.groupby("space", observed=True).apply(_per_space, include_groups=False)
