"""Convert headcount forecasts into outdoor-air requirements.

TODO (you): confirm the per-person rate and room type in AS 1668.2 before use,
and cite the clause in your report.
"""

from __future__ import annotations

import pandas as pd


def outdoor_air_flow(headcount: pd.Series, rate_l_s_per_person: float) -> pd.Series:
    """Required outdoor-air flow for each interval.

    Args:
        headcount: Forecast or measured headcount (people).
        rate_l_s_per_person: Outdoor-air rate per person (L/s/person), from
            AS 1668.2 for the relevant room type.

    Returns:
        Required flow (L/s), same index as ``headcount``.

    Raises:
        ValueError: If ``rate_l_s_per_person`` is not positive.
    """
    raise NotImplementedError
