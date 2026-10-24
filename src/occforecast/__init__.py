"""Short-term occupancy forecasting for demand-based HVAC scheduling.

MMA3001 project (Dataset 2: Monash Smart Infrastructure occupancy data).

Modules:
    load: Read, validate and audit the raw occupancy event CSV.
    resample: Convert irregular occupancy events into a regular time series.
    baselines: Reference forecasting methods (weekly profile, persistence).
    model: Machine-learning forecasting model.
    evaluate: Error metrics and chronological validation.
    ventilation: Convert headcount forecasts into outdoor-air requirements.
"""

__version__ = "0.1.0"
