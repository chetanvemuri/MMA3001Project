"""Short-term occupancy forecasting for demand-based HVAC scheduling.

MMA3001 project (Dataset 2: Monash Smart Infrastructure occupancy data).

Modules:
    load: Read, validate and audit the raw occupancy event CSV.
    resample: Convert irregular occupancy events into a regular time series.
    baselines: Reference forecasting methods (weekly profile, persistence).
    model: Day-ahead features and the linear, decision tree and MLP forecasters.
    evaluate: Error metrics and rolling-origin (chronological) validation.
    ventilation: Outdoor-air requirements (AS 1668.2) and ventilation strategy comparison.
"""

__version__ = "1.0.0"
