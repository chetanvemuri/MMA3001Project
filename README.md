# Short-Term Occupancy Forecasting for Demand-Based HVAC Scheduling

MMA3001 Project, Dataset 2 (Monash Smart Infrastructure occupancy data).

## Purpose

HVAC systems often run on fixed schedules regardless of how spaces are used.
This project forecasts the headcount of each space **one day ahead at
15-minute resolution**, so ventilation could be scheduled to match demand. The
forecasts are converted into outdoor-air requirements (AS 1668.2: 10 L/s per
person) and compared with a conventional fixed schedule.

| | Description |
|---|---|
| **Input** | Event-based occupancy sensor CSV: 9,091,732 rows, 5 spaces, Nov 2023 - Apr 2026. Columns used: timestamp (with UTC offset), space ID, occupancy status, headcount (people) |
| **Output** | Day-ahead forecast of mean headcount (people) per space per 15-min interval, and the outdoor-air flow (L/s) it requires |
| **Baselines** | Weekday x time-of-day average profile; persistence (same time 1 or 7 days earlier) |
| **Models** | Linear regression, decision tree regression and neural network (MLP) regression (scikit-learn), using calendar, holiday and lagged-history features that are all known 24 h in advance |
| **Chosen model** | MLP, selected by rolling-origin validation (linear regression is a near-equal, ~200x cheaper alternative) |

## Key results

| Question | Result | Notebook |
|---|---|---|
| Is the data usable? | All 9.09M rows valid; headcount and status agree in every row; long silences match holidays and weekends | 01 |
| Do the ML models beat the baselines? | Yes, on average: MLP MAE 0.531 vs 0.614 for the weekly profile (-13.5%) over 42 rolling 8-week test windows; the MLP beats the best baseline in 71% of windows | 03, 04 |
| Are the conclusions robust? | Insensitive to the outage-handling rule (`max_hold`) and to the occupancy threshold; decision tree overfits beyond depth ~6-8; linear regression needs only ~3 months of history | 04 |
| What did optimisation achieve? | Cached loading 390x faster; 36x less memory (109 MB vs ~3.9 GB); one proposed optimisation measured 2.9x slower and rejected | 05 |
| What does it mean for ventilation? | The fixed schedule supplies ~7x the required air yet under-ventilates 41.5% of occupied intervals. Forecast-based ventilation with a +100% safety margin uses **71% less outdoor air** and has **22% less under-ventilation** | 06 |

## Repository structure

```
├── src/occforecast/       Package source code (documented in docs/)
│   ├── load.py            Read, validate, audit and cache the raw CSV
│   ├── resample.py        Irregular events -> regular 15-min series (+ loop reference version)
│   ├── baselines.py       Weekly-profile and persistence forecasts
│   ├── model.py           Day-ahead features; linear, tree and MLP forecasters
│   ├── evaluate.py        MAE, RMSE, occupancy F1; rolling-origin validation
│   └── ventilation.py     AS 1668.2 outdoor-air requirement; strategy comparison
├── tests/                 pytest tests (51, 100% line coverage)
├── examples/              Analysis notebooks (see below)
├── docs/                  HTML API documentation (pdoc)
├── reports/               Test report and coverage report (HTML)
├── data/                  Local data only - not committed (see data/README.md)
├── requirements.txt       Dependencies
└── LICENSE                MIT
```

### Notebooks (run in order)

| Notebook | Contents |
|---|---|
| `01_data_audit.ipynb` | Validation, data-quality audit, gap analysis, choice of `max_hold = 48 h`, resampling |
| `02_baselines.ipynb` | Chronological train/test split; baseline forecasts and scores |
| `03_models.ipynb` | Linear, tree and MLP models vs baselines; feature importance; error breakdown; timing |
| `04_validation_sensitivity.ipynb` | Rolling-origin validation and model selection; sensitivity to tree depth, `max_hold`, history length, threshold |
| `05_performance.ipynb` | Timing, profiling, loading/memory/caching optimisations, loop vs vectorised, FLOPs and roofline |
| `06_ventilation.ipynb` | Fixed vs forecast-based ventilation: outdoor air, under-ventilation, safety margin, indicative energy |

## Setup

Requires Python 3.11+.

```bash
git clone https://github.com/chetanvemuri/MMA3001Project.git
cd MMA3001Project
python -m pip install -r requirements.txt
```

Then download the data as described in [data/README.md](data/README.md).

> **Windows (Microsoft Store Python):** if `python` is not recognised, use
> `python3.11` or the full path to `python3.11.exe` in the commands below.

## Usage

```python
import sys; sys.path.insert(0, "src")          # notebooks in examples/ use "../src"
import pandas as pd
from occforecast.load import load_cached, audit
from occforecast.resample import resample_all
from occforecast.model import make_model, fit_predict
from occforecast.evaluate import score

events = load_cached("data/raw/5occupancySensor_MayToDec2024_9MRows.csv")  # ~13 s first time, then <1 s
print(audit(events))                                                        # data-quality report
series = resample_all(events, freq="15min", max_hold="48h").sort_index()

y = series[series["space"] == series["space"].iloc[0]]["headcount_mean"]   # one space
split = pd.Timestamp("2025-07-01", tz="Australia/Melbourne")
forecast = fit_predict(y, split, make_model("mlp"))                        # day-ahead forecast
print(score(y[forecast.index], forecast))                                   # MAE, RMSE, F1
```

## Testing

```bash
python -m pytest                                    # run all tests (51)
python -m pytest --html=reports/test_report.html --self-contained-html \
                 --cov=occforecast --cov-report=html:reports/coverage
```

Open `reports/test_report.html` (test results) and `reports/coverage/index.html`
(line coverage). Tests check results against hand calculations, check that
features use no future data, and compare the vectorised resampler with an
independently written loop version.

## Documentation

```bash
python -m pdoc src/occforecast -o docs --docformat google
```

Open `docs/index.html` in a browser. All functions have Google-style docstrings.

## Known data limitations

- **Space-to-room mapping is unverified.** The provided Sensor ID and Locations
  sheet marks occupancy IDs as incorrect; MSI confirmed (via the unit forum) the
  IDs are not synced at cloud level and cannot be corrected. Spaces are
  therefore treated as anonymous (A-E).
- **Environmental data cannot be linked to occupancy spaces.** After removing
  duplicated rows, the environmental extract contains two usable sensors, one
  of which is not in the location sheet.
- **File names do not match contents.** The occupancy file named "MayToDec2024"
  covers Nov 2023 - Apr 2026.
- **Rows are not in time order** and must be sorted (done in `load_occupancy`).
- **Do not open and save the CSV in Excel**: Excel truncates it to 1,048,576 rows.
- **Floor areas are unknown**, so AS 1668.2's per-area ventilation minimum
  (0.35 L/s/m^2) is not applied.

## Licence

Code: MIT (see [LICENSE](LICENSE)). The data is not covered by this licence and
is not redistributed.

## AI use disclosure

_Keep this section up to date - the report's AI reflection is built from it.
Items marked TODO can only be answered by me._

| Tool | Used for | Extent | How output was checked |
|---|---|---|---|
| Google Gemini 2.5 | _TODO_ | _TODO_ | _TODO_ |
| Claude (Anthropic) | Interpreting the brief; auditing the provided datasets; drafting the problem statement; writing the package code (`load`, `resample`, `baselines`, `model`, `evaluate`, `ventilation`), tests and notebooks; drafting notebook findings from my results; Git/VS Code help | _TODO: e.g. "most code first drafted by Claude; I ran, checked and corrected it"_ | 51 unit tests against hand calculations (100% line coverage); independent loop implementation to verify the resampler; equality checks after every optimisation; ran every notebook on the real data and checked written findings against the outputs; _TODO: my own checks_ |

**Decisions that stayed mine:** _TODO - confirm/edit, e.g. choice of Dataset 2;
switching from CO2-based estimation to occupancy forecasting after the audit;
accepting `max_hold = 48 h` from the gap evidence; the model-selection rule
fixed before seeing rolling-origin results; rejecting the slower timestamp parser._

**AI errors found:**
- Claude's first resampling code computed interval durations 1,000x too small
  (it assumed nanosecond timestamps; pandas 3 uses microseconds). Caught by
  the unit tests before commit.
- Claude initially proposed estimating occupancy from CO2. The data audit
  showed the environmental and occupancy data cannot be linked, so this was
  dropped.
- Claude's plotting code sliced dates on an unsorted multi-space index, causing
  a KeyError; fixed by sorting the index first.
- A Claude-written baseline test used dates crossing the daylight-saving change,
  where one local hour repeats, so it failed; the test dates were moved.
- Claude proposed a hand-written timestamp parser that was faster in its own
  tests (synthetic data, newer pandas) but **2.9x slower on my machine with the
  real data**. I measured it and rejected it (`examples/05_performance.ipynb`).
- _TODO: add any others._

**Effect on learning:** _TODO._
