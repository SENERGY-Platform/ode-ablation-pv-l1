# ode-ablation-pv-l1

A 24 h-ahead forecast of photovoltaic generation, as hourly mean power in watts,
for the APSystems DS3-S inverter ("Wechselrichter") on the SENERGY platform.

## Summary of the approach

### The problem

Forecast PV generation 24 hours ahead as hourly mean power in W. A run is judged
by `evaluation.yaml`: MAE over UTC hours of September 2026, threshold 30 W. The
model trains only on history before 2026-09-01, then Operator Lib replays
September through `infer()` and scores what it produced.

### Data: real series only, found through the ontology

- **Target.** Searching the ontology for Get-Power under the Generation aspect
  found two real PV devices. You chose the inverter's `root.powerTotal` (W).
  The profiler showed it reports every 60 s in daylight and not at all at
  night. Night hours therefore have no actual and are never scored.
- **Weather.** A 24 h-ahead forecast needs weather information that was
  available 24 h earlier. Observed weather, which is only known afterwards,
  would leak into the forecast. The input is the existing import "Open-Meteo
  forecast archive (previous runs, 51.7/10)":
  - It archives each forecast as it was issued, stamped at its issue time.
  - Its export holds history from 2024-01.
  - The model uses the **lead-2** forecast, issued 48 h before the hour it
    describes, so it was certainly published by the time of a 24 h-ahead
    prediction. The import notes that lead 1 can be slightly optimistic about
    when it was published.
- No simulation was needed. Real data existed for both inputs.

### How the scoring works, and what the operator does with it

Reading Operator Lib 1.8.1 (`util/op_ml.py`) showed how a run is scored:

- The actual for an hour is the mean of the target's own messages in that UTC
  hour.
- A prediction is matched to its hour by the `result_time` the operator returns.
- Errors within an hour are averaged first, then averaged across hours.
- The target series has to be one of the operator's inputs.

So for every inverter message at time T, `op.py` returns:

- `result_time = T + 24 h`;
- `{"prediction": W}` for the UTC hour containing T + 24 h.

Weather messages are only stored. A forecast is used only once its `issued_at`
has passed.

### The model (`training.py`)

- **Target:** the hourly mean of `power` over UTC hours, the same quantity the
  evaluation scores.
- **Features for hour H:**
  - forecast irradiance (shortwave, direct, diffuse), cloud cover and
    temperature at H−1 … H+2;
  - sun geometry for 51.7 N / 10 E: elevation, a clear-sky proxy, and `kt`
    (forecast irradiance divided by clear-sky irradiance);
  - hour of day and day of year;
  - the measured hourly mean at the same hour 2–8 days before H.
- **Regressor:** scikit-learn `HistGradientBoostingRegressor` with an
  absolute-error loss, since the metric is MAE.
- **History:** 970 days, with training hours weighted by recency at a 90-day
  half-life. The window and half-life are set by environment variables.
- **Hold-out:** the last 30 days before the split are held out and logged
  against two baselines: "repeat the same hour of recent days" and "always 0".
  The error is also split by forecast-clear and forecast-cloudy hours.
- **Check on the forecast timestamps:** two diagnostics log how the archive's
  timestamps relate to `issued_at`. Both are 0 h, so the replay used only
  forecasts that had already been issued.
- **Missing forecast:** if an hour has no forecast, the operator falls back to
  the same-hour mean of days 2–8. It does not ask the model about a case it
  was never trained on.
- **Level correction in `op.py`:** each forecast is scaled by
  sum(actual) / sum(forecast) over the last 7 days of completed daylight hours.
  The factor is clipped to [0.8, 1.25] and only applies once at least 24 hours
  have been compared. It uses only past data.
- **Start of the test window:** the model carries the last PV hours and
  already-issued forecasts it saw in training, so it can forecast from the
  first hour of September.

### Results (evaluation runs, test window September 2026, 391 scored hours)

| Run | Commit | Change | Test MAE (W) |
|---|---|---|---|
| 1 | 977b8c3 | First model, 365-day history | 55.3 |
| 2 | 977b8c3 | 970-day history (`TRAINING_WINDOW_DAYS=970`) | 53.3 |
| 3 | 9441059 | Clear/cloudy error breakdown (diagnostics only) | 53.3 |
| 4 | 3105005 | Recency weighting, half-life 90 days | 53.4 |
| 5 | 5c098cd | Online 7-day level correction | **52.8** |

- **Not met:** the 30 W threshold is not met. The best run is 52.8 W.
- **Against the baseline:** on the hold-out, the model's error is about 37 %
  lower than repeating recent days (about 50 W against 79 W). Mean daylight
  output is about 210 W.
- **The bias found was small:** the breakdown showed a systematic
  under-forecast on clear hours of about 22–27 W. Recency weighting and the
  level correction reduced it, but gained at most 0.7 W on the test.
- **What limits it:** the remaining error is mostly the error of a single
  weather forecast issued two days ahead. In my judgment, 30 W (about 14 % of
  mean daylight output) is not reachable with the data this platform holds
  today.

### What was not done, and what would come next

- **A second, independent weather forecast** is the remaining lever: a second
  Open-Meteo previous-runs import using the ECMWF IFS 0.25° model, its export,
  and its irradiance and cloud cover as extra features.
  - ODE could not deploy it: the import type's `apikey` setting counts as a
    credential, so it has to be created in the platform's import dialog.
  - Its export would need `time_path = value.issued_at` and the timestamp
    format your existing archive export uses, so the backfilled history lands
    at its issue times.
- **The yr.no forecast import** that already exists has too little stored
  history to train on: its export starts 2026-08-22.
- **The other PV device** ("Leiste PV", a smart plug) was not modelled.
- **No simulation** was created.

### Reproducing the best run

Launch commit `5c098cd` with `uv run python train.py` and these settings:

- **Environment:** `TRAINING_WINDOW_DAYS=970`, `SAMPLE_HALFLIFE_DAYS=90`.
- **Input 1:** device `urn:infai:ses:device:89169393-068d-4fb1-a032-2f238ad7ac9d`,
  service `urn:infai:ses:service:9ddd93b2-0988-407f-931d-a2ebcc8f6d94`,
  source `value.root.powerTotal` mapped to `power`.
- **Input 2:** import `urn:infai:ses:import:6fd7bbee-c1dd-d165-562d-d43c7d300055`,
  read from its export `1c1872e2-9ef1-46e8-8afd-36ca0dc38680`.
  Map `value.<name>` to `<name>` for each of: `forecasted_for`, `issued_at`,
  `lead_days`, `shortwave_radiation`, `direct_radiation`, `diffuse_radiation`,
  `cloud_cover`, `temperature_2m`.

---

An analytics operator for the SENERGY platform, scaffolded by the Operator
Development Environment. Every file here is yours to change, including this one.

## Layout

| File | What it is |
|---|---|
| "main.py" | Entry point of the deployed operator. Hands the process to Operator Lib. |
| "train.py" | Entry point of an experiment. Trains through Operator Lib, then exits. |
| "op.py" | The operator: "infer", "train", "need_retraining", and its config. |
| "training.py" | The Ray training pass and the model MLflow registers. |
| "pyproject.toml" | Dependencies, with Operator Lib pinned at "v1.8.1". |
| "uv.lock" | The resolved dependencies. Written by the scaffold; refresh it yourself. See below. |
| "Dockerfile" | The image. Built by CI; buildable by hand. |
| ".github/workflows/build.yml" | Builds and pushes "ghcr.io/senergy-platform/ode-ablation-pv-l1". Change the registry here. |
| "operator.yaml" | What the analytics stack registers: inputs, outputs, config. |
| "evaluation.yaml" | Your criteria for whether a run is good, plus what Operator Lib needs to score a test window itself. ODE never writes this. |

## The lock file

The scaffold ran "uv lock" for you and "uv.lock" is in this working copy, uncommitted
like everything else here. Commit it with the rest.

Refresh it whenever you change a dependency in "pyproject.toml", and commit the two
together:

    uv lock

An experiment runs "uv run python train.py" on the cluster, and uv builds the
environment from "pyproject.toml" and this file — on the Ray head for the driver and
on each worker node for the tasks, out of its own cache.

Without a lock file uv resolves at run time, which works and is worse in one
specific way: the run records a commit SHA as the code that produced it, and two
runs of the same commit can then resolve different dependency versions. The lock
file is what makes the recorded SHA describe the whole run rather than only its
source. That is why it is not left to be remembered — and if the scaffold reported
that it could not write one, the command above is the repair.

## Building by hand

    docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t ghcr.io/senergy-platform/ode-ablation-pv-l1:dev .

## The Operator Lib pin

"pyproject.toml" pins Operator Lib at "v1.8.1", the newest at the time
this repository was scaffolded. The library tracks latest and promises no
stability, so moving the pin is a deliberate edit — change it, run "uv lock", and
commit the two together.
