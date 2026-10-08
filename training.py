"""Training, on Ray: a day-ahead PV model.

The target is the hourly mean of the PV power series, bucketed on UTC hours --
the same quantity the evaluation scores. The features for an hour H are what
would have been known 24 hours before H:

  * the archived Open-Meteo forecast for the hours around H, from the lead time
    FORECAST_LEAD_DAYS (default 2: issued 48 h before the hour it describes, so
    it was certainly published by the time a 24 h-ahead prediction is made),
  * sun geometry for the site,
  * the PV hourly mean at the same hour 2..8 days before H.

The regressor is a gradient-boosted tree ensemble with an absolute-error loss,
because the evaluation metric is MAE. Training hours are weighted by recency
(half-life SAMPLE_HALFLIFE_DAYS), so the level of the forecast follows what the
system has produced lately while the long history still shapes the response
to the weather.
"""

import datetime
import os
import typing

import numpy as np
import pandas as pd
import ray
from mlflow.pyfunc import PythonModel

from operator_lib.util.helpers import provide_historic_data, TrainMlflowLogger


# How much history one training pass reads.
TRAINING_WINDOW = datetime.timedelta(days=int(os.environ.get("TRAINING_WINDOW_DAYS", "365")))
# The tail of the training history held out to report a validation error.
VALIDATION_DAYS = int(os.environ.get("VALIDATION_DAYS", "30"))
# Site coordinates; the weather imports are for 51.7 / 10.
SITE_LAT = float(os.environ.get("SITE_LAT", "51.7"))
SITE_LON = float(os.environ.get("SITE_LON", "10.0"))
# Which forecast lead time to train and predict on.
FORECAST_LEAD_DAYS = int(os.environ.get("FORECAST_LEAD_DAYS", "2"))
# Recency weighting: a training hour this many days old counts half as much as
# the newest one. 0 or less switches weighting off.
SAMPLE_HALFLIFE_DAYS = float(os.environ.get("SAMPLE_HALFLIFE_DAYS", "90"))

HORIZON = datetime.timedelta(hours=24)
PV_FIELD = "power"
WEATHER_VARS = [
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "cloud_cover",
    "temperature_2m",
]
WEATHER_FIELDS = ["forecasted_for", "issued_at", "lead_days"] + WEATHER_VARS
# Forecast hours used for target bucket [H, H+1h), relative to H. Open-Meteo's
# radiation at hour X is the mean over the hour before X, so +1 is the one that
# covers the bucket; the others make the model tolerant of a shifted time base.
FC_OFFSETS = (-1, 0, 1, 2)
LAG_DAYS = tuple(range(2, 9))
PRIMARY_FEATURE = "shortwave_radiation_+1"

# Hold-out diagnostics: hours the forecast called clear vs. cloudy, by kt.
KT_CLEAR = 0.7
KT_CLOUDY = 0.4

FEATURES = (
    [f"{v}_{o:+d}" for v in WEATHER_VARS for o in FC_OFFSETS]
    + ["sin_elev", "clear_sky", "kt", "hour", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    + ["lag2", "lag_mean", "lag_max", "lag_n"]
)

REGRESSOR_PARAMS = dict(
    loss="absolute_error",
    learning_rate=float(os.environ.get("GBM_LEARNING_RATE", "0.05")),
    max_iter=int(os.environ.get("GBM_MAX_ITER", "600")),
    max_leaf_nodes=int(os.environ.get("GBM_MAX_LEAF_NODES", "31")),
    min_samples_leaf=int(os.environ.get("GBM_MIN_SAMPLES_LEAF", "20")),
    l2_regularization=float(os.environ.get("GBM_L2", "0.0")),
    random_state=0,
)


# --------------------------------------------------------------------------
# Shared helpers (used by training here and by inference in op.py)
# --------------------------------------------------------------------------

def to_utc(value) -> typing.Optional[pd.Timestamp]:
    """Any timestamp representation as an aware UTC pandas Timestamp, or None.

    A string without an offset is taken as UTC. A number is taken as epoch
    milliseconds when it is large enough to be one, else epoch seconds.
    """
    if value is None:
        return None
    try:
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            if value != value:
                return None
            unit = "ms" if abs(value) > 1e11 else "s"
            ts = pd.Timestamp(float(value), unit=unit)
        else:
            ts = pd.Timestamp(value)
    except (ValueError, TypeError, OverflowError):
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def sin_elevation(times: pd.DatetimeIndex, lat: float, lon: float) -> np.ndarray:
    """Sine of the solar elevation at UTC instants (NOAA approximation)."""
    doy = np.asarray(times.dayofyear, dtype=float)
    hour = np.asarray(times.hour + times.minute / 60.0 + times.second / 3600.0, dtype=float)
    g = 2.0 * np.pi / 365.0 * (doy - 1.0 + (hour - 12.0) / 24.0)
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                    - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g)
            - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
            - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    tst = hour * 60.0 + eqt + 4.0 * lon
    ha = np.radians(tst / 4.0 - 180.0)
    la = np.radians(lat)
    return np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha)


def assemble_features(hours: pd.DatetimeIndex, fc: typing.Dict[str, np.ndarray],
                      lags: np.ndarray, lat: float, lon: float) -> pd.DataFrame:
    """The feature frame for target buckets starting at `hours` (UTC).

    fc maps "<var>_<offset>" to one value per hour; lags is (n, len(LAG_DAYS)).
    """
    n = len(hours)
    q1 = sin_elevation(hours + pd.Timedelta(minutes=15), lat, lon)
    q3 = sin_elevation(hours + pd.Timedelta(minutes=45), lat, lon)
    se = (np.clip(q1, 0, None) + np.clip(q3, 0, None)) / 2.0
    clear_sky = 1000.0 * np.power(se, 1.15)
    sw = np.asarray(fc.get(PRIMARY_FEATURE, np.full(n, np.nan)), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        kt = np.where(clear_sky > 20.0, sw / clear_sky, np.nan)
    kt = np.clip(kt, 0.0, 1.5)

    hr = np.asarray(hours.hour, dtype=float)
    doy = np.asarray(hours.dayofyear, dtype=float)
    lags = np.asarray(lags, dtype=float).reshape(n, len(LAG_DAYS))
    valid = ~np.isnan(lags)
    lag_n = valid.sum(axis=1).astype(float)
    lag_sum = np.where(valid, lags, 0.0).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        lag_mean = np.where(lag_n > 0, lag_sum / np.maximum(lag_n, 1), np.nan)
    lag_max = np.where(lag_n > 0, np.where(valid, lags, -np.inf).max(axis=1), np.nan)

    cols = {}
    for v in WEATHER_VARS:
        for o in FC_OFFSETS:
            key = f"{v}_{o:+d}"
            cols[key] = np.asarray(fc.get(key, np.full(n, np.nan)), dtype=float)
    cols.update({
        "sin_elev": se,
        "clear_sky": clear_sky,
        "kt": kt,
        "hour": hr,
        "hour_sin": np.sin(2 * np.pi * hr / 24.0),
        "hour_cos": np.cos(2 * np.pi * hr / 24.0),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "lag2": lags[:, 0],
        "lag_mean": lag_mean,
        "lag_max": lag_max,
        "lag_n": lag_n,
    })
    return pd.DataFrame(cols, columns=FEATURES)


class PvDayAheadModel(PythonModel):
    """The model MLflow registers and op.py later loads.

    op.py calls predict_hour() on the unwrapped model; predict() serves a
    caller that already has a feature frame.
    """

    def __init__(self, regressor, lat: float, lon: float, lead_days: int, cap: float,
                 pv_recent: typing.Dict[pd.Timestamp, float],
                 fc_recent: typing.List[typing.Tuple[int, pd.Timestamp, pd.Timestamp, typing.Dict[str, float]]]) -> None:
        self.regressor = regressor
        self.lat = lat
        self.lon = lon
        self.lead_days = lead_days
        self.cap = cap
        self.pv_recent = pv_recent
        self.fc_recent = fc_recent
        self.features = list(FEATURES)

    def predict_hour(self, hour: pd.Timestamp,
                     fc_values: typing.Dict[str, float],
                     lag_values: typing.List[float]) -> float:
        """One hour's forecast.

        The regressor was trained only on hours that had a forecast, so it is
        not asked about an hour without one: that hour falls back to the mean
        of the same hour 2..8 days before, or 0 when there is none either.
        """
        if fc_values.get(PRIMARY_FEATURE) is None:
            known = [v for v in lag_values if v is not None]
            fallback = float(np.mean(known)) if known else 0.0
            return float(min(max(fallback, 0.0), self.cap))
        hours = pd.DatetimeIndex([hour])
        fc = {k: np.array([v if v is not None else np.nan], dtype=float) for k, v in fc_values.items()}
        lags = np.array([[v if v is not None else np.nan for v in lag_values]], dtype=float)
        X = assemble_features(hours, fc, lags, self.lat, self.lon)
        y = float(self.regressor.predict(X[self.features])[0])
        return float(min(max(y, 0.0), self.cap))

    def predict(self, context, model_input=None, params=None):
        payload = model_input if model_input is not None else context
        X = pd.DataFrame(payload)[self.features]
        return np.clip(self.regressor.predict(X), 0.0, self.cap)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def _resolve(refs) -> list:
    out = []
    for ref in refs or []:
        out.append(ray.get(ref) if isinstance(ref, ray.ObjectRef) else ref)
    return out


def _columns(dataset) -> typing.List[str]:
    try:
        return list(dataset.schema().names)
    except Exception:
        try:
            return list(dataset.columns())
        except Exception:
            return []


def _pv_hourly(dataset) -> pd.Series:
    """Hourly mean of the PV field over UTC buckets, aggregated batch by batch."""
    acc = None
    for batch in dataset.iter_batches(batch_size=200_000, batch_format="pandas"):
        if PV_FIELD not in batch.columns or "time" not in batch.columns:
            continue
        t = pd.to_datetime(batch["time"], utc=True, errors="coerce")
        v = pd.to_numeric(batch[PV_FIELD], errors="coerce")
        df = pd.DataFrame({"h": t.dt.floor("h"), "v": v}).dropna()
        if df.empty:
            continue
        g = df.groupby("h")["v"].agg(["sum", "count"])
        acc = g if acc is None else acc.add(g, fill_value=0)
    if acc is None or acc.empty:
        return pd.Series(dtype=float)
    return (acc["sum"] / acc["count"]).sort_index()


def _weather_frame(dataset) -> pd.DataFrame:
    df = dataset.to_pandas()
    if df.empty:
        return df
    df["msg_time"] = pd.to_datetime(df["time"], utc=True, errors="coerce") if "time" in df.columns else pd.NaT
    df["ff"] = pd.to_datetime(df["forecasted_for"].map(to_utc), utc=True, errors="coerce")
    df["ia"] = pd.to_datetime(df["issued_at"].map(to_utc), utc=True, errors="coerce")
    df["lead"] = pd.to_numeric(df["lead_days"], errors="coerce")
    for v in WEATHER_VARS:
        df[v] = pd.to_numeric(df[v], errors="coerce") if v in df.columns else np.nan
    df = df.dropna(subset=["ff", "lead"])
    df["ff"] = df["ff"].dt.floor("h")
    df["lead"] = df["lead"].astype(int)
    return df


def _forecast_table(weather: pd.DataFrame, lead: int) -> pd.DataFrame:
    sel = weather[weather["lead"] == lead]
    sel = sel.sort_values("ia").drop_duplicates("ff", keep="last")
    return sel.set_index("ff")[WEATHER_VARS].sort_index()


def _fc_arrays(table: pd.DataFrame, hours: pd.DatetimeIndex) -> typing.Dict[str, np.ndarray]:
    out = {}
    for o in FC_OFFSETS:
        shifted = table.reindex(hours + pd.Timedelta(hours=o))
        for v in WEATHER_VARS:
            out[f"{v}_{o:+d}"] = shifted[v].to_numpy(dtype=float)
    return out


def _lag_matrix(pv: pd.Series, hours: pd.DatetimeIndex) -> np.ndarray:
    cols = [pv.reindex(hours - pd.Timedelta(days=d)).to_numpy(dtype=float) for d in LAG_DAYS]
    return np.column_stack(cols) if cols else np.empty((len(hours), 0))


def _recency_weights(hours: pd.DatetimeIndex, end: pd.Timestamp) -> typing.Optional[np.ndarray]:
    """Weight per training hour, halving every SAMPLE_HALFLIFE_DAYS before `end`."""
    if SAMPLE_HALFLIFE_DAYS <= 0:
        return None
    age_days = np.asarray((end - hours).total_seconds(), dtype=float) / 86400.0
    return np.power(0.5, np.clip(age_days, 0.0, None) / SAMPLE_HALFLIFE_DAYS)


def _effective_hours(weights: typing.Optional[np.ndarray], n: int) -> float:
    """Kish effective sample size: how many equally weighted hours the weights amount to."""
    if weights is None:
        return float(n)
    return float(weights.sum() ** 2 / np.square(weights).sum())


@ray.remote
def _fit(X: pd.DataFrame, y: np.ndarray, params: dict, weights: typing.Optional[np.ndarray] = None):
    from sklearn.ensemble import HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor(**params)
    model.fit(X, y, sample_weight=weights)
    return model


def _mae(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - actual))) if len(actual) else float("nan")


def _subset_metrics(name: str, mask: np.ndarray, pred: np.ndarray, actual: np.ndarray) -> typing.Dict[str, float]:
    """MAE, bias and hour count over one subset of the hold-out; nothing when it is empty."""
    n = int(mask.sum())
    if n == 0:
        return {f"{name}_hours": 0.0}
    return {
        f"{name}": _mae(pred[mask], actual[mask]),
        f"{name}_bias": float(np.mean(pred[mask] - actual[mask])),
        f"{name}_mean_actual": float(np.mean(actual[mask])),
        f"{name}_hours": float(n),
    }


def train_model(logger: TrainMlflowLogger) -> typing.Optional[PythonModel]:
    """Read the history, fit, and hand back a model for MLflow to register."""
    with logger.trace("read history"):
        datasets = _resolve(provide_historic_data(TRAINING_WINDOW))
    if not datasets:
        return None

    pv_ds, weather_ds = None, None
    for ds in datasets:
        cols = set(_columns(ds))
        if PV_FIELD in cols:
            pv_ds = ds
        elif "shortwave_radiation" in cols:
            weather_ds = ds
    if pv_ds is None or weather_ds is None:
        logger.log_params({"error": f"inputs incomplete: pv={pv_ds is not None} weather={weather_ds is not None}"})
        return None

    with logger.trace("aggregate"):
        pv = _pv_hourly(pv_ds)
        weather = _weather_frame(weather_ds)
    if pv.empty or weather.empty:
        logger.log_params({"error": f"empty history: pv_hours={len(pv)} weather_rows={len(weather)}"})
        return None

    # Diagnostics of the forecast archive's time base: both should be ~0 hours.
    diag = {}
    if weather["msg_time"].notna().any() and weather["ia"].notna().any():
        diag["diag_msgtime_minus_issued_h"] = float(
            ((weather["msg_time"] - weather["ia"]).dt.total_seconds() / 3600.0).median())
    if weather["ia"].notna().any():
        diag["diag_ff_minus_issued_minus_lead_h"] = float(
            ((weather["ff"] - weather["ia"]).dt.total_seconds() / 3600.0 - 24.0 * weather["lead"]).median())

    leads = weather["lead"].value_counts()
    lead = FORECAST_LEAD_DAYS if FORECAST_LEAD_DAYS in leads.index else int(leads.idxmax())
    table = _forecast_table(weather, lead)

    hours = pd.DatetimeIndex(pv.index)
    fc = _fc_arrays(table, hours)
    lags = _lag_matrix(pv, hours)
    X = assemble_features(hours, fc, lags, SITE_LAT, SITE_LON)
    y = pv.to_numpy(dtype=float)
    keep = ~np.isnan(X[PRIMARY_FEATURE].to_numpy()) & ~np.isnan(y)
    X, y, hours = X[keep].reset_index(drop=True), y[keep], hours[keep]
    if len(y) < 24 * 14:
        logger.log_params({"error": f"too few aligned hours: {len(y)}"})
        return None

    end = hours.max() + pd.Timedelta(hours=1)
    val_start = end - pd.Timedelta(days=VALIDATION_DAYS)
    is_val = np.asarray(hours >= val_start)
    cap = float(np.nanmax(y)) * 1.05 if len(y) else 0.0

    metrics = dict(diag)
    # Logged as metrics too, because a run summary may carry metrics only.
    metrics.update({
        "train_hours": float(len(y)),
        "training_window_days": float(TRAINING_WINDOW.days),
        "history_span_days": float((end - hours.min()).total_seconds() / 86400.0),
        "forecast_lead_days": float(lead),
        "sample_halflife_days": float(SAMPLE_HALFLIFE_DAYS),
    })
    with logger.trace("validate"):
        if is_val.sum() > 0 and (~is_val).sum() > 24 * 14:
            # The hold-out model is weighted relative to the end of its own
            # training data, exactly as the final model is relative to `end`.
            w_val = _recency_weights(hours[~is_val], val_start)
            val_model = ray.get(_fit.remote(X[~is_val], y[~is_val], REGRESSOR_PARAMS, w_val))
            pred = np.clip(val_model.predict(X[is_val]), 0.0, cap)
            actual = y[is_val]
            day = X.loc[is_val, "sin_elev"].to_numpy() > 0
            kt = X.loc[is_val, "kt"].to_numpy()
            lag_base = X.loc[is_val, "lag_mean"].fillna(0.0).to_numpy()
            metrics.update({
                "val_mae": _mae(pred, actual),
                "val_bias": float(np.mean(pred - actual)),
                "val_mae_daylight": _mae(pred[day], actual[day]),
                "val_mae_lag_baseline": _mae(lag_base, actual),
                "val_mae_zero_baseline": _mae(np.zeros_like(actual), actual),
                "val_hours": float(len(actual)),
                "val_mean_actual": float(np.mean(actual)),
            })
            # Where the error lives: hours the forecast called clear (a model
            # error there is systematic) vs. cloudy (mostly forecast error).
            with np.errstate(invalid="ignore"):
                clear = np.nan_to_num(kt, nan=-1.0) >= KT_CLEAR
                cloudy = (np.nan_to_num(kt, nan=99.0) < KT_CLOUDY) & day
            metrics.update(_subset_metrics("val_mae_clear", clear, pred, actual))
            metrics.update(_subset_metrics("val_mae_cloudy", cloudy, pred, actual))

    with logger.trace("fit"):
        w_all = _recency_weights(hours, end)
        regressor = ray.get(_fit.remote(X, y, REGRESSOR_PARAMS, w_all))
    metrics["train_mae"] = _mae(np.clip(regressor.predict(X), 0.0, cap), y)
    metrics["train_effective_hours"] = _effective_hours(w_all, len(y))

    # What the operator needs at the start of a test window or a deployment:
    # PV hours for the lag features and forecasts already issued for the next days.
    pv_recent = {ts: float(v) for ts, v in pv[pv.index >= end - pd.Timedelta(days=10)].items()}
    recent = weather[weather["ff"] >= end - pd.Timedelta(hours=6)]
    fc_recent = [
        (int(r.lead), r.ff, r.ia if not pd.isna(r.ia) else r.ff - pd.Timedelta(days=int(r.lead)),
         {v: (None if pd.isna(getattr(r, v)) else float(getattr(r, v))) for v in WEATHER_VARS})
        for r in recent.itertuples(index=False)
    ]

    logger.log_params({
        "training_window_days": TRAINING_WINDOW.days,
        "validation_days": VALIDATION_DAYS,
        "forecast_lead_days": lead,
        "sample_halflife_days": SAMPLE_HALFLIFE_DAYS,
        "site_lat": SITE_LAT,
        "site_lon": SITE_LON,
        "train_hours": int(len(y)),
        "history_start": str(hours.min()),
        "history_end": str(end),
        **{f"gbm_{k}": v for k, v in REGRESSOR_PARAMS.items()},
    })
    logger.log_metrics(metrics)

    return PvDayAheadModel(regressor, SITE_LAT, SITE_LON, lead, cap, pv_recent, fc_recent)
