"""The operator: a 24 h-ahead forecast of hourly mean PV power, in watts.

Two inputs reach infer():

  * "pv": the PV power series (destination "power"). Every message at time T
    produces a forecast for the UTC hour containing T + 24 h, stamped with
    result time T + 24 h, so the output carries the hour it is about.
  * "weather": the Open-Meteo previous-runs forecast import. Each message is
    one forecast hour at one lead time, published at its issue time; infer()
    only stores it. A forecast is used only once its issue time has passed.
"""

import datetime
import typing

import pandas as pd
from mlflow.pyfunc import PyFuncModel, PythonModel

from operator_lib.util import Config, MLOperator, Selector
from operator_lib.util.helpers import TrainMlflowLogger

from training import (
    FC_OFFSETS,
    HORIZON,
    LAG_DAYS,
    PV_FIELD,
    WEATHER_FIELDS,
    WEATHER_VARS,
    to_utc,
    train_model,
)


class CustomConfig(Config):
    """Deployment configuration, typed."""

    # Retrain at most this often, in seconds.
    retrain_after_s = 86400


class Operator(MLOperator):
    configType = CustomConfig

    selectors = [
        Selector({"name": "pv", "args": [PV_FIELD]}),
        Selector({"name": "weather", "args": list(WEATHER_FIELDS)}),
    ]

    def init(self, *args, **kwargs):
        # State first: under a data split, super().init() trains and replays the
        # test window before it returns.
        self.trained_at: typing.Optional[datetime.datetime] = None
        self._forecasts: typing.Dict[typing.Tuple[int, pd.Timestamp], typing.Tuple[pd.Timestamp, dict]] = {}
        self._pv_sum: typing.Dict[pd.Timestamp, float] = {}
        self._pv_count: typing.Dict[pd.Timestamp, int] = {}
        self._pv_seed: typing.Dict[pd.Timestamp, float] = {}
        self._memo: typing.Dict[pd.Timestamp, typing.Tuple[tuple, float]] = {}
        self._seeded_from: typing.Optional[int] = None
        self._messages = 0
        super().init(*args, **kwargs)

    # ------------------------------------------------------------------ state

    def _python_model(self, model: PyFuncModel):
        try:
            return model.unwrap_python_model()
        except Exception:
            return None

    def _seed(self, pm) -> None:
        """Take the PV hours and issued forecasts the model saw in training."""
        if pm is None or self._seeded_from == id(pm):
            return
        self._seeded_from = id(pm)
        for hour, value in getattr(pm, "pv_recent", {}).items():
            self._pv_seed[to_utc(hour)] = float(value)
        for lead, ff, ia, values in getattr(pm, "fc_recent", []):
            self._store_forecast(int(lead), to_utc(ff), to_utc(ia), values)

    def _store_forecast(self, lead: int, ff: pd.Timestamp, ia: pd.Timestamp, values: dict) -> None:
        if ff is None or ia is None:
            return
        key = (lead, ff.floor("h"))
        current = self._forecasts.get(key)
        if current is None or ia >= current[0]:
            self._forecasts[key] = (ia, values)

    def _forecast_value(self, var: str, hour: pd.Timestamp, now: pd.Timestamp, lead_pref: int):
        leads = [lead_pref] + [l for l in (1, 2, 3, 4, 5, 6, 7) if l != lead_pref]
        for lead in leads:
            entry = self._forecasts.get((lead, hour))
            if entry is not None and entry[0] <= now:
                value = entry[1].get(var)
                if value is not None:
                    return value
        return None

    def _pv_value(self, hour: pd.Timestamp):
        count = self._pv_count.get(hour)
        if count:
            return self._pv_sum[hour] / count
        return self._pv_seed.get(hour)

    def _prune(self, now: pd.Timestamp) -> None:
        old_fc = now - pd.Timedelta(days=3)
        old_pv = now - pd.Timedelta(days=10)
        self._forecasts = {k: v for k, v in self._forecasts.items() if k[1] >= old_fc}
        for store in (self._pv_sum, self._pv_count, self._pv_seed):
            for hour in [h for h in store if h < old_pv]:
                del store[hour]
        self._memo = {h: v for h, v in self._memo.items() if h >= now}

    # ---------------------------------------------------------------- infer

    def infer(
        self,
        model: typing.Optional[PyFuncModel],
        data: typing.Dict[str, typing.Any],
        selector: str,
        device_id: str,
        timestamp: datetime.datetime,
    ) -> typing.Tuple[
        typing.Optional[datetime.datetime], typing.Optional[typing.Any], typing.Optional[PythonModel]
    ]:
        now = to_utc(timestamp)
        if now is None:
            return None, None, None
        pm = self._python_model(model) if model is not None else None
        self._seed(pm)

        self._messages += 1
        if self._messages % 5000 == 0:
            self._prune(now)

        if selector == "weather":
            ff = to_utc(data.get("forecasted_for"))
            ia = to_utc(data.get("issued_at"))
            try:
                lead = int(float(data.get("lead_days")))
            except (TypeError, ValueError):
                return None, None, None
            if ia is None and ff is not None:
                ia = ff - pd.Timedelta(days=lead)
            values = {}
            for v in WEATHER_VARS:
                try:
                    values[v] = None if data.get(v) is None else float(data.get(v))
                except (TypeError, ValueError):
                    values[v] = None
            self._store_forecast(lead, ff, ia, values)
            return None, None, None

        # selector == "pv"
        raw = data.get(PV_FIELD)
        try:
            value = None if raw is None else float(raw)
        except (TypeError, ValueError):
            value = None
        if value is not None:
            hour_now = now.floor("h")
            self._pv_sum[hour_now] = self._pv_sum.get(hour_now, 0.0) + value
            self._pv_count[hour_now] = self._pv_count.get(hour_now, 0) + 1

        if pm is None or not hasattr(pm, "predict_hour"):
            return None, None, None

        target_time = now + HORIZON
        target_hour = target_time.floor("h")
        fc_values = {}
        for o in FC_OFFSETS:
            hour = target_hour + pd.Timedelta(hours=o)
            for v in WEATHER_VARS:
                fc_values[f"{v}_{o:+d}"] = self._forecast_value(v, hour, now, pm.lead_days)
        lag_values = [self._pv_value(target_hour - pd.Timedelta(days=d)) for d in LAG_DAYS]

        key = tuple(fc_values[k] for k in sorted(fc_values)) + tuple(lag_values)
        cached = self._memo.get(target_hour)
        if cached is not None and cached[0] == key:
            prediction = cached[1]
        else:
            prediction = pm.predict_hour(target_hour, fc_values, lag_values)
            self._memo[target_hour] = (key, prediction)

        result = {
            "prediction": prediction,
            "forecast_hour": target_hour.isoformat(),
            "horizon_h": HORIZON.total_seconds() / 3600.0,
        }
        return target_time.to_pydatetime(), result, None

    # ---------------------------------------------------------------- train

    def train(
        self, model: typing.Optional[PyFuncModel], logger: TrainMlflowLogger
    ) -> typing.Optional[PythonModel]:
        self.trained_at = datetime.datetime.now(datetime.timezone.utc)
        return train_model(logger)

    def need_retraining(self, model: typing.Optional[PyFuncModel]) -> bool:
        if model is None:
            return True
        if self.trained_at is None:
            return False
        age = datetime.datetime.now(datetime.timezone.utc) - self.trained_at
        return age.total_seconds() >= self.config.retrain_after_s
