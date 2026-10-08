"""Training, on Ray.

Separate from op.py because the two run in different places: op.py runs in the
operator's own process for every message, while this runs distributed and rarely.

provide_historic_data() is Operator Lib's reader over the platform's timeseries
store. It returns Ray Datasets, so the training below never holds the whole
history in one process.
"""

import datetime
import typing

import ray
from mlflow.pyfunc import PythonModel

from operator_lib.util.helpers import provide_historic_data, TrainMlflowLogger


# How much history one training pass reads.
TRAINING_WINDOW = datetime.timedelta(days=90)


class OdeAblationPvL1Model(PythonModel):
    """The model MLflow registers and op.py later loads.

    predict() receives exactly the payload infer() builds, so the two are one
    contract in two files. Keep it serialisable: MLflow stores this object.
    """

    def __init__(self, baseline: float) -> None:
        self.baseline = baseline

    def predict(self, context, model_input=None, params=None):
        # The pyfunc signature carries a context when MLflow calls it and not when the
        # model is called directly, so the payload is taken from whichever argument
        # holds it.
        payload = model_input if model_input is not None else context
        value = float(payload.get("value", 0.0))
        return value - self.baseline


@ray.remote
def _fit(datasets: typing.List[typing.Any]) -> float:
    """The distributed part. Replace the body; keep the shape.

    A Ray task rather than a plain function so that training scales with the
    cluster rather than with the operator's pod.
    """
    total, count = 0.0, 0
    for dataset in datasets:
        for batch in dataset.iter_batches(batch_size=4096):
            values = batch.get("value")
            if values is None:
                continue
            for value in values:
                if value is None:
                    continue
                total += float(value)
                count += 1
    return total / count if count else 0.0


def train_model(logger: TrainMlflowLogger) -> typing.Optional[PythonModel]:
    """Read the history, fit, and hand back a model for MLflow to register."""
    with logger.trace("read history"):
        datasets = provide_historic_data(TRAINING_WINDOW)
    if not datasets:
        # Explicitly nothing rather than a model fitted on no data: returning None
        # leaves the previously registered model in place.
        return None

    with logger.trace("fit"):
        baseline = ray.get(_fit.remote(datasets))

    logger.log_params({"training_window_days": TRAINING_WINDOW.days})
    logger.log_metrics({"baseline": baseline})
    return OdeAblationPvL1Model(baseline=baseline)
