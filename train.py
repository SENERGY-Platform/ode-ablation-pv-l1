"""Entry point of an experiment: train once, record the run, exit.

The deployed operator starts at main.py, where Operator Lib takes the process,
trains if no model is registered yet, and then consumes Kafka until it is stopped.
An experiment wants the first half of that and not the second, so this file runs
Operator Lib's own init sequence and stops exactly where main.py would enter the
loop it never leaves.

It is deliberately the same path rather than a smaller one. MLOperator.init() is
what sets the tracking URI, connects to Ray, calls train() and registers the
result, so a run started here does what the deployed operator does when it first
comes up. ODE adds only the commit tag on the run.

ODE runs this file. It is yours to change, but keep the init()/train_once() pair
at the end: everything a run records happens inside one of the two.

Needs Operator Lib v1.5.0 or newer, which pyproject.toml pins, and v1.7.0 for a
session with a data split.
"""

import json
import sys

import confluent_kafka
import mlflow
from mlflow import MlflowClient

import operator_lib.util as util

from op import Operator


def _already_registered(model_id: str) -> bool:
    """Whether a model is registered under this key.

    This is the same question MLOperator.init() asks before it trains, and it is
    asked here because init() does not report the answer. Getting it wrong costs a
    duplicate training pass, not a wrong result.
    """
    try:
        MlflowClient().get_model_version_by_alias(model_id, "production")
        return True
    except Exception:
        return False


def main() -> int:
    dep_config = util.DeploymentConfig()
    config_json = json.loads(dep_config.config)
    opr_config = util.OperatorConfig(config_json)
    util.init_logger(opr_config.config.logger_level, "SENERGY-Platform/ode-ablation-pv-l1")

    operator = Operator()

    # Built because init() takes them, never polled: this process stops before
    # operator.start(), and start() is the call that would subscribe. Constructing
    # a consumer does not open a connection, so an experiment needs no reachable
    # broker unless one of its input topics is replayed from Kafka rather than
    # read from timescale.
    kafka_consumer = confluent_kafka.Consumer(
        {
            "bootstrap.servers": dep_config.config_bootstrap_servers or "",
            "group.id": dep_config.config_application_id,
            "auto.offset.reset": dep_config.consumer_auto_offset_reset_config,
        },
        logger=util.logger,
    )
    kafka_producer = confluent_kafka.Producer(
        {"bootstrap.servers": dep_config.config_bootstrap_servers or ""},
        logger=util.logger,
    )

    filter_handler = util.create_filter_handler(
        opr_config.inputTopics, dep_config.pipeline_id, operator.selectors
    )

    typed_config = operator.configType(config_json.get("config", {}))

    # An experiment trains every time, and init() trains only when it finds no
    # model registered under this pipeline and operator. ODE keeps that pair stable
    # per developer and repository — so model versions accumulate under one key and
    # the "production" alias moves, as they do for a deployed operator — which means
    # the first run trains inside init() and every run after it does not.
    #
    # So the answer is taken first and the pass asked for when init() will not make
    # one. Asking unconditionally would train twice on the first run.
    mlflow.set_tracking_uri(opr_config.config.mlflow_url)
    model_id = f"pipeline-{dep_config.pipeline_id}_operator-{dep_config.operator_id}"

    # Under a data split (Operator Lib v1.7.0), test_end on the config switches
    # init() into its evaluation mode: it trains unconditionally, whatever the
    # registry holds, and then replays the test window itself. Asking for a
    # second training pass here would train twice on the same bounds — wasteful,
    # not wrong, but pointless — so this is the one case trains_inside_init is
    # true regardless of what the registry says. getattr guards a typed_config
    # from a library older than v1.7.0, which has no test_end attribute at all.
    trains_inside_init = not _already_registered(model_id) or bool(
        getattr(typed_config, "test_end", None)
    )

    operator.init(
        kafka_consumer=kafka_consumer,
        kafka_producer=kafka_producer,
        filter_handler=filter_handler,
        output_topic=dep_config.output,
        pipeline_id=dep_config.pipeline_id,
        operator_id=dep_config.operator_id,
        config=typed_config,
        result_error_handler=None,
    )

    if not trains_inside_init:
        # Operator Lib v1.4.0 and newer. On an older pin this raises
        # AttributeError, and the pin is in pyproject.toml, where the floor is
        # v1.5.0 for a separate reason: how history is read.
        operator.train_once()
    return 0


if __name__ == "__main__":
    sys.exit(main())
