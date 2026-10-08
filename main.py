"""Entry point. Operator Lib owns the process; the operator owns the modelling.

Kept thin on purpose: everything about Kafka, configuration, the model registry
and the lifecycle is Operator Lib's, and code added here runs outside all of it.
"""

from op import Operator

from operator_lib.operator_lib import OperatorLib


if __name__ == "__main__":
    OperatorLib(
        Operator(),
        name="SENERGY-Platform/ode-ablation-pv-l1",
        # Written by the Dockerfile from the build's commit, so a running operator
        # can say which source it is (§5.11 item 7).
        git_info_file="git_commit",
    )
