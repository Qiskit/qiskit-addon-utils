# This code is a Qiskit project.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Reminder: update the RST file in docs/apidocs when adding new interfaces.
"""XSlow gate."""

from qiskit.circuit import ControlFlowOp
from qiskit.dagcircuit import DAGOpNode
from qiskit.transpiler.exceptions import TranspilerError


def validate_op_is_supported(node: DAGOpNode):
    """Raises if the given node contains an operation that is not supported by the post selection passes.

    The supported operations are:

        * Standard gates.
        * Barriers.
        * Measurements.
        * Control-flow operations.

    Args:
        node: The node to validate.

    Raises:
        TranspilerError: If ``node`` contains an unsupported operation.
    """
    if (
        node.is_standard_gate()
        or node.op.name in ["barrier", "measure"]
        or isinstance(node.op, ControlFlowOp)
    ):
        return
    raise TranspilerError(f"``'{node.op.name}'`` is not supported.")
