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
"""Transpiler pass to add post selection measurements."""

from __future__ import annotations

from enum import Enum

import numpy as np
from qiskit.circuit import ClassicalRegister, Clbit, ControlFlowOp, Qubit
from qiskit.circuit.library import Barrier, Measure, RXGate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError

from ....constants import DEFAULT_POST_SELECTION_SUFFIX
from .utils import validate_op_is_supported
from .xslow_gate import XSlowGate


class XPulseType(str, Enum):
    """The type of X-pulse to apply for the post-selection measurements."""

    XSLOW = "xslow"
    """An ``xslow`` gate."""

    RX = "rx"
    """Twenty ``rx`` gates with angles ``pi/20``."""


class AddPostSelectionMeasures(TransformationPass):
    """Add a post selection measurement after every terminal measurement.

    A post selection measurement is a measurement that follows a regular measurement on a given qubit. It
    consists of a narrowband X-pulse followed by a regular measurement operation. In the absence of noise,
    it is expected to return ``(b + 1) % 2``, where ``b`` is the outcome of the original measurement.

    This pass adds post selection measurements after every terminal measurement, i.e., after every measurement
    that is not followed by another operation on the same wire. The added measurements are placed after a
    barrier, and write to new classical registers that are copies of the DAG's registers, with modified
    names.

    .. note::
        When this pass encounters a control flow operation, it iterates through all of its blocks. It marks
        as "terminated" only those qubits that are terminated in every one of the blocks, and it treats as
        unterminated every other qubit.
    """

    def __init__(
        self,
        x_pulse_type: str | XPulseType = XPulseType.XSLOW,  # type: ignore
        *,
        post_selection_suffix: str = DEFAULT_POST_SELECTION_SUFFIX,
    ):
        """Initialize the pass.

        Args:
            x_pulse_type: The type of X-pulse to apply for the post-selection measurements.
            post_selection_suffix: A fixed suffix to append to the names of the classical registers when copying them.
        """
        super().__init__()
        self.x_pulse_type = XPulseType(x_pulse_type)
        self.post_selection_suffix = post_selection_suffix

        if self.x_pulse_type == XPulseType.XSLOW:
            self.pulse_sequence = [XSlowGate()]
        else:
            self.pulse_sequence = [RXGate(np.pi / 20)] * 20

    def run(self, dag: DAGCircuit):  # noqa: D102
        # Find what qubits have a terminal measurement
        terminal_measurements: dict[Qubit, Clbit] = {
            qubit: clbit for qubit, clbit in self._find_terminal_measurements(dag).items() if clbit
        }
        if not terminal_measurements:
            return dag

        # Add the new registers and create a map between the original clbit and the new ones
        clbits_map = {}
        for name, creg in dag.cregs.items():
            dag.add_creg(
                new_creg := ClassicalRegister(creg.size, name + self.post_selection_suffix)
            )
            clbits_map.update({clbit: clbit_ps for clbit, clbit_ps in zip(creg, new_creg)})

        # Add a barrier to separate the post selection measurements from the rest of the circuit
        qubits = tuple(terminal_measurements)
        dag.apply_operation_back(Barrier(len(qubits)), qubits)

        # Append the post selection measurements
        for qubit, clbit in terminal_measurements.items():
            for gate in self.pulse_sequence:
                dag.apply_operation_back(gate, [qubit])
            dag.apply_operation_back(Measure(), [qubit], [clbits_map[clbit]])

        return dag

    def _find_terminal_measurements(self, dag: DAGCircuit) -> dict[Qubit, Clbit]:
        """Helper function to find the terminal measurements.

        This function returns a map between qubits to ``None`` (for qubits not terminated by measurements)
        or :class:`.Clbit`s (for qubits that are terminated by measurements). It is used recursively for boxes.

        Args:
            dag: The dag to iterate over.
        """
        # Map from terminal qubits to the bits they measure into, or ``None`` if they are unterminated.
        terminal_measurements: dict[Qubit, Clbit | None] = {}

        for node in dag.topological_op_nodes():
            validate_op_is_supported(node)

            if node.is_standard_gate():
                for qarg in node.qargs:
                    terminal_measurements[qarg] = None
            elif (name := node.op.name) == "barrier":
                continue
            elif name == "measure":
                terminal_measurements[node.qargs[0]] = node.cargs[0]
            elif isinstance(node.op, ControlFlowOp):
                # A list of terminal measurements dictionaries, one per block
                all_terminal_measurements: list[dict[Qubit, Clbit]] = []

                for block in node.op.blocks:
                    block_dag = circuit_to_dag(block)

                    clbit_map = {
                        block_clbit: clbit
                        for block_clbit, clbit in zip(block_dag.clbits, node.cargs)
                    }
                    qubit_map = {
                        block_qubit: qubit
                        for block_qubit, qubit in zip(block_dag.qubits, node.qargs)
                    }

                    all_terminal_measurements.append(
                        {
                            qubit_map[qubit]: clbit_map[clbit] if clbit else None
                            for qubit, clbit in self._find_terminal_measurements(block_dag).items()
                        }
                    )

                if len(all_terminal_measurements) == 1:
                    # If the control-flow op has a single block (e.g., it contains a BoxOp),
                    # simply update the terminal measurements.
                    terminal_measurements.update(all_terminal_measurements[0])
                else:
                    # Otherwise, make sure to mark as terminated only those qubits that
                    # write to the same clbit in every block
                    for qubit in node.qargs:
                        clbits = {d.get(qubit) for d in all_terminal_measurements}
                        if len(clbits) == 1:
                            terminal_measurements[qubit] = next(iter(clbits))
                        else:
                            terminal_measurements[qubit] = None
            else:  # pragma: no cover
                raise TranspilerError(f"``'{node.op.name}'`` is not supported.")

        return terminal_measurements
