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
"""Transpiler pass to add post check measurements."""

from __future__ import annotations

from enum import Enum

import numpy as np
from qiskit.circuit import ClassicalRegister, Clbit, ControlFlowOp, Qubit
from qiskit.circuit.library import Barrier, Measure, RXGate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError

from ...constants import DEFAULT_POST_CHECK_SUFFIX
from ..xslow_gate import XSlowGate
from ._utils import validate_op_is_supported


class XPulseType(str, Enum):
    """The type of X-pulse to apply for the post-check measurements."""

    XSLOW = "xslow"
    """An ``xslow`` gate."""

    RX = "rx"
    """Twenty ``rx`` gates with angles ``pi/20``."""


class AddPostCircuitBitFlipChecks(TransformationPass):
    """Add a post check measurement after every terminal measurement.

    A post check measurement is a measurement that follows a regular measurement on a given qubit. It
    consists of a narrowband X-pulse followed by a regular measurement operation. In the absence of noise,
    it is expected to return ``(b + 1) % 2``, where ``b`` is the outcome of the original measurement.

    This pass adds post check measurements after every terminal measurement, i.e., after every measurement
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
        post_check_suffix: str = DEFAULT_POST_CHECK_SUFFIX,
        ignore_creg_suffixes: list[str] | None = None,
    ):
        """Initialize the pass.

        Args:
            x_pulse_type: The type of X-pulse to apply for the post-check measurements.
            post_check_suffix: A fixed suffix to append to the names of the classical registers when copying them.
            ignore_creg_suffixes: A list of suffixes for classical registers that should be ignored (not copied).
                By default, registers ending with "_pre" are ignored to avoid adding post-check to pre-check registers.
        """
        super().__init__()
        self.x_pulse_type = XPulseType(x_pulse_type)
        self.post_check_suffix = post_check_suffix
        self.ignore_creg_suffixes = (
            ignore_creg_suffixes if ignore_creg_suffixes is not None else ["_pre"]
        )

        if self.x_pulse_type == XPulseType.XSLOW:
            self.pulse_sequence = [XSlowGate()]
        else:
            self.pulse_sequence = [RXGate(np.pi / 20)] * 20

    def run(self, dag: DAGCircuit):  # noqa: D102
        # Find what qubits have a terminal measurement
        all_terminal_measurements = self._find_terminal_measurements(dag)

        # Add the new registers and map each original clbit to its post-check copy. Skip registers
        # with ignored suffixes, registers whose post-check counterpart already exists, and registers
        # that *are* a post-check counterpart (re-suffixing them would chain another ``_ps``), so the
        # pass is safe to re-run and to run after ``AddSpectatorPostCircuitBitFlipChecks``.
        existing_creg_names = set(dag.cregs)
        suffix = self.post_check_suffix
        clbits_map = {}
        for name, creg in dag.cregs.items():
            if any(name.endswith(s) for s in self.ignore_creg_suffixes):
                continue
            if name + suffix in existing_creg_names:
                continue
            if name.endswith(suffix) and name[: -len(suffix)] in existing_creg_names:
                continue
            dag.add_creg(new_creg := ClassicalRegister(creg.size, name + suffix))
            clbits_map.update({clbit: clbit_ps for clbit, clbit_ps in zip(creg, new_creg)})

        # Keep only terminal measurements whose clbit got a post-check copy (excludes pre-check registers)
        terminal_measurements: dict[Qubit, Clbit] = {
            qubit: clbit
            for qubit, clbit in all_terminal_measurements.items()
            if clbit and clbit in clbits_map
        }
        if not terminal_measurements:
            return dag

        # Add a barrier before post-check to ensure all terminal measurements finish
        qubits = tuple(terminal_measurements)
        dag.apply_operation_back(Barrier(len(qubits)), qubits)

        # Apply all pulse sequences
        for qubit in terminal_measurements:
            for gate in self.pulse_sequence:
                dag.apply_operation_back(gate, [qubit])

        # Add a barrier before measurements - AddSpectatorPostCircuitBitFlipChecks will extend it
        dag.apply_operation_back(Barrier(len(qubits)), qubits)

        # Then add all measurements
        for qubit, clbit in terminal_measurements.items():
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

            # Skip reset operations - they are part of pre-check protocol
            if node.op.name == "reset":
                continue
            elif node.is_standard_gate() or (name := node.op.name) in ("xslow", "delay"):
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
                    # Single block (e.g. a BoxOp): just merge its terminal measurements.
                    terminal_measurements.update(all_terminal_measurements[0])
                else:
                    # Multiple blocks: a qubit is terminated only if it writes the same clbit in every block.
                    for qubit in node.qargs:
                        clbits = {d.get(qubit) for d in all_terminal_measurements}
                        if len(clbits) == 1:
                            terminal_measurements[qubit] = next(iter(clbits))
                        else:
                            terminal_measurements[qubit] = None
            else:  # pragma: no cover
                raise TranspilerError(f"``'{node.op.name}'`` is not supported.")

        return terminal_measurements
