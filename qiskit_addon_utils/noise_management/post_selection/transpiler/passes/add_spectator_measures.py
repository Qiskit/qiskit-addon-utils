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
"""Transpiler pass to add measurement on spectator qubits."""

from __future__ import annotations

from copy import deepcopy

from qiskit.circuit import ClassicalRegister, ControlFlowOp, Qubit
from qiskit.circuit.library import Barrier, Measure
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError

from ....constants import DEFAULT_SPECTATOR_CREG_NAME
from .utils import validate_op_is_supported


class AddSpectatorMeasures(TransformationPass):
    """Add measurements on spectator qubits.

    An active qubit is a qubit acted on in the circuit by a non-barrier instruction. A terminated qubit is
    one whose last action is a measurement. A spectator qubit is a qubit that is inactive, but adjacent to
    an active qubit under the coupling map. This pass adds a measurement to all spectator qubits and,
    optionally via ``include_unmeasured``, to all active qubits that are not terminated qubits.

    The added measurements write to a new register that has one bit per spectator qubit and name
    ``spectator_creg_name``.

    .. note::
        When this pass encounters a control flow operation, it iterates through all of its blocks. It marks
        as "active" every qubit that is active within at least one of the blocks, and as "terminated" every
        qubit that is terminated in every one of the blocks.
    """

    def __init__(
        self,
        coupling_map: CouplingMap | list[tuple[int, int]],
        *,
        include_unmeasured: bool = True,
        spectator_creg_name: str = DEFAULT_SPECTATOR_CREG_NAME,
        add_barrier: bool = True,
    ):
        """Initialize the pass.

        Args:
            coupling_map: A coupling map or a list of tuples indicating pairs of neighboring qubits.
            include_unmeasured: Whether the qubits that are active but are not terminated by a measurement should
                also be treated as spectators. If ``True``, a terminal measurement is added on each of them.
            spectator_creg_name: The name of the classical register added for the measurements on the spectator qubits.
            add_barrier: Whether to add a barrier acting on all active and spectator qubits prior to the spectator
                measurements.
        """
        super().__init__()
        self.spectator_creg_name = spectator_creg_name
        self.include_unmeasured = include_unmeasured
        self.coupling_map = (
            deepcopy(coupling_map)
            if isinstance(coupling_map, CouplingMap)
            else CouplingMap(couplinglist=coupling_map)
        )
        self.coupling_map.make_symmetric()
        self.add_barrier = add_barrier

    def run(self, dag: DAGCircuit):  # noqa: D102
        active_qubits, terminated_qubits = self._find_active_and_terminated_qubits(dag)

        qubit_map = {qubit: idx for idx, qubit in enumerate(dag.qubits)}
        spectator_qubits = set(
            dag.qubits[neighbor_idx]
            for qubit in active_qubits
            for neighbor_idx in self.coupling_map.neighbors(qubit_map[qubit])
            if neighbor_idx < dag.num_qubits()
        )
        spectator_qubits.difference_update(terminated_qubits)

        if self.include_unmeasured:
            unterminated_qubits = active_qubits.difference(terminated_qubits)
            spectator_qubits = spectator_qubits.union(unterminated_qubits)

        if (num_spectators := len(spectator_qubits)) != 0:
            # sort the spectator qubits, so that qubit `i` writes to clbit `i`
            spectator_qubits_ls = list(spectator_qubits)
            spectator_qubits_ls.sort(key=lambda qubit: qubit_map[qubit])

            if self.add_barrier is True:
                qubits = active_qubits.union(spectator_qubits_ls)
                dag.apply_operation_back(Barrier(len(qubits)), qubits)

            dag.add_creg(new_reg := ClassicalRegister(num_spectators, self.spectator_creg_name))
            for qubit, clbit in zip(spectator_qubits_ls, new_reg):
                dag.apply_operation_back(Measure(), [qubit], [clbit])

        return dag

    def _find_active_and_terminated_qubits(self, dag: DAGCircuit) -> tuple[set[Qubit], set[Qubit]]:
        """Helper function to find the sets of active qubits and of qubits terminated with measurements.

        This method recurses into control flow operations.
        """
        # The qubits that undergo any non-barrier action
        active_qubits: set[Qubit] = set()

        # The qubits whose last action is a measurement
        terminated_qubits: set[Qubit] = set()

        for node in dag.topological_op_nodes():
            validate_op_is_supported(node)

            if node.is_standard_gate():
                active_qubits.update(node.qargs)
                terminated_qubits.difference_update(node.qargs)
            elif (name := node.op.name) == "barrier":
                continue
            elif name == "measure":
                active_qubits.add(node.qargs[0])
                terminated_qubits.add(node.qargs[0])
            elif isinstance(node.op, ControlFlowOp):
                # The qubits whose last action is a measurement, block by block
                all_terminated_qubits: list[set[Qubit]] = []

                for block in node.op.blocks:
                    block_dag = circuit_to_dag(block)
                    qubit_map = {
                        block_qubit: qubit
                        for block_qubit, qubit in zip(block_dag.qubits, node.qargs)
                    }

                    block_active_qubits, block_terminated_qubits = (
                        self._find_active_and_terminated_qubits(block_dag)
                    )

                    active_qubits.update({qubit_map[qubit] for qubit in block_active_qubits})

                    terminated_qubits.difference_update(block_dag.qubits)
                    all_terminated_qubits.append(
                        {qubit_map[qubit] for qubit in block_terminated_qubits}
                    )

                terminated_qubits.update(set.intersection(*all_terminated_qubits))
            else:  # pragma: no cover
                raise TranspilerError(f"``'{node.op.name}'`` is not supported.")

        return active_qubits, terminated_qubits
