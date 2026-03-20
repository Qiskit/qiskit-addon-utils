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
    ):
        """Initialize the pass.

        Args:
            coupling_map: A coupling map or a list of tuples indicating pairs of neighboring qubits.
            include_unmeasured: Whether the qubits that are active but are not terminated by a measurement should
                also be treated as spectators. If ``True``, a terminal measurement is added on each of them.
            spectator_creg_name: The name of the classical register added for the measurements on the spectator qubits.
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

        if len(spectator_qubits) == 0:
            return dag

        # Create spectator register
        spectator_qubits_ls = list(spectator_qubits)
        spectator_qubits_ls.sort(key=lambda qubit: qubit_map[qubit])
        spec_reg = ClassicalRegister(len(spectator_qubits_ls), self.spectator_creg_name)
        dag.add_creg(spec_reg)

        # Map spectator qubits to their classical bits
        spectator_clbit_map = {qubit: spec_reg[i] for i, qubit in enumerate(spectator_qubits_ls)}

        # Build adjacency graph: spectators <-> terminals
        # Find connected components where all qubits should be measured together
        spectator_to_terminals = {}
        terminal_to_spectators = {}

        for spectator_qubit in spectator_qubits:
            adjacent_terminals = [
                dag.qubits[neighbor_idx]
                for neighbor_idx in self.coupling_map.neighbors(qubit_map[spectator_qubit])
                if neighbor_idx < dag.num_qubits() and dag.qubits[neighbor_idx] in terminated_qubits
            ]
            spectator_to_terminals[spectator_qubit] = set(adjacent_terminals)

            for terminal_qubit in adjacent_terminals:
                if terminal_qubit not in terminal_to_spectators:
                    terminal_to_spectators[terminal_qubit] = set()
                terminal_to_spectators[terminal_qubit].add(spectator_qubit)

        # Find connected components using DFS
        visited_spectators = set()
        visited_terminals = set()
        measurement_groups = []  # Each group is (spectators, terminals) to measure together

        def dfs_find_component(start_spectator):
            """Find all spectators and terminals connected to start_spectator."""
            component_spectators = set()
            component_terminals = set()
            stack = [("spectator", start_spectator)]

            while stack:
                node_type, node = stack.pop()

                if node_type == "spectator":
                    if node in visited_spectators:
                        continue
                    visited_spectators.add(node)
                    component_spectators.add(node)

                    # Add all adjacent terminals to explore
                    for terminal in spectator_to_terminals.get(node, []):
                        if terminal not in visited_terminals:
                            stack.append(("terminal", terminal))

                else:  # node_type == 'terminal'
                    if node in visited_terminals:
                        continue
                    visited_terminals.add(node)
                    component_terminals.add(node)

                    # Add all adjacent spectators to explore
                    for spectator in terminal_to_spectators.get(node, []):
                        if spectator not in visited_spectators:
                            stack.append(("spectator", spectator))

            return component_spectators, component_terminals

        # Find all connected components
        for spectator_qubit in spectator_qubits:
            if spectator_qubit not in visited_spectators:
                spec_group, term_group = dfs_find_component(spectator_qubit)
                if term_group:  # Only add if there are terminals
                    measurement_groups.append((spec_group, term_group))

        # Map terminal measurement nodes for removal
        terminal_measurement_nodes = {}
        for node in dag.topological_op_nodes():
            if node.op.name == "measure" and node.qargs[0] in terminated_qubits:
                terminal_measurement_nodes[node.qargs[0]] = node

        # Process each measurement group
        for spectators, terminals in measurement_groups:
            # Remove terminal measurement nodes
            terminal_clbits = {}
            for terminal_qubit in terminals:
                node = terminal_measurement_nodes[terminal_qubit]
                terminal_clbits[terminal_qubit] = node.cargs[0]
                dag.remove_op_node(node)

            # Add barrier across ALL qubits in this component
            barrier_qubits = list(spectators) + list(terminals)
            dag.apply_operation_back(Barrier(len(barrier_qubits)), qargs=barrier_qubits)

            # Add measurements for all terminals
            for terminal_qubit in terminals:
                dag.apply_operation_back(
                    Measure(), qargs=[terminal_qubit], cargs=[terminal_clbits[terminal_qubit]]
                )

            # Add measurements for all spectators
            for spectator_qubit in spectators:
                dag.apply_operation_back(
                    Measure(), qargs=[spectator_qubit], cargs=[spectator_clbit_map[spectator_qubit]]
                )

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
