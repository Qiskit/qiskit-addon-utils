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
        ignore_creg_suffixes: list[str] | None = None,
        post_selection_suffix: str = "_ps",
    ):
        """Initialize the pass.

        Args:
            coupling_map: A coupling map or a list of tuples indicating pairs of neighboring qubits.
            include_unmeasured: Whether the qubits that are active but are not terminated by a measurement should
                also be treated as spectators. If ``True``, a terminal measurement is added on each of them.
            spectator_creg_name: The name of the classical register added for the measurements on the spectator qubits.
            add_barrier: Whether to add a barrier acting on all active and spectator qubits prior to the spectator
                measurements.
            ignore_creg_suffixes: A list of suffixes for classical registers that should be ignored when determining
                active/terminated qubits. By default, registers ending with "_pre" are ignored to avoid treating
                pre-selection measurements as regular measurements.
            post_selection_suffix: The suffix used for post-selection classical registers. Used to identify which
                qubits have post-selection measurements for barrier extension. Defaults to "_ps".
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
        self.ignore_creg_suffixes = (
            ignore_creg_suffixes if ignore_creg_suffixes is not None else ["_pre"]
        )
        self.post_selection_suffix = post_selection_suffix

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

            # Find all qubits that have post-selection measurements (from AddPostSelectionMeasures)
            # These are qubits that have measurements into registers ending with the post_selection_suffix
            data_qubits_with_postselection = set()
            for node in dag.topological_op_nodes():
                if node.op.name == "measure" and len(node.cargs) == 1:
                    clbit = node.cargs[0]
                    for creg in dag.cregs.values():
                        if clbit in creg and creg.name.endswith(self.post_selection_suffix):
                            data_qubits_with_postselection.add(node.qargs[0])
                            break

            # DEBUG
            import sys

            print(
                f"DEBUG: Found {len(data_qubits_with_postselection)} post-selection qubits",
                file=sys.stderr,
            )
            print(f"DEBUG: Post-selection suffix: {self.post_selection_suffix}", file=sys.stderr)

            # Combine data qubits with post-selection and spectator qubits for unified barrier
            all_postselection_qubits = list(
                data_qubits_with_postselection.union(spectator_qubits_ls)
            )
            all_postselection_qubits.sort(key=lambda qubit: qubit_map[qubit])

            # Extend the barrier from AddPostSelectionMeasures to include spectator qubits
            # We need to find the LAST barrier that acts EXACTLY on post-selection qubits
            # This is the barrier right before post-selection measurements
            if self.add_barrier and len(data_qubits_with_postselection) > 0:
                # Find the last barrier that acts exactly on post-selection qubits
                # (not a superset, which would include pre-selection barriers)
                # Store the node ID instead of the node object itself
                last_barrier_node_id = None
                for node in dag.topological_op_nodes():
                    if (
                        node.op.name == "barrier"
                        and set(node.qargs) == data_qubits_with_postselection
                    ):
                        last_barrier_node_id = node._node_id

                # Rebuild the DAG to replace the barrier with an extended one
                if last_barrier_node_id is not None:
                    new_dag = DAGCircuit()
                    for qreg in dag.qregs.values():
                        new_dag.add_qreg(qreg)
                    for creg in dag.cregs.values():
                        new_dag.add_creg(creg)

                    # Map old qubits to new qubits by index
                    qubit_indices = [dag.qubits.index(q) for q in all_postselection_qubits]
                    new_qubits = [new_dag.qubits[i] for i in qubit_indices]

                    # Copy all operations, replacing the last barrier
                    for node in dag.topological_op_nodes():
                        if node._node_id == last_barrier_node_id:
                            # Replace with extended barrier using new DAG's qubits
                            new_dag.apply_operation_back(Barrier(len(new_qubits)), new_qubits)
                        else:
                            new_dag.apply_operation_back(node.op, node.qargs, node.cargs)

                    dag = new_dag

            # Add spectator measurements
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

        # Track measurements into pre-selection registers to handle them specially
        preselection_measurements: dict[Qubit, bool] = {}

        for node in dag.topological_op_nodes():
            validate_op_is_supported(node)

            # Skip xslow, rx, and reset gates - they are part of pre/post-selection protocol
            if ("xslow" in node.op.name) or ("rx" in node.op.name) or (node.op.name == "reset"):
                continue
            elif node.is_standard_gate():
                # Check if this is an X gate that's part of a pre-selection sequence
                # (X gate immediately before a measurement into a _pre register)
                if node.op.name == "x" and len(node.qargs) == 1:
                    # Look ahead to see if next operation on this qubit is a measurement into _pre
                    qubit = node.qargs[0]
                    successors = list(dag.successors(node))
                    is_preselection_x = False
                    for succ in successors:
                        if (
                            hasattr(succ, "op")
                            and hasattr(succ.op, "name")
                            and succ.op.name == "measure"
                            and len(succ.qargs) == 1
                            and succ.qargs[0] == qubit
                            and len(succ.cargs) == 1
                        ):
                            # Check if measuring into an ignored register
                            clbit = succ.cargs[0]
                            for creg in dag.cregs.values():
                                if clbit in creg and any(
                                    creg.name.endswith(suffix)
                                    for suffix in self.ignore_creg_suffixes
                                ):
                                    is_preselection_x = True
                                    break
                            break

                    if not is_preselection_x:
                        active_qubits.update(node.qargs)
                        terminated_qubits.difference_update(node.qargs)
                else:
                    active_qubits.update(node.qargs)
                    terminated_qubits.difference_update(node.qargs)
            elif (name := node.op.name) == "barrier":
                continue
            elif name == "measure":
                # Check if this is a measurement into an ignored register
                if len(node.cargs) == 1:
                    clbit = node.cargs[0]
                    is_ignored = False
                    for creg in dag.cregs.values():
                        if clbit in creg and any(
                            creg.name.endswith(suffix) for suffix in self.ignore_creg_suffixes
                        ):
                            is_ignored = True
                            preselection_measurements[node.qargs[0]] = True
                            break

                    if not is_ignored:
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
