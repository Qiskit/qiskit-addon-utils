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
"""Transpiler pass to add post-selection measurements on spectator qubits."""

from __future__ import annotations

from copy import deepcopy
from enum import Enum

import numpy as np
from qiskit.circuit import ClassicalRegister, ControlFlowOp, Qubit
from qiskit.circuit.library import Barrier, Measure, RXGate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError

from ....constants import DEFAULT_POST_SELECTION_SUFFIX, DEFAULT_SPECTATOR_CREG_NAME
from .utils import validate_op_is_supported
from .xslow_gate import XSlowGate


class XPulseType(str, Enum):
    """The type of X-pulse to apply between the two spectator measurements."""

    XSLOW = "xslow"
    """An ``xslow`` gate."""

    RX = "rx"
    """Twenty ``rx`` gates with angles ``pi/20``."""


class AddSpectatorMeasures(TransformationPass):
    """Add post-selection measurements on spectator qubits.

    An *active* qubit is a qubit acted on in the circuit by a non-barrier
    instruction. A *terminated* qubit is one whose last action is a
    measurement. A *spectator* qubit is a qubit that is inactive, but adjacent
    to an active qubit under the coupling map.

    For every spectator qubit (and, optionally via ``include_unmeasured``,
    every unterminated active qubit), the pass appends the same parity check
    used for data-qubit post-selection:

    1. ``measure`` into the spectator register (expected to read 0).
    2. A narrowband X-pulse — either an ``xslow`` gate or 20 ``rx(pi/20)``
       rotations — that should flip the qubit's state by ``pi``.
    3. ``measure`` again, into the post-selection spectator register; this
       reading should be the bit-flip of the first.

    Two classical registers are added: one named ``spectator_creg_name``
    (default ``"spec"``) holding the first measurement, and one named
    ``spectator_creg_name + post_selection_suffix`` (default ``"spec_ps"``)
    holding the second. A shot is kept when the two registers disagree on a
    given qubit (the qubit successfully flipped).

    When the input circuit already contains a data-qubit post-selection
    structure produced by :class:`.AddPostSelectionMeasures`, this pass
    integrates the spectator parity check into that structure: the spectator
    measurements share the existing pre-/post-pulse barriers and the spectator
    pulses run alongside the data-qubit pulses.

    .. note::
        When this pass encounters a control flow operation, it iterates
        through all of its blocks. It marks as *active* every qubit that is
        active within at least one of the blocks, and as *terminated* every
        qubit that is terminated in every one of the blocks.
    """

    def __init__(
        self,
        coupling_map: CouplingMap | list[tuple[int, int]],
        x_pulse_type: str | XPulseType = XPulseType.XSLOW,  # type: ignore
        *,
        include_unmeasured: bool = True,
        spectator_creg_name: str = DEFAULT_SPECTATOR_CREG_NAME,
        ignore_creg_suffixes: list[str] | None = None,
        post_selection_suffix: str = DEFAULT_POST_SELECTION_SUFFIX,
    ):
        """Initialize the pass.

        Args:
            coupling_map: A coupling map or a list of tuples indicating pairs of neighboring qubits.
            x_pulse_type: The type of X-pulse to apply between the two spectator measurements.
            include_unmeasured: Whether qubits that are active but not terminated should also be
                treated as spectators. If ``True``, the parity check is added to each of them as well.
            spectator_creg_name: The name of the classical register holding the first spectator
                measurement. The post-selection register is named
                ``spectator_creg_name + post_selection_suffix``.
            ignore_creg_suffixes: A list of suffixes for classical registers that should be ignored
                when determining active/terminated qubits. By default, registers ending with
                ``"_pre"`` are ignored so that pre-selection measurements aren't treated as regular
                terminations.
            post_selection_suffix: The suffix appended to ``spectator_creg_name`` to form the
                post-selection register name, and used to identify the data-qubit post-selection
                barriers that the spectator parity check is integrated into. Defaults to ``"_ps"``.
        """
        super().__init__()
        self.x_pulse_type = XPulseType(x_pulse_type)
        self.spectator_creg_name = spectator_creg_name
        self.include_unmeasured = include_unmeasured
        self.coupling_map = (
            deepcopy(coupling_map)
            if isinstance(coupling_map, CouplingMap)
            else CouplingMap(couplinglist=coupling_map)
        )
        self.coupling_map.make_symmetric()
        self.ignore_creg_suffixes = (
            ignore_creg_suffixes if ignore_creg_suffixes is not None else ["_pre"]
        )
        self.post_selection_suffix = post_selection_suffix

        # Same pulse sequence as ``AddPostSelectionMeasures``: a single full pi rotation
        # delivered as one ``xslow`` or as 20 fine ``rx`` rotations.
        if self.x_pulse_type == XPulseType.XSLOW:
            self.pulse_sequence = [XSlowGate()]
        else:
            self.pulse_sequence = [RXGate(np.pi / 20)] * 20

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

        if (num_spectators := len(spectator_qubits)) == 0:
            return dag

        spectator_qubits_ls = sorted(spectator_qubits, key=lambda q: qubit_map[q])

        spec_creg = ClassicalRegister(num_spectators, self.spectator_creg_name)
        spec_ps_creg = ClassicalRegister(
            num_spectators, self.spectator_creg_name + self.post_selection_suffix
        )

        # Find data qubits already carrying a post-selection measurement (i.e. those
        # touched by ``AddPostSelectionMeasures``). When present, we integrate the
        # spectator parity check into the existing barrier/pulse/barrier sandwich
        # rather than building our own.
        data_with_postsel: set[Qubit] = set()
        for node in dag.topological_op_nodes():
            if node.op.name == "measure" and len(node.cargs) == 1:
                clbit = node.cargs[0]
                for creg in dag.cregs.values():
                    if clbit in creg and creg.name.endswith(self.post_selection_suffix):
                        data_with_postsel.add(node.qargs[0])
                        break

        # Existing post-sel barriers are exactly the pair acting on ``data_with_postsel``;
        # the last two such barriers in topological order are the ones we want to extend.
        # Earlier matches (e.g. the pre-selection barrier when its qubit set happens to
        # coincide with the post-selection one) are intentionally ignored.
        matching_barriers = [
            n
            for n in dag.topological_op_nodes()
            if n.op.name == "barrier" and set(n.qargs) == data_with_postsel
        ]

        if data_with_postsel and len(matching_barriers) >= 2:
            return self._integrate_with_postsel(
                dag,
                spectator_qubits_ls,
                data_with_postsel,
                matching_barriers[-2]._node_id,
                matching_barriers[-1]._node_id,
                qubit_map,
                spec_creg,
                spec_ps_creg,
            )
        return self._add_standalone(
            dag,
            spectator_qubits_ls,
            terminated_qubits,
            qubit_map,
            spec_creg,
            spec_ps_creg,
        )

    def _add_standalone(
        self,
        dag: DAGCircuit,
        spectator_qubits_ls: list[Qubit],
        terminated_qubits: set[Qubit],
        qubit_map: dict[Qubit, int],
        spec_creg: ClassicalRegister,
        spec_ps_creg: ClassicalRegister,
    ) -> DAGCircuit:
        """Append the spectator parity check when no data-qubit post-selection is present."""
        dag.add_creg(spec_creg)
        dag.add_creg(spec_ps_creg)

        # Sync barrier across terminated data and the about-to-be-measured spectator qubits.
        sync_qubits = sorted(
            terminated_qubits.union(set(spectator_qubits_ls)), key=lambda q: qubit_map[q]
        )
        dag.apply_operation_back(Barrier(len(sync_qubits)), sync_qubits)

        # First spectator measurement.
        for qubit, clbit in zip(spectator_qubits_ls, spec_creg):
            dag.apply_operation_back(Measure(), [qubit], [clbit])

        # Sync barrier before the pi-rotation pulses.
        dag.apply_operation_back(Barrier(len(spectator_qubits_ls)), spectator_qubits_ls)

        # Pi-rotation pulses on each spectator qubit.
        for qubit in spectator_qubits_ls:
            for gate in self.pulse_sequence:
                dag.apply_operation_back(gate, [qubit])

        # Sync barrier before the second measurement.
        dag.apply_operation_back(Barrier(len(spectator_qubits_ls)), spectator_qubits_ls)

        # Second spectator measurement (post-selection check).
        for qubit, clbit in zip(spectator_qubits_ls, spec_ps_creg):
            dag.apply_operation_back(Measure(), [qubit], [clbit])

        return dag

    def _integrate_with_postsel(
        self,
        dag: DAGCircuit,
        spectator_qubits_ls: list[Qubit],
        data_with_postsel: set[Qubit],
        barrier1_id: int,
        barrier2_id: int,
        qubit_map: dict[Qubit, int],
        spec_creg: ClassicalRegister,
        spec_ps_creg: ClassicalRegister,
    ) -> DAGCircuit:
        r"""Splice the spectator parity check into an existing post-sel barrier sandwich.

        Each spectator qubit is paired with a single data neighbour; pairs that
        share a data partner are bundled together. Just before the paired data
        qubit's terminal measurement we emit a *small* barrier covering only the
        data qubit and its partner spec(s), then re-emit the data terminal
        measurement and the spec qubit(s)' first measurement. This forces the
        synchronised pair of measurements without forcing every data qubit to
        idle on a full-width sync.

        Two existing barriers, ``barrier1`` and ``barrier2`` (originally
        emitted by :class:`.AddPostSelectionMeasures` on the data qubits only),
        are extended to cover the spectator qubits as well; together they
        sandwich the pi-rotation pulses on both data and spec qubits.

        Data qubits whose terminal measurement is buried inside a control-flow
        op (or that have no spec neighbour) are left untouched: their
        measurement happens whenever it is naturally scheduled.
        """
        all_topo_nodes = list(dag.topological_op_nodes())
        barrier1_idx = next(i for i, n in enumerate(all_topo_nodes) if n._node_id == barrier1_id)

        # Pair each spec qubit with one data neighbour: deterministic, choose
        # the neighbour with the smallest qubit index. A spec qubit whose only
        # data neighbour has its terminal measurement buried inside control flow
        # is unpaired and falls through to the (extended) ``barrier1`` for sync.
        data_terminal_nodes_full: dict[Qubit, DAGOpNode] = {}
        for node in all_topo_nodes[:barrier1_idx]:
            if (
                node.op.name == "measure"
                and len(node.qargs) == 1
                and node.qargs[0] in data_with_postsel
            ):
                data_terminal_nodes_full[node.qargs[0]] = node

        spec_to_data: dict[Qubit, Qubit] = {}
        for spec_q in spectator_qubits_ls:
            spec_idx = qubit_map[spec_q]
            data_neighbours = sorted(
                (
                    dag.qubits[n]
                    for n in self.coupling_map.neighbors(spec_idx)
                    if n < dag.num_qubits() and dag.qubits[n] in data_terminal_nodes_full
                ),
                key=lambda q: qubit_map[q],
            )
            if data_neighbours:
                spec_to_data[spec_q] = data_neighbours[0]

        # Bundle: data qubit -> [its paired spec qubits, sorted by index].
        data_to_specs: dict[Qubit, list[Qubit]] = {}
        for spec_q, data_q in spec_to_data.items():
            data_to_specs.setdefault(data_q, []).append(spec_q)
        for specs in data_to_specs.values():
            specs.sort(key=lambda q: qubit_map[q])

        # Defer ONLY the data terminal measure nodes that are paired with at
        # least one spec qubit. Unpaired data qubits keep their measurement in
        # its natural position.
        data_terminal_nodes: dict[Qubit, DAGOpNode] = {
            q: data_terminal_nodes_full[q] for q in data_to_specs
        }
        deferred_terminal_node_ids = {n._node_id for n in data_terminal_nodes.values()}

        # Spectator-only ops that landed *after* the first post-sel barrier in the
        # topological sort are logically pre-selection ops on the spectator wires
        # (e.g. ``measure -> reset`` from ``AddSpectatorMeasuresPreSelection``).
        # Defer them so they emerge before the spec parity check on the spec wires.
        spec_qubit_set = set(spectator_qubits_ls)
        late_spec_nodes = [
            n
            for n in all_topo_nodes[barrier1_idx + 1 :]
            if n.op.name != "barrier"
            and len(n.qargs) > 0
            and all(q in spec_qubit_set for q in n.qargs)
        ]
        late_spec_node_ids = {n._node_id for n in late_spec_nodes}

        new_dag = DAGCircuit()
        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)
        new_dag.add_creg(spec_creg)
        new_dag.add_creg(spec_ps_creg)

        spec_q_indices = [dag.qubits.index(q) for q in spectator_qubits_ls]
        spec_q_new = [new_dag.qubits[i] for i in spec_q_indices]
        # Look-up from spec qubit (old DAG) -> (new DAG qubit, spec_creg clbit).
        spec_clbit_for: dict[Qubit, tuple[Qubit, object]] = {
            spec: (new_dag.qubits[dag.qubits.index(spec)], spec_creg[i])
            for i, spec in enumerate(spectator_qubits_ls)
        }

        extended = sorted(
            data_with_postsel.union(set(spectator_qubits_ls)), key=lambda q: qubit_map[q]
        )
        extended_new = [new_dag.qubits[dag.qubits.index(q)] for q in extended]

        # Stable iteration order for the per-data-qubit bundles.
        bundled_data_qubits = sorted(data_to_specs, key=lambda q: qubit_map[q])

        for node in all_topo_nodes:
            if node._node_id in late_spec_node_ids or node._node_id in deferred_terminal_node_ids:
                continue
            if node._node_id == barrier1_id:
                # Flush deferred spec-only ops so they appear before the spec
                # parity check on the spec wires.
                for spec_node in late_spec_nodes:
                    new_dag.apply_operation_back(spec_node.op, spec_node.qargs, spec_node.cargs)
                # For each (data qubit, paired specs) bundle: emit a small
                # barrier on (data + specs), then the data terminal measure,
                # then each spec's first measurement. Synchronises just the
                # bundle without idling any other data qubit.
                for data_q in bundled_data_qubits:
                    specs = data_to_specs[data_q]
                    bundle_old = sorted([data_q, *specs], key=lambda q: qubit_map[q])
                    bundle_new = [new_dag.qubits[dag.qubits.index(q)] for q in bundle_old]
                    new_dag.apply_operation_back(Barrier(len(bundle_new)), bundle_new)
                    tm_node = data_terminal_nodes[data_q]
                    new_dag.apply_operation_back(tm_node.op, tm_node.qargs, tm_node.cargs)
                    for spec_q in specs:
                        spec_q_n, spec_clbit = spec_clbit_for[spec_q]
                        new_dag.apply_operation_back(Measure(), [spec_q_n], [spec_clbit])
                # Spec qubits with no paired data neighbour still need their
                # first measurement before the extended barrier1.
                for spec_q in spectator_qubits_ls:
                    if spec_q in spec_to_data:
                        continue
                    spec_q_n, spec_clbit = spec_clbit_for[spec_q]
                    new_dag.apply_operation_back(Measure(), [spec_q_n], [spec_clbit])
                # Replace barrier1 with the extended version.
                new_dag.apply_operation_back(Barrier(len(extended_new)), extended_new)
                # Spectator pulses run alongside the data-qubit pulses between the two barriers.
                for q in spec_q_new:
                    for gate in self.pulse_sequence:
                        new_dag.apply_operation_back(gate, [q])
            elif node._node_id == barrier2_id:
                # Replace the second post-sel barrier with the extended version.
                new_dag.apply_operation_back(Barrier(len(extended_new)), extended_new)
                # Second spectator measurement (parallel to the ``c_ps`` measure).
                for q, clbit in zip(spec_q_new, spec_ps_creg):
                    new_dag.apply_operation_back(Measure(), [q], [clbit])
            else:
                new_dag.apply_operation_back(node.op, node.qargs, node.cargs)

        return new_dag

    def _find_active_and_terminated_qubits(self, dag: DAGCircuit) -> tuple[set[Qubit], set[Qubit]]:
        """Helper function to find the sets of active qubits and of qubits terminated with measurements.

        This method recurses into control flow operations.
        """
        # Pre-selection-only qubits: their only measurements are into ignored
        # (pre-sel) registers. Every standard gate on such a qubit is part of
        # the pre-selection pulse sequence (e.g. the trailing X in
        # ``xslow -> X -> measure(_pre)``) and must NOT be counted as making
        # the qubit "active". Without this, spectator pre-selection qubits
        # (which only have a pre-sel sequence on them) would be misclassified
        # as data qubits in the post-selection pass, and *their* neighbours
        # would in turn be picked up as post-sel spectators.
        pre_select_only_qubits = self._find_pre_select_only_qubits(dag)

        # The qubits that undergo any non-barrier action
        active_qubits: set[Qubit] = set()

        # The qubits whose last action is a measurement
        terminated_qubits: set[Qubit] = set()

        for node in dag.topological_op_nodes():
            validate_op_is_supported(node)

            # Skip xslow, rx, and reset gates - they are part of pre/post-selection protocol
            if ("xslow" in node.op.name) or ("rx" in node.op.name) or (node.op.name == "reset"):
                continue
            elif node.is_standard_gate():
                # Filter out qargs that participate only in pre-selection — their
                # standard gates (typically the trailing X in the pre-sel pulse
                # sequence) are protocol gates, not main-circuit activity.
                relevant_qargs = [q for q in node.qargs if q not in pre_select_only_qubits]
                if relevant_qargs:
                    active_qubits.update(relevant_qargs)
                    terminated_qubits.difference_update(relevant_qargs)
            elif (name := node.op.name) == "barrier":
                continue
            elif name == "measure":
                # Check if this is a measurement into an ignored register
                if len(node.cargs) == 1:  # pragma: no branch
                    clbit = node.cargs[0]
                    is_ignored = False
                    for creg in dag.cregs.values():
                        if clbit in creg and any(
                            creg.name.endswith(suffix) for suffix in self.ignore_creg_suffixes
                        ):
                            is_ignored = True
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

    def _find_pre_select_only_qubits(self, dag: DAGCircuit) -> set[Qubit]:
        """Qubits whose only measurements go to ignored (pre-selection) registers.

        Recurses through control flow. Measurements buried inside control-flow
        blocks are treated as primary measurements — pre-selection
        measurements are emitted at the top level by the pre-selection passes,
        so any in-block measurement is by construction a main-circuit one.
        """
        qubits_with_ignored_meas: set[Qubit] = set()
        qubits_with_primary_meas: set[Qubit] = set()

        def visit(sub_dag: DAGCircuit, qmap: dict[Qubit, Qubit] | None) -> None:
            for node in sub_dag.topological_op_nodes():
                if node.op.name == "measure" and len(node.cargs) == 1:
                    qubit = node.qargs[0]
                    if qmap is not None:
                        qubit = qmap.get(qubit, qubit)
                    if qmap is None:
                        clbit = node.cargs[0]
                        is_ignored = False
                        for creg in sub_dag.cregs.values():
                            if clbit in creg and any(
                                creg.name.endswith(suffix) for suffix in self.ignore_creg_suffixes
                            ):
                                is_ignored = True
                                qubits_with_ignored_meas.add(qubit)
                                break
                        if not is_ignored:
                            qubits_with_primary_meas.add(qubit)
                    else:
                        # Inside control flow: treat as a main-circuit measurement.
                        qubits_with_primary_meas.add(qubit)
                elif isinstance(node.op, ControlFlowOp):
                    for block in node.op.blocks:
                        block_dag = circuit_to_dag(block)
                        inner_map = {
                            bq: (qmap.get(pq, pq) if qmap is not None else pq)
                            for bq, pq in zip(block_dag.qubits, node.qargs)
                        }
                        visit(block_dag, inner_map)

        visit(dag, None)
        return qubits_with_ignored_meas - qubits_with_primary_meas
