# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Reminder: update the RST file in docs/apidocs when adding new interfaces.
"""Transpiler pass to add post-circuit bit-flip checks on spectator qubits."""

from __future__ import annotations

from copy import deepcopy
from typing import Literal

import numpy as np
from qiskit.circuit import ClassicalRegister, ControlFlowOp, Qubit
from qiskit.circuit.library import Barrier, Measure, RXGate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError

from ...constants import (
    DEFAULT_POST_CHECK_SUFFIX,
    DEFAULT_SPECTATOR_CREG_NAME,
    RX_PULSE_COUNT,
)
from ..xslow_gate import XSlowGate
from ._utils import validate_op_is_supported
from .x_pulse_type import XPulseType


class AddSpectatorPostCircuitBitFlipChecks(TransformationPass):
    r"""Add bit-flip checks at the end of the circuit on qubits adjacent to active qubits.

    Each spectator qubit receives a post-circuit bit-flip check: a measurement, then a narrowband X-pulse
    that flips the qubit's state (:math:`|x\rangle\mapsto|x\oplus1\rangle`), then a second measurement.
    If the QPU fails to flip the qubit between the two measurements on a given shot, that sample may be
    considered unreliable and discarded. Postselecting only samples that pass all checks can improve the
    fidelity of distributions sampled from the QPU. Optionally via ``include_unmeasured``, active qubits
    that are not terminated by a measurement are also treated as spectators.

    Two classical registers are added, each with one bit per spectator qubit: ``spectator_creg_name``
    (default ``"spec"``) holds the first measurement and ``spectator_creg_name + post_check_suffix``
    (default ``"spec_ps"``) holds the second, and a shot is kept when the two disagree.

    .. note::

      These passes are only supported on Heron QPUs where `fractional gates <http://quantum.cloud.ibm.com/docs/guides/fractional-gates>`__ are supported.
    """

    def __init__(
        self,
        coupling_map: CouplingMap | list[tuple[int, int]],
        x_pulse_type: Literal["xslow", "rx"] | XPulseType = XPulseType.XSLOW,  # type: ignore
        *,
        include_unmeasured: bool = True,
        spectator_creg_name: str = DEFAULT_SPECTATOR_CREG_NAME,
        ignore_creg_suffixes: list[str] | None = None,
        post_check_suffix: str = DEFAULT_POST_CHECK_SUFFIX,
    ):
        """Initialize the pass.

        Args:
            coupling_map: A coupling map or a list of tuples indicating pairs of neighboring qubits.
            x_pulse_type: The type of X-pulse to apply between the two spectator measurements. Either ``"xslow"`` or ``"rx"``.
            include_unmeasured: Whether qubits that are active but not terminated should also be
                treated as spectators. If ``True``, the parity check is added to each of them as well.
            spectator_creg_name: The name of the classical register holding the first spectator
                measurement. The post-check register is named
                ``spectator_creg_name + post_check_suffix``.
            ignore_creg_suffixes: A list of suffixes for classical registers that should be ignored
                when determining active/terminated qubits. By default, registers ending with
                ``"_pre"`` are ignored so that pre-check measurements aren't treated as regular
                terminations.
            post_check_suffix: The suffix appended to ``spectator_creg_name`` to form the
                post-check register name, and used to identify the data-qubit post-check
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
        self.post_check_suffix = post_check_suffix

        # Same pulse sequence as ``AddPostCircuitBitFlipChecks``: one full pi rotation.
        if self.x_pulse_type == XPulseType.XSLOW:
            self.pulse_sequence = [XSlowGate()]
        else:
            self.pulse_sequence = [RXGate(np.pi / RX_PULSE_COUNT)] * RX_PULSE_COUNT

    def run(self, dag: DAGCircuit):  # noqa: D102
        # Coupling-map node ``i`` maps to register qubit ``i``, so the circuit must span the
        # full coupling graph (e.g. be laid out on the backend), not just the active qubits.
        if self.coupling_map.size() > dag.num_qubits():
            raise TranspilerError(
                f"Circuit has {dag.num_qubits()} qubits but the coupling map spans "
                f"{self.coupling_map.size()}; run this pass on a circuit laid out over the "
                f"full coupling graph."
            )
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
            num_spectators, self.spectator_creg_name + self.post_check_suffix
        )

        # Data qubits already carrying a post-check measurement (from
        # ``AddPostCircuitBitFlipChecks``); when present we splice into its existing
        # barrier/pulse/barrier sandwich instead of building our own.
        data_with_postsel: set[Qubit] = set()
        for node in dag.topological_op_nodes():
            if node.op.name == "measure" and len(node.cargs) == 1:
                clbit = node.cargs[0]
                for creg in dag.cregs.values():
                    if clbit in creg and creg.name.endswith(self.post_check_suffix):
                        data_with_postsel.add(node.qargs[0])
                        break

        # The post-sel barriers are the pair acting exactly on ``data_with_postsel``; take the
        # last two in topological order (earlier coincidental matches, e.g. a pre-check barrier
        # on the same qubit set, are intentionally ignored).
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
        """Append the spectator parity check when no data-qubit post-check is present."""
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

        # Second spectator measurement (post-check check).
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
        emitted by :class:`.AddPostCircuitBitFlipChecks` on the data qubits only),
        are extended to cover the spectator qubits as well; together they
        sandwich the pi-rotation pulses on both data and spec qubits.

        Data qubits whose terminal measurement is buried inside a control-flow
        op (or that have no spec neighbour) are left untouched: their
        measurement happens whenever it is naturally scheduled.
        """
        all_topo_nodes = list(dag.topological_op_nodes())
        barrier1_idx = next(i for i, n in enumerate(all_topo_nodes) if n._node_id == barrier1_id)

        # Pair each spec qubit with one data neighbour (smallest qubit index, deterministic).
        # A spec whose only data neighbour has its terminal measure buried in control flow stays
        # unpaired and falls through to the extended ``barrier1`` for sync.
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

        # Defer only the data terminal measures paired with a spec qubit; unpaired data qubits
        # keep their measurement in its natural position.
        data_terminal_nodes: dict[Qubit, DAGOpNode] = {
            q: data_terminal_nodes_full[q] for q in data_to_specs
        }
        deferred_terminal_node_ids = {n._node_id for n in data_terminal_nodes.values()}

        # Spectator-only ops after the first post-sel barrier are logically pre-check ops on the
        # spec wires (e.g. ``measure -> reset`` from ``AddSpectatorPreCircuitBitFlipChecks``);
        # defer them so they emerge before the spec parity check.
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
                # Flush deferred spec-only ops before the spec parity check.
                for spec_node in late_spec_nodes:
                    new_dag.apply_operation_back(spec_node.op, spec_node.qargs, spec_node.cargs)
                # For each (data qubit, paired specs) bundle: a small barrier on (data + specs),
                # then the data terminal measure, then each spec's first measure — synchronises
                # just the bundle without idling any other data qubit.
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
                # Spec qubits with no paired data neighbour still need their first measure here.
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
        # Pre-check-only qubits measure only into ignored (pre-sel) registers. Their standard
        # gates are pre-check pulses (e.g. the trailing X in ``xslow -> X -> measure(_pre)``) and
        # must NOT count as "active": otherwise a spectator pre-check qubit would be misclassified
        # as a data qubit here, and *its* neighbours would in turn be picked up as post-sel spectators.
        pre_select_only_qubits = self._find_pre_select_only_qubits(dag)

        # The qubits that undergo any non-barrier action
        active_qubits: set[Qubit] = set()

        # The qubits whose last action is a measurement
        terminated_qubits: set[Qubit] = set()

        for node in dag.topological_op_nodes():
            validate_op_is_supported(node)

            # Skip xslow, rx, and reset gates - they are part of pre/post-check protocol
            if ("xslow" in node.op.name) or ("rx" in node.op.name) or (node.op.name == "reset"):
                continue
            elif node.is_standard_gate() or node.op.name == "delay":
                # Drop qargs that participate only in pre-check: their standard gates are
                # protocol pulses (the trailing X), not main-circuit activity.
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
        """Qubits whose only measurements go to ignored (pre-check) registers.

        Recurses through control flow. Measurements buried inside control-flow
        blocks are treated as primary measurements — pre-check
        measurements are emitted at the top level by the pre-check passes,
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
