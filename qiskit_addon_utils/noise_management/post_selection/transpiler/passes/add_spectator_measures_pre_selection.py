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
"""Transpiler pass to add pre-selection measurements on spectator qubits."""

from __future__ import annotations

from copy import deepcopy
from enum import Enum

import numpy as np
from qiskit.circuit import ClassicalRegister, ControlFlowOp, Qubit, Reset
from qiskit.circuit.library import Barrier, Measure, RXGate, XGate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError

from ....constants import DEFAULT_SPECTATOR_PRE_CREG_NAME
from ....post_selection.transpiler.passes.utils import validate_op_is_supported
from ....post_selection.transpiler.passes.xslow_gate import XSlowGate


class XPulseType(str, Enum):
    """The type of X-pulse to apply for the pre-selection measurements."""

    XSLOW = "xslow"
    """An ``xslow`` gate."""

    RX = "rx"
    """Twenty ``rx`` gates with angles ``pi/20``."""


class AddSpectatorMeasuresPreSelection(TransformationPass):
    """Add pre-selection measurements on spectator qubits.

    An **active qubit** is a qubit acted on in the circuit by a non-barrier instruction. A **terminated qubit**
    is one whose last action is a measurement. A **spectator qubit** is a qubit that is inactive, but adjacent
    to an active qubit under the coupling map.

    This pass adds a pre-selection measurement to all spectator qubits and,
    optionally via ``include_unmeasured``, to all active qubits that are not terminated qubits.


    The added measurements write to a new classical register with one bit per spectator qubit and name
    ``spectator_creg_name`` (default: ``"spectator_pre"``).

    .. note::
        This pass is designed to work in conjunction with :class:`.AddPreSelectionMeasures`. Typically,
        you would use both passes together to add pre-selection measurements on both active and spectator qubits.

    Example:
        .. code-block:: python

            from qiskit import QuantumCircuit
            from qiskit.transpiler import PassManager, CouplingMap
            from qiskit_addon_utils.noise_management.post_selection.transpiler.passes import (
                AddPreSelectionMeasures,
                AddSpectatorMeasuresPreSelection,
            )

            # Create a circuit that uses qubits 0, 1, 2
            qc = QuantumCircuit(5, 3)  # 5 qubits total, 3 classical bits
            qc.h(0)
            qc.cx(0, 1)
            qc.cx(1, 2)
            qc.measure([0, 1, 2], [0, 1, 2])

            # Define coupling map (qubits 3 and 4 are spectators adjacent to active qubits)
            coupling_map = CouplingMap([(0, 1), (1, 2), (2, 3), (1, 4)])

            # Add pre-selection measurements on both active and spectator qubits
            pm = PassManager([
                AddPreSelectionMeasures(coupling_map),
                AddSpectatorMeasuresPreSelection(coupling_map),
            ])
            qc_with_pre = pm.run(qc)

            # The resulting circuit will have:
            # 1. Pre-selection measurements on active qubits 0, 1, 2 (to c_pre register)
            # 2. Pre-selection measurements on spectator qubits 3, 4 (to spectator_pre register)
            # 3. A barrier
            # 4. The original circuit operations
    """

    def __init__(
        self,
        coupling_map: CouplingMap | list[tuple[int, int]],
        x_pulse_type: str | XPulseType = XPulseType.XSLOW,  # type: ignore
        *,
        include_unmeasured: bool = True,
        spectator_creg_name: str = DEFAULT_SPECTATOR_PRE_CREG_NAME,
        add_barrier: bool = True,
        ignore_spectator_creg_names: list[str] | None = None,
        ignore_creg_suffixes: list[str] | None = None,
        pre_selection_suffix: str = "_pre",
    ):
        """Initialize the pass.

        Args:
            coupling_map: A coupling map or a list of tuples indicating pairs of neighboring qubits.
            x_pulse_type: The type of X-pulse to apply for the pre-selection measurements.
            include_unmeasured: Whether the qubits that are active but are not terminated by a measurement should
                also be treated as spectators. If ``True``, a terminal measurement is added on each of them.
            spectator_creg_name: The name of the classical register added for the measurements on the spectator qubits.
            add_barrier: Whether to add a barrier acting on all active and spectator qubits prior to the spectator
                measurements.
            ignore_spectator_creg_names: List of classical register names to ignore when determining active qubits.
                Qubits that only have measurements to these registers are not considered active, preventing cascading
                spectator selection. Defaults to ``["spec"]`` (the default name used by :class:`.AddSpectatorMeasures`).
            ignore_creg_suffixes: A list of suffixes for classical registers that should be ignored when determining
                terminated qubits. Qubits with measurements into registers with these suffixes are not considered
                terminated, allowing pre-selection measurements to be added. By default, registers ending with "_ps"
                are ignored to allow pre-selection after post-selection.
            pre_selection_suffix: The suffix used by AddPreSelectionMeasures for pre-selection registers. This is used
                to identify which qubits have pre-selection measurements and which barrier to extend. Defaults to "_pre".
        """
        super().__init__()
        self.x_pulse_type = XPulseType(x_pulse_type)
        self.spectator_creg_name = spectator_creg_name
        self.include_unmeasured = include_unmeasured
        self.pre_selection_suffix = pre_selection_suffix
        self.ignore_spectator_creg_names = (
            ignore_spectator_creg_names if ignore_spectator_creg_names is not None else ["spec"]
        )
        self.ignore_creg_suffixes = (
            ignore_creg_suffixes if ignore_creg_suffixes is not None else ["_ps"]
        )
        self.coupling_map = (
            deepcopy(coupling_map)
            if isinstance(coupling_map, CouplingMap)
            else CouplingMap(couplinglist=coupling_map)
        )
        self.coupling_map.make_symmetric()
        self.add_barrier = add_barrier

        # Pre-selection sequence: xslow (or rx pulses) + X gate
        if self.x_pulse_type == XPulseType.XSLOW:
            self.pulse_sequence = [XSlowGate(), XGate()]
        else:
            self.pulse_sequence = [RXGate(np.pi / 20)] * 20 + [XGate()]

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

        # sort the spectator qubits, so that qubit `i` writes to clbit `i`
        spectator_qubits_ls = list(spectator_qubits)
        spectator_qubits_ls.sort(key=lambda qubit: qubit_map[qubit])

        # Create a new DAG to build the circuit with pre-selection at the front
        new_dag = DAGCircuit()
        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)

        # Add the new spectator register (or use existing one if it already exists)
        if self.spectator_creg_name in dag.cregs:
            new_reg = dag.cregs[self.spectator_creg_name]
            if new_reg.size != num_spectators:
                raise TranspilerError(
                    f"Classical register '{self.spectator_creg_name}' already exists with size "
                    f"{new_reg.size}, but {num_spectators} spectator qubits were found."
                )
        else:
            new_dag.add_creg(new_reg := ClassicalRegister(num_spectators, self.spectator_creg_name))

        # Find all qubits that have pre-selection measurements (from AddPreSelectionMeasures)
        # These are qubits that have measurements into registers ending with the pre_selection_suffix
        data_qubits_with_preselection = set()
        for node in dag.topological_op_nodes():
            if node.op.name == "measure" and len(node.cargs) == 1:
                clbit = node.cargs[0]
                for creg in dag.cregs.values():
                    if clbit in creg and creg.name.endswith(self.pre_selection_suffix):
                        data_qubits_with_preselection.add(node.qargs[0])
                        break

        # Combine data qubits with pre-selection and spectator qubits for unified barrier
        all_preselection_qubits = list(data_qubits_with_preselection.union(spectator_qubits_ls))
        all_preselection_qubits.sort(key=lambda qubit: qubit_map[qubit])

        # Copy all operations from the original DAG to the new DAG
        # When we encounter the FIRST barrier from AddPreSelectionMeasures, extend it and add spectator measurements after
        barrier_extended = False
        for node in dag.topological_op_nodes():
            if (
                not barrier_extended
                and node.op.name == "barrier"
                and set(node.qargs) == data_qubits_with_preselection
                and self.add_barrier
                and len(data_qubits_with_preselection) > 0
            ):
                # Replace with full-width barrier that includes spectators
                new_dag.apply_operation_back(
                    Barrier(len(all_preselection_qubits)), all_preselection_qubits
                )
                # Add spectator measurements and resets immediately after the barrier
                # Note: Spectators only get measurement + reset, NOT the pulse sequence
                for qubit, clbit in zip(spectator_qubits_ls, new_reg):
                    new_dag.apply_operation_back(Measure(), [qubit], [clbit])
                    # Add reset after measurement in case post-selection measures are added later
                    new_dag.apply_operation_back(Reset(), [qubit])
                barrier_extended = True
            else:
                new_dag.apply_operation_back(node.op, node.qargs, node.cargs)
        return new_dag

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

            # Skip xslow and rx gates - they are part of pre/post-selection protocol
            if ("xslow" in node.op.name) or ("rx" in node.op.name):
                continue
            elif node.is_standard_gate():
                active_qubits.update(node.qargs)
                terminated_qubits.difference_update(node.qargs)
            elif (name := node.op.name) == "barrier":
                continue
            elif name == "measure":
                # Check if this is a measurement into an ignored spectator register or ignored suffix register
                if len(node.cargs) == 1:
                    clbit = node.cargs[0]
                    is_ignored_measurement = False
                    for creg in dag.cregs.values():
                        # Check if measuring into an ignored spectator register or register with ignored suffix
                        if clbit in creg and (
                            creg.name in self.ignore_spectator_creg_names
                            or any(
                                creg.name.endswith(suffix) for suffix in self.ignore_creg_suffixes
                            )
                        ):
                            is_ignored_measurement = True
                            break

                    # Only add to active qubits if NOT measuring into an ignored register
                    if not is_ignored_measurement:
                        active_qubits.add(node.qargs[0])
                        terminated_qubits.add(node.qargs[0])
                    # If it IS an ignored measurement, don't mark as terminated
                    # (so pre-selection can be added)
                else:
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
            elif "xslow" in node.op.name:
                # xslow gates (from pre/post-selection) don't make a qubit "active"
                # They are just part of the measurement protocol
                continue
            else:  # pragma: no cover
                raise TranspilerError(f"``'{node.op.name}'`` is not supported.")

        return active_qubits, terminated_qubits


# Made with Bob
