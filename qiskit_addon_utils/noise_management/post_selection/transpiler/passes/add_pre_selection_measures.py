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
"""Transpiler pass to add pre selection measurements."""

from __future__ import annotations

from copy import deepcopy
from enum import Enum
from typing import Any

import numpy as np
from qiskit.circuit import ClassicalRegister, ControlFlowOp, Qubit
from qiskit.circuit.library import Barrier, Measure, RXGate, XGate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError

from ....constants import DEFAULT_PRE_SELECTION_SUFFIX
from ....post_selection.transpiler.passes.utils import validate_op_is_supported
from ....post_selection.transpiler.passes.xslow_gate import XSlowGate


class XPulseType(str, Enum):
    """The type of X-pulse to apply for the pre-selection measurements."""

    XSLOW = "xslow"
    """An ``xslow`` gate."""

    RX = "rx"
    """Twenty ``rx`` gates with angles ``pi/20``."""


class AddPreSelectionMeasures(TransformationPass):
    """Add a pre-selection measurement at the beginning of the circuit.

    A pre-selection measurement is a measurement that precedes the main circuit operations. It
    consists of a narrowband X-pulse (e.g. a sequence of N rx(pi/N) gates), followed by an X gate,
    followed by a measurement. In the absence of noise, it is expected to return ``0`` (since
    the qubit starts in the ground state, gets flipped to the excited state by the two X gate
    applications, then measured). Shots where the pre-selection measurement returns ``1`` indicate
    that the qubit was not properly initialized to the ground state and should be discarded.

    This pass adds pre-selection measurements at the beginning of the circuit for all qubits that:
    1. Are active in the circuit (have gates applied to them)
    2. Have terminal measurements

    The added measurements write to new classical registers that are copies of the DAG's registers,
    with modified names (by default, appending ``"_pre"`` to the register name).

    The pre-selection protocol works as follows:

    1. **xslow pulse (or rx sequence)**: A narrowband X-pulse that slowly rotates the qubit.
       This can be either a single ``xslow`` gate or 20 ``rx(π/20)`` gates.
    2. **X gate**: A standard X gate to complete the flip from ground to excited state.
    3. **Measurement**: Measures the qubit state. Should return ``0`` if initialization was good.
    4. **Barrier**: Separates pre-selection measurements from the main circuit.
    5. **Main circuit**: The original circuit operations proceed.

    Example:
        .. code-block:: python

            from qiskit import QuantumCircuit
            from qiskit.transpiler import PassManager
            from qiskit_addon_utils.noise_management.post_selection.transpiler.passes import (
                AddPreSelectionMeasures,
            )

            # Create a simple circuit
            qc = QuantumCircuit(2, 2)
            qc.h(0)
            qc.cx(0, 1)
            qc.measure([0, 1], [0, 1])

            # Add pre-selection measurements
            coupling_map = [(0, 1)]
            pm = PassManager([AddPreSelectionMeasures(coupling_map)])
            qc_with_pre = pm.run(qc)

            # The resulting circuit will have:
            # 1. Pre-selection measurements at the start (xslow + X + measure to c_pre)
            # 2. A barrier
            # 3. The original circuit (H, CX, measure to c)
    """

    def __init__(
        self,
        coupling_map: CouplingMap | list[tuple[int, int]],
        x_pulse_type: str | XPulseType = XPulseType.XSLOW,  # type: ignore
        *,
        pre_selection_suffix: str = DEFAULT_PRE_SELECTION_SUFFIX,
        ignore_creg_suffixes: list[str] | None = None,
        ignore_creg_names: list[str] | None = None,
    ):
        """Initialize the pass.

        Args:
            coupling_map: A coupling map or a list of tuples indicating pairs of neighboring qubits.
            x_pulse_type: The type of X-pulse to apply for the pre-selection measurements.
            pre_selection_suffix: A fixed suffix to append to the names of the classical registers when copying them.
            ignore_creg_suffixes: A list of suffixes for classical registers that should be ignored (not copied).
                By default, registers ending with "_ps" are ignored to avoid adding pre-selection to post-selection registers.
            ignore_creg_names: A list of exact classical register names that should be ignored (not copied).
                By default, registers named "spec" are ignored to avoid adding pre-selection to spectator registers.
        """
        super().__init__()
        self.x_pulse_type = XPulseType(x_pulse_type)
        self.pre_selection_suffix = pre_selection_suffix
        self.ignore_creg_suffixes = (
            ignore_creg_suffixes if ignore_creg_suffixes is not None else ["_ps"]
        )
        self.ignore_creg_names = ignore_creg_names if ignore_creg_names is not None else ["spec"]
        self.coupling_map = (
            deepcopy(coupling_map)
            if isinstance(coupling_map, CouplingMap)
            else CouplingMap(couplinglist=coupling_map)
        )
        self.coupling_map.make_symmetric()

        # Pre-selection sequence: xslow (or rx pulses) + X gate
        if self.x_pulse_type == XPulseType.XSLOW:
            self.pulse_sequence = [XSlowGate(), XGate()]
        else:
            self.pulse_sequence = [RXGate(np.pi / 20)] * 20 + [XGate()]

    def run(self, dag: DAGCircuit):  # noqa: D102
        # Find what qubits are active in the circuit
        active_qubits = self._find_active_qubits(dag)

        if not active_qubits:
            return dag

        # Find which classical bit each qubit measures into by scanning the circuit
        # This needs to handle measurements inside control flow operations (boxes, if/else, etc.)
        qubit_to_clbit_map = self._find_measurements(dag)

        # Only pre-select qubits that have measurements
        qubits_to_preselect = set(qubit_to_clbit_map.keys()) & active_qubits

        if not qubits_to_preselect:
            return dag

        # Add the new registers and create a map between the original clbit and the new ones
        # Skip registers with ignored suffixes or names (e.g., post-selection or spectator registers)
        clbits_map = {}
        for name, creg in dag.cregs.items():
            if any(name.endswith(suffix) for suffix in self.ignore_creg_suffixes):
                # Skip registers with ignored suffixes
                continue
            if name in self.ignore_creg_names:
                # Skip registers with ignored names
                continue
            # Create a pre-selection register with the same size as the original
            dag.add_creg(new_creg := ClassicalRegister(creg.size, name + self.pre_selection_suffix))
            # Map existing clbits to the new register
            clbits_map.update({clbit: new_clbit for clbit, new_clbit in zip(creg, new_creg)})

        # Filter qubits to only include those that measure into non-ignored registers
        qubits_to_preselect = {
            qubit for qubit in qubits_to_preselect if qubit_to_clbit_map[qubit] in clbits_map
        }

        if not qubits_to_preselect:
            return dag

        # Create a new DAG to build the pre-selection circuit
        new_dag = DAGCircuit()
        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)

        # Add the pre-selection measurements at the front
        # We need to add them in a consistent order based on the qubit-to-clbit mapping
        # Sort by clbit index to ensure consistent ordering
        qubits_list = sorted(qubits_to_preselect, key=lambda q: qubit_to_clbit_map[q]._index)

        # Apply all pulse sequences first
        for qubit in qubits_list:
            for gate in self.pulse_sequence:
                new_dag.apply_operation_back(gate, [qubit])

        # Add barrier before measurements - AddSpectatorMeasuresPreSelection will extend it.
        # ``qubits_list`` is non-empty: we returned early above when ``qubits_to_preselect``
        # was empty, so this guard is purely defensive.
        if qubits_list:  # pragma: no branch
            new_dag.apply_operation_back(Barrier(len(qubits_list)), qubits_list)

        # Then add all measurements
        for qubit in qubits_list:
            clbit = qubit_to_clbit_map[qubit]
            new_dag.apply_operation_back(Measure(), [qubit], [clbits_map[clbit]])
            # Note: No reset on data qubits - they continue to the main circuit

        # Copy all operations from the original DAG to the new DAG
        for node in dag.topological_op_nodes():
            # do this to preserve meas ordering
            # if node.op.name == "measure" and len(node.qargs) == 1:
            #     qubit = node.qargs[0]
            #     clbit = qubit_to_clbit_map[qubit]
            #     new_dag.apply_operation_back(Measure(), [qubit], [clbit])
            # else:
            new_dag.apply_operation_back(node.op, node.qargs, node.cargs)

        return new_dag

    def _find_active_qubits(self, dag: DAGCircuit) -> set[Qubit]:
        """Helper function to find the active qubits.

        This function returns a set of qubits that are acted upon by any non-barrier instruction.
        It is used recursively for control flow operations.

        Args:
            dag: The dag to iterate over.
        """
        active_qubits: set[Qubit] = set()

        for node in dag.topological_op_nodes():
            validate_op_is_supported(node)

            # Skip xslow, rx, and reset gates - they are part of pre/post-selection protocol
            if ("xslow" in node.op.name) or ("rx" in node.op.name) or (node.op.name == "reset"):
                continue
            elif node.is_standard_gate():
                active_qubits.update(node.qargs)
            elif (name := node.op.name) == "barrier":
                continue
            elif name == "measure":
                active_qubits.add(node.qargs[0])
            elif isinstance(node.op, ControlFlowOp):
                for block in node.op.blocks:
                    block_dag = circuit_to_dag(block)
                    qubit_map = {
                        block_qubit: qubit
                        for block_qubit, qubit in zip(block_dag.qubits, node.qargs)
                    }
                    block_active_qubits = self._find_active_qubits(block_dag)
                    active_qubits.update({qubit_map[qubit] for qubit in block_active_qubits})
            else:  # pragma: no cover
                raise TranspilerError(f"``'{node.op.name}'`` is not supported.")

        return active_qubits

    def _find_measurements(self, dag: DAGCircuit) -> dict[Qubit, Any]:
        """Helper function to find measurements in the circuit to non-ignored registers.

        This function returns a map from qubits to the classical bits they measure into,
        but only for measurements into registers that are not in the ignore list.
        It recursively searches through control flow operations to find measurements.

        Args:
            dag: The dag to iterate over.
        """
        qubit_to_clbit_map = {}

        for node in dag.topological_op_nodes():
            if node.op.name == "measure" and len(node.qargs) == 1 and len(node.cargs) == 1:
                # Check if this measurement is to an ignored register
                clbit = node.cargs[0]
                is_ignored = False
                for creg in dag.cregs.values():
                    if clbit in creg and any(
                        creg.name.endswith(suffix) for suffix in self.ignore_creg_suffixes
                    ):
                        is_ignored = True
                        break

                # Only record measurements to non-ignored registers
                if not is_ignored:
                    qubit_to_clbit_map[node.qargs[0]] = node.cargs[0]
            elif isinstance(node.op, ControlFlowOp):
                # Recursively search for measurements in control flow blocks
                for block in node.op.blocks:
                    block_dag = circuit_to_dag(block)

                    # Create mappings from block qubits/clbits to parent qubits/clbits
                    qubit_map = {
                        block_qubit: qubit
                        for block_qubit, qubit in zip(block_dag.qubits, node.qargs)
                    }
                    clbit_map = {
                        block_clbit: clbit
                        for block_clbit, clbit in zip(block_dag.clbits, node.cargs)
                    }

                    # Find measurements in the block and map them back to parent circuit.
                    # The ``in`` checks are defensive: every block qubit/clbit produced by
                    # ``_find_measurements`` should already appear in the maps we built
                    # from ``node.qargs`` / ``node.cargs``.
                    block_measurements = self._find_measurements(block_dag)
                    for block_qubit, block_clbit in block_measurements.items():
                        if block_qubit in qubit_map and block_clbit in clbit_map:  # pragma: no branch
                            qubit_to_clbit_map[qubit_map[block_qubit]] = clbit_map[block_clbit]

        return qubit_to_clbit_map


# Made with Bob
