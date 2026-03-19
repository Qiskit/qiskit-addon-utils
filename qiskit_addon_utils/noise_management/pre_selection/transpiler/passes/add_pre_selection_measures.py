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
    the qubit starts in |0⟩, gets flipped to |1⟩ by the two X gate applications, then measured). Shots where the
    pre-selection measurement returns ``1`` indicate that the qubit was not properly initialized
    to |0⟩ and should be discarded.

    This pass adds pre-selection measurements at the beginning of the circuit for all qubits that:
    1. Are active in the circuit (have gates applied to them)
    2. Have terminal measurements

    The added measurements write to new classical registers that are copies of the DAG's registers,
    with modified names (by default, appending ``"_pre"`` to the register name).

    The pre-selection protocol works as follows:

    1. **xslow pulse (or rx sequence)**: A narrowband X-pulse that slowly rotates the qubit.
       This can be either a single ``xslow`` gate or 20 ``rx(π/20)`` gates.
    2. **X gate**: A standard X gate to complete the flip from |0⟩ to |1⟩.
    3. **Measurement**: Measures the qubit state. Should return ``0`` if initialization was good.
    4. **Barrier**: Separates pre-selection measurements from the main circuit.
    5. **Main circuit**: The original circuit operations proceed.

    Example:
        .. code-block:: python

            from qiskit import QuantumCircuit
            from qiskit.transpiler import PassManager
            from qiskit_addon_utils.noise_management.pre_selection.transpiler.passes import (
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
    ):
        """Initialize the pass.

        Args:
            coupling_map: A coupling map or a list of tuples indicating pairs of neighboring qubits.
            x_pulse_type: The type of X-pulse to apply for the pre-selection measurements.
            pre_selection_suffix: A fixed suffix to append to the names of the classical registers when copying them.
        """
        super().__init__()
        self.x_pulse_type = XPulseType(x_pulse_type)
        self.pre_selection_suffix = pre_selection_suffix
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
        clbits_map = {}
        for name, creg in dag.cregs.items():
            # Create a pre-selection register with the same size as the original
            dag.add_creg(
                new_creg := ClassicalRegister(creg.size, name + self.pre_selection_suffix)
            )
            # Map existing clbits to the new register
            clbits_map.update({clbit: new_clbit for clbit, new_clbit in zip(creg, new_creg)})

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
        
        for qubit in qubits_list:
            for gate in self.pulse_sequence:
                new_dag.apply_operation_back(gate, [qubit])
            # Measure to the corresponding clbit in the pre-selection registers
            clbit = qubit_to_clbit_map[qubit]
            new_dag.apply_operation_back(Measure(), [qubit], [clbits_map[clbit]])

        # Add a barrier to separate the pre-selection measurements from the rest of the circuit
        if qubits_list:
            new_dag.apply_operation_back(Barrier(len(qubits_list)), qubits_list)

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

            if node.is_standard_gate():
                active_qubits.update(node.qargs)
            elif (name := node.op.name) == "xslow":
                continue
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

    def _find_measurements(self, dag: DAGCircuit) -> dict[Qubit, any]:
        """Helper function to find all measurements in the circuit, including those in control flow.

        This function returns a map from qubits to the classical bits they measure into.
        It recursively searches through control flow operations to find measurements.

        Args:
            dag: The dag to iterate over.
        """
        qubit_to_clbit_map = {}

        for node in dag.topological_op_nodes():
            if node.op.name == "measure" and len(node.qargs) == 1 and len(node.cargs) == 1:
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
                    
                    # Find measurements in the block and map them back to parent circuit
                    block_measurements = self._find_measurements(block_dag)
                    for block_qubit, block_clbit in block_measurements.items():
                        if block_qubit in qubit_map and block_clbit in clbit_map:
                            qubit_to_clbit_map[qubit_map[block_qubit]] = clbit_map[block_clbit]

        return qubit_to_clbit_map

# Made with Bob
