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
    """Add a pre selection measurement at the beginning of the circuit.

    A pre selection measurement is a measurement that precedes the main circuit operations. It
    consists of a narrowband X-pulse (xslow), followed by an X gate, followed by a measurement.
    In the absence of noise, it is expected to return ``1``. Shots where the pre-selection
    measurement returns ``0`` indicate that the qubit was not properly initialized and should
    be discarded.

    This pass adds pre selection measurements at the beginning of the circuit for all active qubits
    and optionally spectator qubits (qubits adjacent to active qubits under the coupling map).
    The added measurements write to new classical registers that are copies of the DAG's registers,
    with modified names.

    .. note::
        When this pass encounters a control flow operation, it iterates through all of its blocks to
        determine which qubits are active in the circuit.
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
        qubit_to_clbit_map = {}
        for node in dag.topological_op_nodes():
            if node.op.name == "measure" and len(node.qargs) == 1 and len(node.cargs) == 1:
                qubit_to_clbit_map[node.qargs[0]] = node.cargs[0]

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
        # Iterate through measurements in the order they appear in the original circuit
        # to preserve the qubit-to-clbit mapping
        qubits_list = []
        for node in dag.topological_op_nodes():
            if node.op.name == "measure" and len(node.qargs) == 1:
                qubit = node.qargs[0]
                if qubit in qubits_to_preselect:
                    qubits_list.append(qubit)
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

# Made with Bob
