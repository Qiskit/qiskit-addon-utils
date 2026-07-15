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
"""Transpiler pass to add pre-circuit bit-flip checks."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from qiskit.circuit import ClassicalRegister, ControlFlowOp, Qubit
from qiskit.circuit.library import Barrier, Measure, RXGate, XGate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError

from ...constants import DEFAULT_PRE_CHECK_SUFFIX
from ..xslow_gate import XSlowGate
from ._utils import validate_op_is_supported
from .x_pulse_type import XPulseType


class AddPreCircuitBitFlipChecks(TransformationPass):
    r"""Add bit-flip checks at the beginning of the circuit on active qubits terminated by a measurement.

    A pre-circuit bit-flip check consists of a narrowband X-pulse that rotates the qubit from
    :math:`|0\rangle\mapsto|1\rangle` followed by a normal X-pulse that rotates the qubit back
    to the ground state :math:`|0\rangle` and a measurement. If the QPU fails to flip the qubit from
    :math:`|0\rangle\mapsto|1\rangle\mapsto|0\rangle` on a given shot, that sample may be considered
    unreliable and discarded. Postselecting only samples that pass all checks can improve the fidelity
    of distributions sampled from the QPU.

    The added measurements write to new classical registers that are copies of the DAG's registers,
    with modified names (by default, appending ``"_pre"`` to the register name).

    .. note::

      These passes are only supported on Heron QPUs where `fractional gates <http://quantum.cloud.ibm.com/docs/guides/fractional-gates>`__ are supported.
    """

    def __init__(
        self,
        x_pulse_type: Literal["xslow", "rx"] | XPulseType = XPulseType.XSLOW,  # type: ignore
        *,
        pre_check_suffix: str = DEFAULT_PRE_CHECK_SUFFIX,
        ignore_creg_suffixes: list[str] | None = None,
        ignore_creg_names: list[str] | None = None,
    ):
        """Initialize the pass.

        Args:
            x_pulse_type: The type of X-pulse to apply for the pre-check measurements. Either ``"xslow"`` or ``"rx"``.
            pre_check_suffix: A fixed suffix to append to the names of the classical registers when copying them.
            ignore_creg_suffixes: A list of suffixes for classical registers that should be ignored (not copied).
                By default, registers ending with "_ps" are ignored to avoid adding pre-check to post-check registers.
            ignore_creg_names: A list of exact classical register names that should be ignored (not copied).
                By default, registers named "spec" are ignored to avoid adding pre-check to spectator registers.
        """
        super().__init__()
        self.x_pulse_type = XPulseType(x_pulse_type)
        self.pre_check_suffix = pre_check_suffix
        self.ignore_creg_suffixes = (
            ignore_creg_suffixes if ignore_creg_suffixes is not None else ["_ps"]
        )
        self.ignore_creg_names = ignore_creg_names if ignore_creg_names is not None else ["spec"]

        # Pre-check sequence: xslow (or rx pulses) + X gate
        if self.x_pulse_type == XPulseType.XSLOW:
            self.pulse_sequence = [XSlowGate(), XGate()]
        else:
            self.pulse_sequence = [RXGate(np.pi / 20)] * 20 + [XGate()]

    def run(self, dag: DAGCircuit):  # noqa: D102
        # Find what qubits are active in the circuit
        active_qubits = self._find_active_qubits(dag)

        if not active_qubits:
            return dag

        # Map each qubit to the clbit it measures into (handles control flow blocks).
        qubit_to_clbit_map = self._find_measurements(dag)

        # Only pre-select qubits that have measurements
        qubits_to_preselect = set(qubit_to_clbit_map.keys()) & active_qubits

        if not qubits_to_preselect:
            return dag

        # Add the new registers and map each original clbit to its pre-check copy. Skip registers
        # with ignored suffixes/names, registers whose pre-check counterpart already exists, and any
        # register already ending in the pre-check suffix -- the last check is unconditional (unlike
        # the post-check pass), since a pre-check leaves a lone ``_pre`` register with no base. This
        # keeps the pass safe to re-run and to run after ``AddSpectatorPreCircuitBitFlipChecks``.
        existing_creg_names = set(dag.cregs)
        clbits_map = {}
        for name, creg in dag.cregs.items():
            if any(name.endswith(suffix) for suffix in self.ignore_creg_suffixes):
                continue
            if name in self.ignore_creg_names:
                continue
            if name + self.pre_check_suffix in existing_creg_names:
                continue
            if name.endswith(self.pre_check_suffix):
                continue
            # Create a pre-check register with the same size as the original
            dag.add_creg(new_creg := ClassicalRegister(creg.size, name + self.pre_check_suffix))
            # Map existing clbits to the new register
            clbits_map.update({clbit: new_clbit for clbit, new_clbit in zip(creg, new_creg)})

        # Filter qubits to only include those that measure into non-ignored registers
        qubits_to_preselect = {
            qubit for qubit in qubits_to_preselect if qubit_to_clbit_map[qubit] in clbits_map
        }

        if not qubits_to_preselect:
            return dag

        # Create a new DAG to build the pre-check circuit
        new_dag = DAGCircuit()
        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)

        # Sort by clbit index so pre-check measurements are added in a consistent order.
        qubits_list = sorted(qubits_to_preselect, key=lambda q: qubit_to_clbit_map[q]._index)

        # Apply all pulse sequences first
        for qubit in qubits_list:
            for gate in self.pulse_sequence:
                new_dag.apply_operation_back(gate, [qubit])

        # Add barrier before measurements - AddSpectatorPreCircuitBitFlipChecks will extend it.
        # ``qubits_list`` is non-empty here (we returned early otherwise); the guard is defensive.
        if qubits_list:  # pragma: no branch
            new_dag.apply_operation_back(Barrier(len(qubits_list)), qubits_list)

        # Then add all measurements
        for qubit in qubits_list:
            clbit = qubit_to_clbit_map[qubit]
            new_dag.apply_operation_back(Measure(), [qubit], [clbits_map[clbit]])
            # Note: No reset on data qubits - they continue to the main circuit

        # Copy all operations from the original DAG to the new DAG
        for node in dag.topological_op_nodes():
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

            # Skip xslow, rx, and reset gates - they are part of pre/post-check protocol
            if ("xslow" in node.op.name) or ("rx" in node.op.name) or (node.op.name == "reset"):
                continue
            elif node.is_standard_gate() or node.op.name == "delay":
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

                    # Find measurements in the block and map them back to the parent circuit. The
                    # ``in`` checks are defensive: these qubits/clbits should already be in the maps.
                    block_measurements = self._find_measurements(block_dag)
                    for block_qubit, block_clbit in block_measurements.items():
                        if (
                            block_qubit in qubit_map and block_clbit in clbit_map
                        ):  # pragma: no branch
                            qubit_to_clbit_map[qubit_map[block_qubit]] = clbit_map[block_clbit]

        return qubit_to_clbit_map
