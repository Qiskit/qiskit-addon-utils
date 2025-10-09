# This code is a Qiskit project.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A method for slicing quantum circuits by gate types."""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager

from .transpiler.passes import CollectOpType


def slice_by_gate_types(circuit: QuantumCircuit) -> list[QuantumCircuit]:
    """Split a ``QuantumCircuit`` into depth-1 slices of operations of the same type.

    .. warning::
       Note: Adjacent slices sharing no qubits in common may be ordered arbitrarily.

    Args:
        circuit: The circuit to be split.

    Returns:
        A sequence of :class:`~qiskit.circuit.QuantumCircuit` objects, one for each slice.
    """
    dag = circuit_to_dag(circuit)
    # We leverage DAGCircuit.count_ops to find the set of operation types that exist in our circuit.
    # This benefits from an internal cache of operation names constructed inside the DAG circuit.
    op_types = set(dag.count_ops(recurse=False).keys())

    passes = []
    for op_type in sorted(op_types):
        passes.append(CollectOpType(op_type))

    pass_manager = PassManager(passes)
    sliced_circuit = pass_manager.run(circuit)
    slices = []
    for op_node in circuit_to_dag(sliced_circuit).op_nodes():
        # NOTE: due to our transpiler pass above, we know that each op_node in this circuit will be
        # an Instruction instance representing a slice
        qargs = [circuit.find_bit(q).index for q in op_node.qargs]
        cargs = [circuit.find_bit(c).index for c in op_node.cargs]
        qc = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
        qc.append(op_node.op, qargs, cargs)
        slices.append(qc.decompose())

    return slices
