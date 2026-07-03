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

"""A method for slicing quantum circuits by barriers."""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag


def slice_by_barriers(circuit: QuantumCircuit) -> list[QuantumCircuit]:
    """Split a ``QuantumCircuit`` into slices around full-circuit barriers.

    Barriers which do not act on all circuit qubits will be treated as
    normal operations and included in the slices. Barriers which act on
    all qubits will be interpreted as slice locations and will not be
    included in the output slices.

    Args:
        circuit: The circuit to be split.

    Returns:
        A sequence of :class:`~qiskit.circuit.QuantumCircuit` objects, one for each slice.
    """
    slices = []
    current_slice: QuantumCircuit | None = None
    for op_node in circuit_to_dag(circuit).op_nodes(include_directives=True):
        qargs = [circuit.find_bit(q).index for q in op_node.qargs]
        if op_node.name == "barrier" and len(qargs) == circuit.num_qubits:
            slices.append(current_slice)
            current_slice = None
            continue

        if current_slice is None:
            current_slice = QuantumCircuit(circuit.num_qubits)

        current_slice.append(op_node.op, qargs=qargs)

    if current_slice is not None:
        slices.append(current_slice)

    return slices
