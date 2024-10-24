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

"""A method for slicing quantum circuits by depth."""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit


def slice_by_depth(circuit: QuantumCircuit, max_slice_depth: int) -> list[QuantumCircuit]:
    """Split a ``QuantumCircuit`` into slices based on depth.

    This function transforms the input circuit into a :class:`~qiskit.dagcircuit.DAGCircuit` and
    batches the sequence of depth-1 layers output from :meth:`~qiskit.dagcircuit.DAGCircuit.layers`
    into slices of depth not exceeding ``max_slice_depth``. This is achieved by composing
    layers into slices until the max slice depth is reached and then starting
    a new slice with the next layer. The final slice may be composed of fewer than
    ``max_slice_depth`` layers.

    Args:
        circuit: The circuit to be split.
        max_slice_depth: The maximum depth of a given slice.

    Returns:
        A sequence of :class:`~qiskit.circuit.QuantumCircuit` objects, one for each slice.
    """
    if max_slice_depth <= 0:
        raise ValueError("max_slice_depth must be > 0.")

    # Get list of depth-1 layers
    dag_circuit = circuit_to_dag(circuit)
    layers = [layer["graph"] for layer in dag_circuit.layers()]

    # Collect lists of layers of length == max_slice_depth. The final list may
    # have fewer layers.
    slice_layers = [layers[i : i + max_slice_depth] for i in range(0, len(layers), max_slice_depth)]

    # Create output slices
    slices = []
    for slice_ in slice_layers:
        s = slice_[0]
        for i in range(1, len(slice_)):
            s.compose(slice_[i], inplace=True)
        slices.append(dag_to_circuit(s))

    return slices
