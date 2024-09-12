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

"""A method for slicing quantum circuits by coloring."""

from __future__ import annotations

from collections import defaultdict

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager

from qiskit_addon_utils.coloring import is_valid_edge_coloring

from .transpiler.passes import CollectOpColor, CollectOpSize


def slice_by_coloring(
    circuit: QuantumCircuit, coloring: dict[tuple[int, int], int]
) -> list[QuantumCircuit]:
    """Split a ``QuantumCircuit`` into slices using the provided edge coloring.

    Two-qubit gates acting on identically colored qubit connections (edges) will be grouped greedily
    into slices using :class:`.CollectOpColor`. This will be done in order of increasing color value
    (the integer values which each edge is mapped to).

    .. warning::
       Note, that this does *not* mean that low valued color slices are guaranteed to be left-most
       in your circuit. Below is an example to emphasize this.

    .. code-block:: python

       >>> from qiskit import QuantumCircuit

       >>> circuit = QuantumCircuit(5)
       >>> _ = circuit.cx(0, 1)
       >>> _ = circuit.cx(3, 4)
       >>> _ = circuit.cx(2, 3)

       >>> circuit.draw()
       q_0: ──■───────
            ┌─┴─┐
       q_1: ┤ X ├─────
            └───┘
       q_2: ───────■──
                 ┌─┴─┐
       q_3: ──■──┤ X ├
            ┌─┴─┐└───┘
       q_4: ┤ X ├─────
            └───┘

       >>> coloring = {(0, 1): 0, (2, 3): 0, (3, 4): 1}

       >>> from qiskit_addon_utils.slicing import combine_slices, slice_by_coloring

       >>> slices = slice_by_coloring(circuit, coloring)

       # for illustration purposes, we are recombining the slices with barriers
       >>> recombined = combine_slices(slices, include_barriers=True)
       >>> recombined.draw()
                  ░
       q_0: ──────░───■──
                  ░ ┌─┴─┐
       q_1: ──────░─┤ X ├
                  ░ └───┘
       q_2: ──────░───■──
                  ░ ┌─┴─┐
       q_3: ──■───░─┤ X ├
            ┌─┴─┐ ░ └───┘
       q_4: ┤ X ├─░──────
            └───┘ ░

    Single-qubit gates will be collected into a single slice using :class:`.CollectOpSize`.

    Args:
        circuit: The circuit to be split.
        coloring: A dictionary mapping edges (pairs of integers) to color values.

    Returns:
        A sequence of :class:`~qiskit.circuit.QuantumCircuit` objects, one for each slice.

    Raises:
        ValueError: The input edge coloring is invalid.
        ValueError: Could not assign a color to circuit instruction.
    """
    if not is_valid_edge_coloring(coloring):
        raise ValueError("The input coloring is invalid.")

    # we invert the coloring dictionary to obtain a mapping from color value to corresponding edges
    colors = defaultdict(set)
    for edge, color in coloring.items():
        colors[color].add(edge)

    passes = []
    passes.append(CollectOpSize(1))
    for color, edges in sorted(colors.items()):
        passes.append(CollectOpColor(f"color_{color}", edges))

    pass_manager = PassManager(passes)
    sliced_circuit = pass_manager.run(circuit)
    slices = []
    for op_node in circuit_to_dag(sliced_circuit).op_nodes():
        if op_node.name != "slice_op":
            raise ValueError(f"Could not assign color to circuit instruction: {op_node.name}")
        qargs = [circuit.find_bit(q).index for q in op_node.qargs]
        qc = QuantumCircuit(circuit.num_qubits)
        qc.append(op_node.op, qargs)
        slices.append(qc.decompose())

    return slices
