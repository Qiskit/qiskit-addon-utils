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

"""A method for re-combining circuit slices."""

from __future__ import annotations

from collections.abc import Sequence

from qiskit import QuantumCircuit


def combine_slices(
    slices: Sequence[QuantumCircuit], include_barriers: bool = False
) -> QuantumCircuit | None:
    """Combine N-qubit slices of a circuit into a single circuit.

    Args:
        slices: The N-qubit circuit slices.
        include_barriers: If ``True``, place barriers between each slice.

    Returns:
        A :class:`~qiskit.circuit.QuantumCircuit` with the slices appended in sequential order.

    Raises:
        ValueError: Two input slices were defined on different numbers of qubits.
    """
    if len(slices) == 0:
        return None

    num_qubits = slices[0].num_qubits
    circuit_out = QuantumCircuit(num_qubits)

    for i, slice_ in enumerate(slices):
        if slice_.num_qubits != num_qubits:
            raise ValueError(
                "All slices must be defined on the same number of qubits. "
                f"slices[0] contains {num_qubits} qubits, but slices[{i}] contains "
                f"{slice_.num_qubits} qubits."
            )
        circuit_out.append(slice_, range(num_qubits))
        if include_barriers and i < len(slices) - 1:
            circuit_out.barrier()

    return circuit_out.decompose()
