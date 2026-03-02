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

"""TREX rescaling factor computation."""

from __future__ import annotations

from qiskit.quantum_info import (
    Pauli,
    PauliLindbladMap,
    QubitSparsePauli,
    SparseObservable,
    SparsePauliOp,
)


def trex_factors(
    measurement_noise_map: PauliLindbladMap,
    basis_dict: dict[Pauli, list[SparsePauliOp | None]],
    /,
):
    """Calculates TREX mitigation algorithm's expectation value scale factor for each Pauli term in ``basis_dict``.

    Calculates <Z^n> for each Pauli term in each observable, where n is the non identity indices in the Pauli term.
    The calculation is done using learned measurement noise.

    Args:
        measurement_noise_map: Learned measurement noise in PauliLindbladMap format.
        basis_dict: Mapping between measure bases and observables in which the TREX algorithm mitigates their
        expectation value calculation.

    Returns:
        A list of numpy array of floats that represent the TREX mitigation algorithm's expectation value scale factor
        for each Pauli term in each observable in each basis in the given basis_dict.

    """
    num_qubits = measurement_noise_map.num_qubits
    scales_each_basis = []
    for _, spo_list in basis_dict.items():
        scales_for_basis = []
        for spo in spo_list:
            sparse_observable = (
                SparseObservable(spo) if spo is not None else SparseObservable.zero(num_qubits)
            )
            scales_for_basis.append(
                [
                    1
                    / measurement_noise_map.pauli_fidelity(
                        QubitSparsePauli(
                            ("Z" * len(term.indices), term.indices), num_qubits=num_qubits
                        )
                    )
                    for term in sparse_observable
                ]
            )
        scales_each_basis.append(scales_for_basis)
    return scales_each_basis
