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

"""Utility functions related to error mitigation algorithms."""

from qiskit.quantum_info import (
    Pauli,
    SparsePauliOp,
    PauliLindbladMap,
    SparseObservable,
    QubitSparsePauli,
)


def trex_factors(
    measurement_noise_map: PauliLindbladMap, basis_dict: dict[Pauli, list[SparsePauliOp | None]]
):
    """Calculates TREX mitigation algorithm's expectation value scale factor for each Pauli term in each observable
    in the basis dictionary.

    Calculates <Z^n> for each Pauli term in each observable, where n is the non identity indices in the Pauli term.
    The calculation is done using learned measurement noise.

    Args:
        measurement_noise_map: Learned measurement noise in PauliLindbladMap format.
        basis_dict: Mapping between measure bases and observables in which the TREX algorithm mitigates their
        expectation value calculation.

    Returns:
        A list of floats that represent the TREX mitigation algorithm's expectation value scale factor for each
        Pauli term in each observable.

    """
    scales_each_basis = []
    for _, spo_list in basis_dict.items():
        sparse_obs_list = []
        for spo in spo_list:
            sparse_obs_list.append(SparseObservable(spo))
        num_bits = sparse_obs_list[0].num_qubits
        observable_sum = sum(sparse_obs_list, start=SparseObservable.zero(num_bits))
        scales_each_basis.append(
            [
                measurement_noise_map.pauli_fidelity(
                    QubitSparsePauli(("Z" * len(term.indices), term.indices), num_qubits=num_bits)
                )
                for term in observable_sum
            ]
        )
    return scales_each_basis
