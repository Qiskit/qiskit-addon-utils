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

"""Utility functions for selecting efficient bases for measurement of observables."""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import Pauli, PauliList, SparsePauliOp


def get_measurement_bases(
    observables: SparsePauliOp | list[SparsePauliOp],
) -> tuple[list[np.typing.NDArray[np.uint8]], dict[Pauli, list[SparsePauliOp]]]:
    """Choose bases to sample in order to calculate expectation values for all given observables.

    Here a "basis" refers to measurement of a full-weight or high-weight Pauli, from which multiple qubit-wise commuting Paulis may be estimated.

    The bases are chosen by grouping commuting Paulis across the different observables.

    Args:
        observables: The observables to calculate using the quantum computer.

    Returns:
        * List of Pauli bases to sample encoded in a list of uint8 where 0=I,1=Z,2=X,3=Y.
        * Dict that maps each measured basis to the relevant Paulis and their coefficients for each observable.
          With the measured bases as keys, for each observable there is a SparsePauliOp representing it.
    """
    if isinstance(observables, SparsePauliOp):
        observables = [observables]
    combined_paulis = sum([obs.paulis for obs in observables]).unique()
    pauli_groups = combined_paulis.group_commuting(qubit_wise=True)
    bases = PauliList([_meas_basis_for_pauli_group(group) for group in pauli_groups])

    observables_as_dicts = [dict(obs.label_iter()) for obs in observables]
    reverser: dict[Pauli, list[SparsePauliOp]] = {}
    for basis, group in zip(bases, pauli_groups):
        reverser[basis] = [[] for _ in range(len(observables))]
        current_basis_weight = np.complex128(0)
        for i, observable in enumerate(observables_as_dicts):
            coeffs = []
            paulis = []
            for pauli in set(group):
                coeff = observable.get(pauli.to_label(), None)
                if coeff:
                    coeffs.append(coeff)
                    paulis.append(pauli)
                    current_basis_weight += coeff
            reverser[basis][i] = SparsePauliOp(paulis, coeffs) if paulis else None
    bases = _convert_basis_to_uint_representation(bases)

    return bases, reverser


def _meas_basis_for_pauli_group(group: PauliList) -> Pauli:
    """Find the collective measurement basis of a given commutative Pauli group.

    Args:
        group: The Pauli group to find the collective measurement basis to.

    Returns:
        The Pauli basis to measure that represents the given Pauli group.
    """
    sum_z = group.z.sum(axis=0, dtype=bool)
    sum_x = group.x.sum(axis=0, dtype=bool)
    return Pauli((sum_z, sum_x))


def _convert_basis_to_uint_representation(bases: PauliList) -> list[np.typing.NDArray[np.uint8]]:
    """Converts list of Paulis in PauliList format into a list of integers representing those Paulis.

    The representation of the Paulis as integers is:
    I=0, Z=1, X=2, Y=3

    Args:
        bases: The bases in PauliList format to convert.

    Returns:
        The bases represented as a list of integers.
    """
    pauli_to_int = {"I": 0, "Z": 1, "X": 2, "Y": 3}
    bases_uint8 = [
        np.array([pauli_to_int[p] for p in pauli.to_label()][::-1], dtype=np.uint8)
        for pauli in bases
    ]
    return bases_uint8
