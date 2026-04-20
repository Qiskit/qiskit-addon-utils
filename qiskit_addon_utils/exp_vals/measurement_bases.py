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

from collections.abc import Sequence

import numpy as np
from qiskit.quantum_info import Pauli, PauliList, SparsePauliOp


def get_measurement_bases(
    observables: SparsePauliOp | list[SparsePauliOp],
    bases_in_int_format: bool = True,
) -> (
    tuple[list[np.typing.NDArray[np.uint8]], dict[Pauli, list[SparsePauliOp]]]
    | tuple[list[str], dict[Pauli, list[SparsePauliOp]]]
):
    """Choose bases to sample in order to calculate expectation values for all given observables.

    Here a "basis" refers to measurement of a full-weight or high-weight Pauli, from which multiple qubit-wise commuting Paulis may be estimated.

    The bases are chosen by grouping commuting Paulis across the different observables.

    Args:
        observables: The observables to calculate using the quantum computer.
        bases_in_int_format: If true, return bases as an array of ints, using the samplomatic convention of: I=0, Z=1, X=2, Y=3.
            otherwise, return the bases as a array of strings.

    Returns:
        * List of Pauli bases to sample encoded in a list of uint8 where 0=I,1=Z,2=X,3=Y or a list of strings (based on bases_in_int_format parameter).
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
    if bases_in_int_format:
        bases = _convert_basis_to_uint_representation(bases)
    else:
        bases = bases.to_labels()

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


def _convert_to_pauli(basis):
    """Converts a basis in various formats into a Pauli object.

    Can convert a string or a list of integers representing the Paulis using this convention:
    0=I, 1=Z, 2=X, 3=Y

    Args:
        basis: the basis to convert.

    Returns:
        The Pauli represented as a Pauli object.

    Raises:
        ValueError: if the basis is in invalid format.
    """
    int_mapping = {0: "I", 1: "Z", 2: "X", 3: "Y"}
    if isinstance(basis, Pauli):
        return basis
    if isinstance(basis, str):
        return Pauli(basis)
    if isinstance(basis, (list, np.ndarray, tuple)) and isinstance(
        basis[0], (np.unsignedinteger, int, np.integer)
    ):
        return Pauli("".join([int_mapping[int_val] for int_val in basis]))

    raise ValueError("basis must be a Pauli instance, str or a list of ints.")


def find_measure_basis_to_observable_mapping(
    observables: Sequence[SparsePauliOp], measure_bases: Sequence[str | int | PauliList]
) -> dict[Pauli, list[SparsePauliOp | None]]:
    """Maps each term for each observable to the first basis it qubit-wise commutes with from the given measure_bases.

    Each observable term must qubit-wise commute with at least one basis.

    Args:
        observables: list of observables.
        measure_bases: list of Pauli bases that the observables are measured with.

    Returns:
        A dictionary mapping from basis to observables terms that commutes with them.

    Raises:
        ValueError: If there is an observable with a term that does not qubit-wise commute with any basis from the given measure_bases.
    """
    measure_paulis = PauliList([_convert_to_pauli(basis) for basis in measure_bases])
    measurement_dict: dict[Pauli, list[SparsePauliOp]] = {}
    observables_elements_basis_found = []
    for basis in measure_paulis:
        measurement_dict[basis] = [[] for _ in range(len(observables))]

    for observable_index, observable in enumerate(observables):
        observables_elements_basis_found.append(np.zeros((len(observable)), dtype=np.bool))
        for basis in measure_paulis:
            basis_paulis = []
            basis_coeffs = []
            # find the elements that commutes with this basis
            for element_index, (observable_element, observable_coeff) in enumerate(
                zip(observable.paulis, observable.coeffs)
            ):
                # use only the first commuting basis found for each observable element
                # TODO: enable multiple bases for each element, lowering variance in the expectation value calculation
                if observables_elements_basis_found[observable_index][element_index]:
                    continue
                commutes = (
                    np.dot(observable_element.z, basis.x) + np.dot(observable_element.x, basis.z)
                ) % 2 == 0
                if commutes:
                    basis_paulis.append(observable_element)
                    basis_coeffs.append(observable_coeff)
                    observables_elements_basis_found[observable_index][element_index] = True
            measurement_dict[basis][observable_index] = (
                SparsePauliOp(basis_paulis, basis_coeffs) if basis_paulis else None
            )
    if any(
        False in observable_elements_list
        for observable_elements_list in observables_elements_basis_found
    ):
        raise ValueError("Some observable elements do not commute with any measurement basis.")
    return measurement_dict
