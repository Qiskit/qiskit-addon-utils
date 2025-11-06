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

"""Utility functions for mapping observables between qubit-indexing definitions."""

from __future__ import annotations

from collections.abc import Sequence

from qiskit.quantum_info import Pauli, SparseObservable, SparsePauliOp


def map_observable_isa_to_canonical(
    isa_observable: Pauli | SparsePauliOp | SparseObservable, canonical_qubits: Sequence[int]
) -> Pauli | SparsePauliOp | SparseObservable:
    """Map an observable defined relative to the transpiled circuit to canonical box-order.

    In the transpiled (or ISA) ordering, the qubits are indexed based on the "physical"
    layout of qubits in the device.

    For info on canonical qubit ordering conventions see the `Samplomatic docs <https://qiskit.github.io/samplomatic/guides/samplex_io.html#qubit-ordering-convention>`_).

    Args:
        isa_observable: A `Pauli`, `SparsePauliOp`, or `SparseObservable` object.
        canonical_qubits: A sequence specifying the physical qubit for each canonical qubit.

    Return:
        A mapped operator of the same type as ``isa_observable``
    """
    # maps canonical qubit to physical (isa) qubit
    c_2_p = {c: p for c, p in enumerate(canonical_qubits)}
    p_2_c = {p: c for c, p in c_2_p.items()}

    if isinstance(isa_observable, Pauli):
        if isa_observable.phase != 0:
            raise NotImplementedError("`Pauli` observable must have zero phase.")

        return isa_observable[canonical_qubits]

    if isinstance(isa_observable, SparsePauliOp):
        return SparsePauliOp.from_sparse_list(
            [
                (pstr, [p_2_c[p] for p in p_qubits], coeff)
                for (pstr, p_qubits, coeff) in isa_observable.to_sparse_list()
            ],
            num_qubits=len(canonical_qubits),
        )

    if isinstance(isa_observable, SparseObservable):
        return SparseObservable.from_sparse_list(
            [
                (pstr, [p_2_c[p] for p in p_qubits], coeff)
                for (pstr, p_qubits, coeff) in isa_observable.to_sparse_list()
            ],
            num_qubits=len(canonical_qubits),
        )
    raise ValueError(
        f"isa_observable of type {type(isa_observable)} is not supported, try "
        f"casting to a Pauli, SparsePauliOp, or SparseObservable. "
    )


def map_observable_virtual_to_canonical(
    virt_observable: Pauli | SparsePauliOp | SparseObservable,
    layout: Sequence[int],
    canonical_qubits: Sequence[int],
):
    """Map an observable with virtual qubit ordering to canonical box-order.

    For info on canonical qubit ordering conventions see the `Samplomatic docs <https://qiskit.github.io/samplomatic/guides/samplex_io.html#qubit-ordering-convention>`_).

    Args:
        virt_observable: A `Pauli`, `SparsePauliOp`, or `SparseObservable` object.
        layout: The list of physical qubits used for the isa circuit.
        canonical_qubits: A dictionary mapping canonical qubits within a box to physical qubits within the layout.

    Return:
        A mapped operator of the same type as ``virt_observable``
    """
    # maps canonical qubit to physical (isa) qubit
    c_2_p = {c: p for c, p in enumerate(canonical_qubits)}
    # maps physical (isa) qubit to virtual qubit (index in original circuit)
    p_2_v = {p: v for v, p in enumerate(layout)}

    # compute maps between virtual and canonical qubit indices.
    c_2_v = {c: p_2_v[p] for c, p in c_2_p.items()}
    v_2_c = {v: c for c, v in c_2_v.items()}

    num_c_qubits = len(canonical_qubits)

    if isinstance(virt_observable, Pauli):
        return Pauli(
            "".join([virt_observable.to_label()[::-1][c_2_v[c]] for c in range(num_c_qubits)])[::-1]
        )

    if isinstance(virt_observable, SparsePauliOp):
        return SparsePauliOp.from_sparse_list(
            [
                (pstr, [v_2_c[v] for v in v_qubits], coeff)
                for (pstr, v_qubits, coeff) in virt_observable.to_sparse_list()
            ],
            num_qubits=len(canonical_qubits),
        )

    if isinstance(virt_observable, SparseObservable):
        return SparseObservable.from_sparse_list(
            [
                (pstr, [v_2_c[v] for v in v_qubits], coeff)
                for (pstr, v_qubits, coeff) in virt_observable.to_sparse_list()
            ],
            num_qubits=len(canonical_qubits),
        )
    raise ValueError(
        f"virt_observable of type {type(virt_observable)} is not supported, try "
        f"casting to a Pauli, SparsePauliOp, or SparseObservable. "
    )


def map_observable_isa_to_virtual(
    isa_observable: Pauli | SparsePauliOp | SparseObservable, layout: Sequence[int]
) -> Pauli | SparsePauliOp | SparseObservable:
    """Map an observable defined relative to the transpiled circuit to virtual order.

    In the transpiled (or ISA) ordering, the qubits are indexed based on the "physical"
    layout of qubits in the device.

    Args:
        isa_observable: A `Pauli`, `SparsePauliOp`, or `SparseObservable` object.
        layout: The list of physical qubits used for the isa circuit.

    Return:
        A mapped operator of the same type as ``isa_observable``
    """
    # maps physical (isa) qubit to virtual qubit (index in original circuit)
    p_2_v = {p: v for v, p in enumerate(layout)}
    v_2_p = {v: p for p, v in p_2_v.items()}

    num_qubits = len(layout)

    if isinstance(isa_observable, Pauli):
        return Pauli(
            "".join([isa_observable.to_label()[::-1][v_2_p[c]] for c in range(num_qubits)])[::-1]
        )

    if isinstance(isa_observable, SparsePauliOp):
        return SparsePauliOp.from_sparse_list(
            [
                (pstr, [p_2_v[p] for p in p_qubits], coeff)
                for (pstr, p_qubits, coeff) in isa_observable.to_sparse_list()
            ],
            num_qubits=num_qubits,
        )

    if isinstance(isa_observable, SparseObservable):
        return SparseObservable.from_sparse_list(
            [
                (pstr, [p_2_v[p] for p in p_qubits], coeff)
                for (pstr, p_qubits, coeff) in isa_observable.to_sparse_list()
            ],
            num_qubits=num_qubits,
        )
    raise ValueError(
        f"isa_observable of type {type(isa_observable)} is not supported, try "
        f"casting to a Pauli, SparsePauliOp, or SparseObservable. "
    )
