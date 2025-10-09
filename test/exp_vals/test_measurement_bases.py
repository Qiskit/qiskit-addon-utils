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

"""Tests for measurement bases determination."""

from __future__ import annotations

from itertools import zip_longest

import numpy as np
import pytest
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit_addon_utils.exp_vals.measurement_bases import get_measurement_bases


@pytest.mark.parametrize(
    ["observables", "exp_bases", "exp_reverser"],
    [
        (
            SparsePauliOp("IXYZ"),
            [np.array([0, 1, 3, 2], dtype=np.uint8)],
            {},
        ),
        (
            [
                SparsePauliOp("IXXI"),
                SparsePauliOp("XXII"),
                SparsePauliOp("ZIIZ"),
                SparsePauliOp("IYYI"),
            ],
            [
                np.array([1, 1, 1, 0], dtype=np.uint8),
                np.array([2, 3, 3, 2], dtype=np.uint8),
            ],
            {},
        ),
    ],
)
def test_get_measurement_bases(
    observables: SparsePauliOp | list[SparsePauliOp],
    exp_bases: list[np.typing.NDArray[np.uint8]],
    exp_reverser: dict[Pauli, list[SparsePauliOp]],
) -> None:
    """Test `get_measurement_bases`.

    Args:
        observables: the observable(s) whose measurement bases to compute.
        bases: the expected measurement bases per group of paulis.
        reverser: the reverse lookup dictionary for associating an observables' Pauli with a basis.
    """
    actual_bases, actual_reverser = get_measurement_bases(observables)
    for actual_basis, exp_basis in zip_longest(actual_bases, exp_bases):
        assert np.allclose(actual_basis, exp_basis)

    # TODO: assert reverser!
    print(actual_reverser, exp_reverser)
