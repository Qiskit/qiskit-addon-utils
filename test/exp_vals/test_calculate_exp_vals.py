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

import numpy as np
import pytest
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit_addon_utils.exp_vals.calculate_exp_vals import calculate_expectation_values


@pytest.mark.parametrize(
    ["meas_results", "reverser", "signs", "gamma", "output"],
    [
        (
            np.array([[False, True]]),
            {Pauli("IZ"): [SparsePauliOp("IZ")]},
            None,
            None,
            (np.array([[-1.0]]), np.array([[0.0]])),
        ),
    ],
)
def test_calculate_expectation_values(
    meas_results: np.typing.NDArray[np.bool_],
    reverser: dict[Pauli, list[SparsePauliOp]],
    signs: np.typing.NDArray[np.bool_] | None,
    gamma: float | None,
    output: tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]],
) -> None:
    expvals = calculate_expectation_values(meas_results, reverser, signs, gamma)
    assert np.allclose(output, expvals)
