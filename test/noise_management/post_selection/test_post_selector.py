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
"""Tests for ``PostSelector``."""

import numpy as np
import pytest
from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_addon_utils.noise_management.post_selection import PostSelectionSummary, PostSelector


def test_constructors():
    """Test the constructors."""
    qreg = QuantumRegister(3, "q")
    creg = ClassicalRegister(3, "alpha")
    creg_ps = ClassicalRegister(3, "alpha_ps")

    circuit = QuantumCircuit(qreg, creg, creg_ps)
    circuit.measure(qreg, creg)
    circuit.measure(qreg, creg_ps)

    coupling_map = [(0, 1), (1, 2), (2, 3)]

    summary = PostSelectionSummary.from_circuit(circuit, coupling_map)
    post_selector = PostSelector(summary)
    assert post_selector.summary == summary

    post_selector = PostSelector.from_circuit(circuit, coupling_map)
    assert post_selector.summary == summary


def test_node_based_post_selection():
    """Test node-based post selection."""
    qreg = QuantumRegister(5, "q")
    creg0 = ClassicalRegister(3, "alpha")
    creg0_ps = ClassicalRegister(3, "alpha_ps")
    creg1 = ClassicalRegister(2, "beta")
    creg1_ps = ClassicalRegister(2, "beta_ps")

    circuit = QuantumCircuit(qreg, creg0, creg0_ps, creg1, creg1_ps)
    circuit.measure(qreg[0], creg0[0])
    circuit.measure(qreg[1], creg0[1])
    circuit.measure(qreg[2], creg0[2])
    circuit.measure(qreg[3], creg1[0])
    circuit.measure(qreg[4], creg1[1])
    circuit.measure(qreg[0], creg0_ps[0])
    circuit.measure(qreg[1], creg0_ps[1])
    circuit.measure(qreg[2], creg0_ps[2])
    circuit.measure(qreg[3], creg1_ps[0])
    circuit.measure(qreg[4], creg1_ps[1])

    coupling_map = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]
    post_selector = PostSelector.from_circuit(circuit, coupling_map)

    # Generate results with 12 randomizations and 15 shots
    outer_shape = (12, 15)
    alpha = np.random.randint(0, high=2, size=(*outer_shape, len(creg0)), dtype=bool)
    beta = np.random.randint(0, high=2, size=(*outer_shape, len(creg1)), dtype=bool)

    # Every post selection measurement is correctly flipped
    alpha_ps0 = ~alpha
    beta_ps0 = ~beta
    result0 = {
        "alpha": alpha,
        "alpha_ps": alpha_ps0,
        "beta": beta,
        "beta_ps": beta_ps0,
    }

    mask = post_selector.compute_mask(result0)
    expected = np.ones(outer_shape, dtype=bool)
    assert np.all(mask == expected)

    # In another round, some post selection measurements fail to flip
    alpha_ps1 = ~alpha
    beta_ps1 = ~beta
    # In shot `0`, these failures occur on two non-neighboring qubits -> shot discarded
    alpha_ps1[0, 0, 0] = alpha[0, 0, 0]
    alpha_ps1[0, 0, 2] = alpha[0, 0, 2]
    # Also in shot `3` these failures occur on two non-neighboring qubits -> shot discarded
    alpha_ps1[5, 3, 1] = alpha[5, 3, 1]
    beta_ps1[5, 3, 0] = beta[5, 3, 0]
    # In shot `10`, these failures occur on two neighboring qubits -> shot discarded
    alpha_ps1[1, 10, 0] = alpha[1, 10, 0]
    beta_ps1[1, 10, -1] = beta[1, 10, -1]

    result1 = {
        "alpha": alpha,
        "alpha_ps": alpha_ps1,
        "beta": beta,
        "beta_ps": beta_ps1,
    }
    mask = post_selector.compute_mask(result1, strategy="node")
    expected = np.ones(outer_shape, dtype=bool)
    expected[0, 0] = expected[5, 3] = expected[1, 10] = False
    assert np.all(mask == expected)


def test_edge_based_post_selection():
    """Test edge-based post selection."""
    qreg = QuantumRegister(5, "q")
    creg0 = ClassicalRegister(3, "alpha")
    creg0_ps = ClassicalRegister(3, "alpha_ps")
    creg1 = ClassicalRegister(2, "beta")
    creg1_ps = ClassicalRegister(2, "beta_ps")

    circuit = QuantumCircuit(qreg, creg0, creg0_ps, creg1, creg1_ps)
    circuit.measure(qreg[0], creg0[0])
    circuit.measure(qreg[1], creg0[1])
    circuit.measure(qreg[2], creg0[2])
    circuit.measure(qreg[3], creg1[0])
    circuit.measure(qreg[4], creg1[1])
    circuit.measure(qreg[0], creg0_ps[0])
    circuit.measure(qreg[1], creg0_ps[1])
    circuit.measure(qreg[2], creg0_ps[2])
    circuit.measure(qreg[3], creg1_ps[0])
    circuit.measure(qreg[4], creg1_ps[1])

    coupling_map = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]
    post_selector = PostSelector.from_circuit(circuit, coupling_map)

    # Generate results with 12 randomizations and 15 shots
    outer_shape = (12, 15)
    alpha = np.random.randint(0, high=2, size=(*outer_shape, len(creg0)), dtype=bool)
    beta = np.random.randint(0, high=2, size=(*outer_shape, len(creg1)), dtype=bool)

    # Every post selection measurement is correctly flipped
    alpha_ps0 = ~alpha
    beta_ps0 = ~beta
    result0 = {
        "alpha": alpha,
        "alpha_ps": alpha_ps0,
        "beta": beta,
        "beta_ps": beta_ps0,
    }

    mask = post_selector.compute_mask(result0)
    expected = np.ones(outer_shape, dtype=bool)
    assert np.all(mask == expected)

    # In another round, some post selection measurements fail to flip
    alpha_ps1 = ~alpha
    beta_ps1 = ~beta
    # In shot `0`, these failures occur on two non-neighboring qubits -> shot kept
    alpha_ps1[0, 0, 0] = alpha[0, 0, 0]
    alpha_ps1[0, 0, 2] = alpha[0, 0, 2]
    # Also in shot `3` these failures occur on two non-neighboring qubits -> shot kept
    alpha_ps1[5, 3, 1] = alpha[5, 3, 1]
    beta_ps1[5, 3, 0] = beta[5, 3, 0]
    # In shot `10`, these failures occur on two neighboring qubits -> shot discarded
    alpha_ps1[1, 10, 0] = alpha[1, 10, 0]
    beta_ps1[1, 10, -1] = beta[1, 10, -1]

    result1 = {
        "alpha": alpha,
        "alpha_ps": alpha_ps1,
        "beta": beta,
        "beta_ps": beta_ps1,
    }
    mask = post_selector.compute_mask(result1, strategy="edge")
    expected = np.ones(outer_shape, dtype=bool)
    expected[1, 10] = False
    assert np.all(mask == expected)


def test_raises():
    """Test that the PostSelector raises."""
    qreg = QuantumRegister(5, "q")
    creg0 = ClassicalRegister(3, "alpha")
    creg0_ps = ClassicalRegister(3, "alpha_ps")
    circuit = QuantumCircuit(qreg, creg0, creg0_ps)

    post_selector = PostSelector.from_circuit(circuit, [])

    with pytest.raises(ValueError):
        post_selector.compute_mask({}, strategy="invalid")

    result = {"alpha": np.zeros((1, 2), dtype=bool), "alpha_ps": np.zeros((2, 2), dtype=bool)}
    with pytest.raises(ValueError, match="arrays of inconsistent shapes"):
        post_selector.compute_mask(result, strategy="node")

    result = {"beta": np.zeros((1, 2), dtype=bool)}
    with pytest.raises(ValueError, match="Result does not contain creg"):
        post_selector.compute_mask(result, strategy="node")
