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
"""Tests for ``PreSelector``."""

import numpy as np
import pytest
from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_addon_utils.noise_management.post_selection import PreSelectionSummary, PreSelector


def test_constructors():
    """Test the constructors."""
    qreg = QuantumRegister(3, "q")
    creg = ClassicalRegister(3, "alpha")
    creg_pre = ClassicalRegister(3, "alpha_pre")

    circuit = QuantumCircuit(qreg, creg, creg_pre)
    circuit.measure(qreg, creg_pre)
    circuit.barrier()
    circuit.measure(qreg, creg)

    coupling_map = [(0, 1), (1, 2), (2, 3)]

    summary = PreSelectionSummary.from_circuit(circuit, coupling_map)
    pre_selector = PreSelector(summary)
    assert pre_selector.summary == summary

    pre_selector = PreSelector.from_circuit(circuit, coupling_map)
    assert pre_selector.summary == summary


def test_node_based_pre_selection():
    """Test node-based pre selection."""
    qreg = QuantumRegister(5, "q")
    creg0 = ClassicalRegister(3, "alpha")
    creg0_pre = ClassicalRegister(3, "alpha_pre")
    creg1 = ClassicalRegister(2, "beta")
    creg1_pre = ClassicalRegister(2, "beta_pre")

    circuit = QuantumCircuit(qreg, creg0, creg0_pre, creg1, creg1_pre)
    # Pre-selection measurements
    circuit.measure(qreg[0], creg0_pre[0])
    circuit.measure(qreg[1], creg0_pre[1])
    circuit.measure(qreg[2], creg0_pre[2])
    circuit.measure(qreg[3], creg1_pre[0])
    circuit.measure(qreg[4], creg1_pre[1])
    circuit.barrier()
    # Terminal measurements
    circuit.measure(qreg[0], creg0[0])
    circuit.measure(qreg[1], creg0[1])
    circuit.measure(qreg[2], creg0[2])
    circuit.measure(qreg[3], creg1[0])
    circuit.measure(qreg[4], creg1[1])

    coupling_map = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]
    pre_selector = PreSelector.from_circuit(circuit, coupling_map)

    # Generate results with 12 randomizations and 15 shots
    outer_shape = (12, 15)
    alpha = np.random.randint(0, high=2, size=(*outer_shape, len(creg0)), dtype=bool)
    beta = np.random.randint(0, high=2, size=(*outer_shape, len(creg1)), dtype=bool)

    # Every pre-selection measurement returns 0 (good initialization)
    alpha_pre0 = np.zeros((*outer_shape, len(creg0_pre)), dtype=bool)
    beta_pre0 = np.zeros((*outer_shape, len(creg1_pre)), dtype=bool)
    result0 = {
        "alpha": alpha,
        "alpha_pre": alpha_pre0,
        "beta": beta,
        "beta_pre": beta_pre0,
    }

    mask = pre_selector.compute_mask(result0)
    expected = np.ones(outer_shape, dtype=bool)
    assert np.all(mask == expected)

    # In another round, some pre-selection measurements return 1 (bad initialization)
    alpha_pre1 = np.zeros((*outer_shape, len(creg0_pre)), dtype=bool)
    beta_pre1 = np.zeros((*outer_shape, len(creg1_pre)), dtype=bool)
    # In shot `0`, these failures occur on two non-neighboring qubits -> shot discarded
    alpha_pre1[0, 0, 0] = True
    alpha_pre1[0, 0, 2] = True
    # Also in shot `3` these failures occur on two non-neighboring qubits -> shot discarded
    alpha_pre1[5, 3, 1] = True
    beta_pre1[5, 3, 0] = True
    # In shot `10`, these failures occur on two neighboring qubits -> shot discarded
    alpha_pre1[1, 10, 0] = True
    beta_pre1[1, 10, -1] = True

    result1 = {
        "alpha": alpha,
        "alpha_pre": alpha_pre1,
        "beta": beta,
        "beta_pre": beta_pre1,
    }
    mask = pre_selector.compute_mask(result1, strategy="node")
    expected = np.ones(outer_shape, dtype=bool)
    expected[0, 0] = expected[5, 3] = expected[1, 10] = False
    assert np.all(mask == expected)


def test_edge_based_pre_selection():
    """Test edge-based pre selection."""
    qreg = QuantumRegister(5, "q")
    creg0 = ClassicalRegister(3, "alpha")
    creg0_pre = ClassicalRegister(3, "alpha_pre")
    creg1 = ClassicalRegister(2, "beta")
    creg1_pre = ClassicalRegister(2, "beta_pre")

    circuit = QuantumCircuit(qreg, creg0, creg0_pre, creg1, creg1_pre)
    # Pre-selection measurements
    circuit.measure(qreg[0], creg0_pre[0])
    circuit.measure(qreg[1], creg0_pre[1])
    circuit.measure(qreg[2], creg0_pre[2])
    circuit.measure(qreg[3], creg1_pre[0])
    circuit.measure(qreg[4], creg1_pre[1])
    circuit.barrier()
    # Terminal measurements
    circuit.measure(qreg[0], creg0[0])
    circuit.measure(qreg[1], creg0[1])
    circuit.measure(qreg[2], creg0[2])
    circuit.measure(qreg[3], creg1[0])
    circuit.measure(qreg[4], creg1[1])

    coupling_map = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]
    pre_selector = PreSelector.from_circuit(circuit, coupling_map)

    # Generate results with 12 randomizations and 15 shots
    outer_shape = (12, 15)
    alpha = np.random.randint(0, high=2, size=(*outer_shape, len(creg0)), dtype=bool)
    beta = np.random.randint(0, high=2, size=(*outer_shape, len(creg1)), dtype=bool)

    # Every pre-selection measurement returns 0 (good initialization)
    alpha_pre0 = np.zeros((*outer_shape, len(creg0_pre)), dtype=bool)
    beta_pre0 = np.zeros((*outer_shape, len(creg1_pre)), dtype=bool)
    result0 = {
        "alpha": alpha,
        "alpha_pre": alpha_pre0,
        "beta": beta,
        "beta_pre": beta_pre0,
    }

    mask = pre_selector.compute_mask(result0)
    expected = np.ones(outer_shape, dtype=bool)
    assert np.all(mask == expected)

    # In another round, some pre-selection measurements return 1 (bad initialization)
    alpha_pre1 = np.zeros((*outer_shape, len(creg0_pre)), dtype=bool)
    beta_pre1 = np.zeros((*outer_shape, len(creg1_pre)), dtype=bool)
    # In shot `0`, these failures occur on two non-neighboring qubits -> shot kept
    alpha_pre1[0, 0, 0] = True
    alpha_pre1[0, 0, 2] = True
    # Also in shot `3` these failures occur on two non-neighboring qubits -> shot kept
    alpha_pre1[5, 3, 1] = True
    beta_pre1[5, 3, 0] = True
    # In shot `10`, these failures occur on two neighboring qubits -> shot discarded
    alpha_pre1[1, 10, 0] = True
    beta_pre1[1, 10, -1] = True

    result1 = {
        "alpha": alpha,
        "alpha_pre": alpha_pre1,
        "beta": beta,
        "beta_pre": beta_pre1,
    }
    mask = pre_selector.compute_mask(result1, strategy="edge")
    expected = np.ones(outer_shape, dtype=bool)
    expected[1, 10] = False
    assert np.all(mask == expected)


def test_raises():
    """Test that the PreSelector raises."""
    qreg = QuantumRegister(5, "q")
    creg0 = ClassicalRegister(3, "alpha")
    creg0_pre = ClassicalRegister(3, "alpha_pre")
    circuit = QuantumCircuit(qreg, creg0, creg0_pre)

    pre_selector = PreSelector.from_circuit(circuit, [])

    with pytest.raises(ValueError):
        pre_selector.compute_mask({}, strategy="invalid")

    result = {"alpha": np.zeros((1, 2), dtype=bool), "alpha_pre": np.zeros((2, 2), dtype=bool)}
    with pytest.raises(ValueError, match="arrays of inconsistent shapes"):
        pre_selector.compute_mask(result, strategy="node")

    result = {"beta": np.zeros((1, 2), dtype=bool)}
    with pytest.raises(ValueError, match="Result does not contain creg"):
        pre_selector.compute_mask(result, strategy="node")


# Made with Bob
