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
"""Tests for ``PostSelector`` (including pre-selection and combined selection)."""

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
    # Add measurements so the circuit has post-selection
    circuit.measure(qreg[0:3], creg0)
    circuit.measure(qreg[0:3], creg0_ps)

    post_selector = PostSelector.from_circuit(circuit, [])

    with pytest.raises(ValueError):
        post_selector.compute_mask({}, strategy="invalid", mode="post")

    result = {"alpha": np.zeros((1, 3), dtype=bool), "alpha_ps": np.zeros((2, 3), dtype=bool)}
    with pytest.raises(ValueError, match="arrays of inconsistent shapes"):
        post_selector.compute_mask(result, strategy="node", mode="post")

    result = {"beta": np.zeros((1, 2), dtype=bool)}
    with pytest.raises(ValueError, match="Result does not contain creg"):
        post_selector.compute_mask(result, strategy="node", mode="post")


def test_pre_selection_constructors():
    """Test the constructors for pre-selection."""
    qreg = QuantumRegister(3, "q")
    creg = ClassicalRegister(3, "alpha")
    creg_pre = ClassicalRegister(3, "alpha_pre")

    circuit = QuantumCircuit(qreg, creg, creg_pre)
    circuit.measure(qreg, creg_pre)
    circuit.barrier()
    circuit.measure(qreg, creg)

    coupling_map = [(0, 1), (1, 2), (2, 3)]

    summary = PostSelectionSummary.from_circuit(
        circuit, coupling_map, pre_selection_suffix="_pre", validation_mode="lenient"
    )
    selector = PostSelector(summary)
    assert selector.summary == summary

    selector = PostSelector.from_circuit(
        circuit, coupling_map, pre_selection_suffix="_pre", validation_mode="lenient"
    )
    assert selector.summary == summary


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
    selector = PostSelector.from_circuit(
        circuit, coupling_map, pre_selection_suffix="_pre", validation_mode="lenient"
    )

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

    mask = selector.compute_mask(result0, mode="pre")
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
    mask = selector.compute_mask(result1, strategy="node", mode="pre")
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
    selector = PostSelector.from_circuit(
        circuit, coupling_map, pre_selection_suffix="_pre", validation_mode="lenient"
    )

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

    mask = selector.compute_mask(result0, mode="pre")
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
    mask = selector.compute_mask(result1, strategy="edge", mode="pre")
    expected = np.ones(outer_shape, dtype=bool)
    expected[1, 10] = False
    assert np.all(mask == expected)


def test_combined_pre_and_post_selection():
    """Test combined pre and post selection."""
    qreg = QuantumRegister(3, "q")
    creg = ClassicalRegister(3, "alpha")
    creg_pre = ClassicalRegister(3, "alpha_pre")
    creg_ps = ClassicalRegister(3, "alpha_ps")

    circuit = QuantumCircuit(qreg, creg, creg_pre, creg_ps)
    # Pre-selection measurements at the start
    circuit.measure(qreg, creg_pre)
    circuit.barrier()
    # Terminal measurements
    circuit.measure(qreg, creg)
    # Post-selection measurements
    circuit.measure(qreg, creg_ps)

    coupling_map = [(0, 1), (1, 2)]
    selector = PostSelector.from_circuit(
        circuit,
        coupling_map,
        post_selection_suffix="_ps",
        pre_selection_suffix="_pre",
        validation_mode="strict",
    )

    # Generate results with 10 shots
    outer_shape = (10,)
    alpha = np.random.randint(0, high=2, size=(*outer_shape, len(creg)), dtype=bool)

    # All pre-selection good, all post-selection good
    alpha_pre = np.zeros((*outer_shape, len(creg_pre)), dtype=bool)
    alpha_ps = ~alpha
    result_all_good = {
        "alpha": alpha,
        "alpha_pre": alpha_pre,
        "alpha_ps": alpha_ps,
    }

    mask = selector.compute_mask(result_all_good, strategy="node", mode="both")
    expected = np.ones(outer_shape, dtype=bool)
    assert np.all(mask == expected)

    # Some pre-selection failures, some post-selection failures
    alpha_pre_bad = np.zeros((*outer_shape, len(creg_pre)), dtype=bool)
    alpha_ps_bad = ~alpha.copy()

    # Shot 0: pre-selection fails
    alpha_pre_bad[0, 0] = True
    # Shot 1: post-selection fails
    alpha_ps_bad[1, 1] = alpha[1, 1]
    # Shot 2: both fail
    alpha_pre_bad[2, 2] = True
    alpha_ps_bad[2, 0] = alpha[2, 0]

    result_mixed = {
        "alpha": alpha,
        "alpha_pre": alpha_pre_bad,
        "alpha_ps": alpha_ps_bad,
    }

    mask = selector.compute_mask(result_mixed, strategy="node", mode="both")
    expected = np.ones(outer_shape, dtype=bool)
    expected[0] = expected[1] = expected[2] = False
    assert np.all(mask == expected)

    # Test edge strategy for both mode
    mask_edge = selector.compute_mask(result_mixed, strategy="edge", mode="both")
    # With edge strategy:
    # - Shot 0: pre-selection fails on qubit 0 (non-edge with qubit 2), but no edge failure -> kept
    # - Shot 1: post-selection fails on qubit 1 (non-edge with qubit 0 or 2 alone), but no edge failure -> kept
    # - Shot 2: both pre and post fail, creating edge failures -> discarded
    # Actually, we need to check if failures create edge conditions
    # For this simple 3-qubit case with edges (0,1) and (1,2):
    # Shot 0: qubit 0 pre fails - check edges (0,1): qubit 1 pre is good, so edge (0,1) is OK
    # Shot 1: qubit 1 post fails - check edges (0,1) and (1,2): need both qubits to fail
    # Shot 2: qubit 2 pre fails and qubit 0 post fails - check edge (1,2): qubit 1 is good
    # So with edge strategy, all shots might pass! Let me recalculate...
    # Actually the test data has failures that don't form edges, so they should all pass with edge strategy
    expected_edge = np.ones(outer_shape, dtype=bool)
    # Only discard if BOTH qubits on an edge fail their checks
    # Shot 0: qubit 0 pre=1 (fail), qubit 1 pre=0 (pass) -> edge (0,1) has one pass -> keep
    # Shot 1: qubit 1 post fails, but we need both qubits on edge to fail
    # Shot 2: qubit 2 pre=1, qubit 0 post fails - edge (1,2) has qubit 1 passing -> keep
    # So all should pass with edge strategy
    assert np.all(mask_edge == expected_edge)


def test_mode_errors():
    """Test that appropriate errors are raised for missing measurements."""
    qreg = QuantumRegister(3, "q")
    creg = ClassicalRegister(3, "alpha")
    creg_ps = ClassicalRegister(3, "alpha_ps")

    # Circuit with only post-selection
    circuit_ps_only = QuantumCircuit(qreg, creg, creg_ps)
    circuit_ps_only.measure(qreg, creg)
    circuit_ps_only.measure(qreg, creg_ps)

    selector_ps = PostSelector.from_circuit(circuit_ps_only, [(0, 1), (1, 2)])

    # Should work for post mode
    result = {
        "alpha": np.zeros((5, 3), dtype=bool),
        "alpha_ps": np.ones((5, 3), dtype=bool),
    }
    mask = selector_ps.compute_mask(result, mode="post")
    assert mask.shape == (5,)

    # Should fail for pre mode
    with pytest.raises(ValueError, match="No pre-selection measurements"):
        selector_ps.compute_mask(result, mode="pre")

    # Should fail for both mode
    with pytest.raises(ValueError, match="No pre-selection measurements"):
        selector_ps.compute_mask(result, mode="both")

    # Circuit with only pre-selection
    creg_pre = ClassicalRegister(3, "alpha_pre")
    circuit_pre_only = QuantumCircuit(qreg, creg, creg_pre)
    circuit_pre_only.measure(qreg, creg_pre)
    circuit_pre_only.barrier()
    circuit_pre_only.measure(qreg, creg)

    selector_pre = PostSelector.from_circuit(
        circuit_pre_only, [(0, 1), (1, 2)], pre_selection_suffix="_pre", validation_mode="lenient"
    )

    # Should work for pre mode
    result_pre = {
        "alpha": np.zeros((5, 3), dtype=bool),
        "alpha_pre": np.zeros((5, 3), dtype=bool),
    }
    mask = selector_pre.compute_mask(result_pre, mode="pre")
    assert mask.shape == (5,)

    # Should fail for post mode
    with pytest.raises(ValueError, match="No post-selection measurements"):
        selector_pre.compute_mask(result_pre, mode="post")

    # Should fail for both mode
    with pytest.raises(ValueError, match="No post-selection measurements"):
        selector_pre.compute_mask(result_pre, mode="both")


def test_pre_selection_with_post_selector():
    """Test that PostSelector works correctly for pre-selection mode."""
    qreg = QuantumRegister(3, "q")
    creg = ClassicalRegister(3, "alpha")
    creg_pre = ClassicalRegister(3, "alpha_pre")

    circuit = QuantumCircuit(qreg, creg, creg_pre)
    circuit.measure(qreg, creg_pre)
    circuit.barrier()
    circuit.measure(qreg, creg)

    coupling_map = [(0, 1), (1, 2)]

    # PostSelector should handle pre-selection
    selector = PostSelector.from_circuit(
        circuit, coupling_map, pre_selection_suffix="_pre", validation_mode="lenient"
    )

    # Test with all good pre-selection measurements
    result_good = {
        "alpha": np.zeros((5, 3), dtype=bool),
        "alpha_pre": np.zeros((5, 3), dtype=bool),
    }

    mask = selector.compute_mask(result_good, mode="pre")
    assert mask.shape == (5,)
    assert np.all(mask)  # All shots should pass

    # Test with some bad pre-selection measurements
    result_bad = {
        "alpha": np.zeros((5, 3), dtype=bool),
        "alpha_pre": np.zeros((5, 3), dtype=bool),
    }
    result_bad["alpha_pre"][2, 1] = True  # Shot 2 fails pre-selection

    mask = selector.compute_mask(result_bad, mode="pre", strategy="node")
    assert mask.shape == (5,)
    assert np.sum(mask) == 4  # 4 out of 5 shots should pass
    assert not mask[2]  # Shot 2 should be discarded


def test_validation_errors_post_selection():
    """Test validation errors for post-selection."""
    qreg = QuantumRegister(3, "q")
    creg = ClassicalRegister(3, "alpha")
    creg_ps = ClassicalRegister(3, "alpha_ps")

    circuit = QuantumCircuit(qreg, creg, creg_ps)
    circuit.measure(qreg, creg)
    circuit.measure(qreg, creg_ps)

    selector = PostSelector.from_circuit(circuit, [(0, 1), (1, 2)])

    # Test missing register error (line 277)
    result_missing = {"alpha": np.zeros((5, 3), dtype=bool)}
    with pytest.raises(ValueError, match="Result does not contain creg 'alpha_ps'"):
        selector.compute_mask(result_missing, mode="post")

    # Test inconsistent shapes error (line 280)
    result_inconsistent = {
        "alpha": np.zeros((5, 3), dtype=bool),
        "alpha_ps": np.zeros((3, 3), dtype=bool),  # Different shape
    }
    with pytest.raises(ValueError, match="arrays of inconsistent shapes"):
        selector.compute_mask(result_inconsistent, mode="post")


def test_validation_errors_pre_selection():
    """Test validation errors for pre-selection."""
    qreg = QuantumRegister(3, "q")
    creg = ClassicalRegister(3, "alpha")
    creg_pre = ClassicalRegister(3, "alpha_pre")

    circuit = QuantumCircuit(qreg, creg, creg_pre)
    circuit.measure(qreg, creg_pre)
    circuit.barrier()
    circuit.measure(qreg, creg)

    selector = PostSelector.from_circuit(
        circuit, [(0, 1), (1, 2)], pre_selection_suffix="_pre", validation_mode="lenient"
    )

    # Test missing register error (line 300)
    result_missing = {"alpha": np.zeros((5, 3), dtype=bool)}
    with pytest.raises(ValueError, match="Result does not contain creg 'alpha_pre'"):
        selector.compute_mask(result_missing, mode="pre")

    # Test inconsistent shapes error (line 303)
    result_inconsistent = {
        "alpha": np.zeros((5, 3), dtype=bool),
        "alpha_pre": np.zeros((3, 3), dtype=bool),  # Different shape
    }
    with pytest.raises(ValueError, match="arrays of inconsistent shapes"):
        selector.compute_mask(result_inconsistent, mode="pre")


# Made with Bob
