# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Tests for ``PostSelector`` (including pre-check and combined selection)."""

import numpy as np
import pytest
from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_addon_utils.noise_management.bit_flip_checks import PostSelectionSummary, PostSelector


def _build_5q_check_circuit(suffix):
    """5-qubit circuit with ``alpha``/``beta`` primaries; ``"_ps"`` post-checks, ``"_pre"`` pre-checks."""
    qreg = QuantumRegister(5, "q")
    creg0 = ClassicalRegister(3, "alpha")
    creg0_chk = ClassicalRegister(3, f"alpha{suffix}")
    creg1 = ClassicalRegister(2, "beta")
    creg1_chk = ClassicalRegister(2, f"beta{suffix}")
    circuit = QuantumCircuit(qreg, creg0, creg0_chk, creg1, creg1_chk)

    primary = [(0, creg0, 0), (1, creg0, 1), (2, creg0, 2), (3, creg1, 0), (4, creg1, 1)]
    check = [
        (0, creg0_chk, 0),
        (1, creg0_chk, 1),
        (2, creg0_chk, 2),
        (3, creg1_chk, 0),
        (4, creg1_chk, 1),
    ]

    if suffix == "_pre":
        for q, creg, bit in check:
            circuit.measure(qreg[q], creg[bit])
        circuit.barrier()
        for q, creg, bit in primary:
            circuit.measure(qreg[q], creg[bit])
    else:
        for q, creg, bit in primary:
            circuit.measure(qreg[q], creg[bit])
        for q, creg, bit in check:
            circuit.measure(qreg[q], creg[bit])
    return circuit


def _make_check_results(mode, alpha, beta, outer_shape):
    """``result0`` passes every check; ``result1`` injects failures on non-neighbouring qubits of
    shots ``(0, 0)``/``(5, 3)`` and on neighbouring qubits of shot ``(1, 10)`` (closing edge ``(0, 4)``)."""
    if mode == "post":
        alpha_chk0 = ~alpha
        beta_chk0 = ~beta
        result0 = {"alpha": alpha, "alpha_ps": alpha_chk0, "beta": beta, "beta_ps": beta_chk0}

        alpha_chk1 = ~alpha
        beta_chk1 = ~beta
        # A post check measurement fails when it does *not* flip the primary bit.
        alpha_chk1[0, 0, 0] = alpha[0, 0, 0]
        alpha_chk1[0, 0, 2] = alpha[0, 0, 2]
        alpha_chk1[5, 3, 1] = alpha[5, 3, 1]
        beta_chk1[5, 3, 0] = beta[5, 3, 0]
        alpha_chk1[1, 10, 0] = alpha[1, 10, 0]
        beta_chk1[1, 10, -1] = beta[1, 10, -1]
        result1 = {"alpha": alpha, "alpha_ps": alpha_chk1, "beta": beta, "beta_ps": beta_chk1}
    else:
        alpha_chk0 = np.zeros((*outer_shape, alpha.shape[-1]), dtype=bool)
        beta_chk0 = np.zeros((*outer_shape, beta.shape[-1]), dtype=bool)
        result0 = {"alpha": alpha, "alpha_pre": alpha_chk0, "beta": beta, "beta_pre": beta_chk0}

        alpha_chk1 = np.zeros((*outer_shape, alpha.shape[-1]), dtype=bool)
        beta_chk1 = np.zeros((*outer_shape, beta.shape[-1]), dtype=bool)
        # A pre check measurement fails when it returns 1 (bad initialization).
        alpha_chk1[0, 0, 0] = True
        alpha_chk1[0, 0, 2] = True
        alpha_chk1[5, 3, 1] = True
        beta_chk1[5, 3, 0] = True
        alpha_chk1[1, 10, 0] = True
        beta_chk1[1, 10, -1] = True
        result1 = {"alpha": alpha, "alpha_pre": alpha_chk1, "beta": beta, "beta_pre": beta_chk1}
    return result0, result1


@pytest.mark.parametrize("suffix", ["_ps", "_pre"])
def test_constructors(suffix):
    """Test the constructors for post-check and pre-check."""
    qreg = QuantumRegister(3, "q")
    creg = ClassicalRegister(3, "alpha")
    creg_chk = ClassicalRegister(3, f"alpha{suffix}")

    circuit = QuantumCircuit(qreg, creg, creg_chk)
    if suffix == "_pre":
        circuit.measure(qreg, creg_chk)
        circuit.barrier()
        circuit.measure(qreg, creg)
        kwargs = {"pre_check_suffix": "_pre"}
    else:
        circuit.measure(qreg, creg)
        circuit.measure(qreg, creg_chk)
        kwargs = {}

    coupling_map = [(0, 1), (1, 2), (2, 3)]

    summary = PostSelectionSummary.from_circuit(circuit, coupling_map, **kwargs)
    post_selector = PostSelector(summary)
    assert post_selector.summary == summary

    post_selector = PostSelector.from_circuit(circuit, coupling_map, **kwargs)
    assert post_selector.summary == summary


@pytest.mark.parametrize(
    "mode, strategy",
    [
        ("post", "node"),
        ("post", "edge"),
        ("pre", "node"),
        ("pre", "edge"),
    ],
)
def test_node_and_edge_based_checks(mode, strategy):
    """Node- and edge-based selection for post/pre modes; ``edge`` cases uniquely cover mode->edge dispatch."""
    suffix = "_ps" if mode == "post" else "_pre"
    circuit = _build_5q_check_circuit(suffix)
    kwargs = {} if mode == "post" else {"pre_check_suffix": "_pre"}
    coupling_map = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]
    selector = PostSelector.from_circuit(circuit, coupling_map, **kwargs)

    # outer_shape is (randomizations, shots).
    outer_shape = (12, 15)
    alpha = np.random.randint(0, high=2, size=(*outer_shape, 3), dtype=bool)
    beta = np.random.randint(0, high=2, size=(*outer_shape, 2), dtype=bool)

    result0, result1 = _make_check_results(mode, alpha, beta, outer_shape)

    mask = selector.compute_mask(result0, mode=mode)
    expected = np.ones(outer_shape, dtype=bool)
    assert np.all(mask == expected)

    mask = selector.compute_mask(result1, strategy=strategy, mode=mode)
    expected = np.ones(outer_shape, dtype=bool)
    if strategy == "node":
        # Node discards any shot with an injected failure.
        expected[0, 0] = expected[5, 3] = expected[1, 10] = False
    else:
        # Edge discards only shot (1, 10), whose failures fall on neighbouring qubits.
        expected[1, 10] = False
    assert np.all(mask == expected)


def test_raises():
    """``compute_mask`` rejects an unknown strategy and inconsistent/missing result arrays."""
    qreg = QuantumRegister(5, "q")
    creg0 = ClassicalRegister(3, "alpha")
    creg0_ps = ClassicalRegister(3, "alpha_ps")
    circuit = QuantumCircuit(qreg, creg0, creg0_ps)
    circuit.measure(qreg[0:3], creg0)
    circuit.measure(qreg[0:3], creg0_ps)

    post_selector = PostSelector.from_circuit(circuit, [])

    with pytest.raises(ValueError, match="invalid"):
        post_selector.compute_mask({}, strategy="invalid", mode="post")

    result = {"alpha": np.zeros((1, 3), dtype=bool), "alpha_ps": np.zeros((2, 3), dtype=bool)}
    with pytest.raises(ValueError, match="arrays of inconsistent shapes"):
        post_selector.compute_mask(result, strategy="node", mode="post")

    result = {"beta": np.zeros((1, 2), dtype=bool)}
    with pytest.raises(ValueError, match="Result does not contain creg"):
        post_selector.compute_mask(result, strategy="node", mode="post")


def test_combined_pre_and_post_check():
    """``mode="both"`` keeps only shots passing both pre- and post-checks (logical AND)."""
    qreg = QuantumRegister(3, "q")
    creg = ClassicalRegister(3, "alpha")
    creg_pre = ClassicalRegister(3, "alpha_pre")
    creg_ps = ClassicalRegister(3, "alpha_ps")

    circuit = QuantumCircuit(qreg, creg, creg_pre, creg_ps)
    circuit.measure(qreg, creg_pre)
    circuit.barrier()
    circuit.measure(qreg, creg)
    circuit.measure(qreg, creg_ps)

    coupling_map = [(0, 1), (1, 2)]
    selector = PostSelector.from_circuit(
        circuit,
        coupling_map,
        post_check_suffix="_ps",
        pre_check_suffix="_pre",
    )

    outer_shape = (10,)
    alpha = np.random.randint(0, high=2, size=(*outer_shape, len(creg)), dtype=bool)

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

    # Shot 0 fails pre-check, shot 1 fails post-check, shot 2 fails both.
    alpha_pre_bad = np.zeros((*outer_shape, len(creg_pre)), dtype=bool)
    alpha_ps_bad = ~alpha.copy()
    alpha_pre_bad[0, 0] = True
    alpha_ps_bad[1, 1] = alpha[1, 1]
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

    # Edge strategy: no injected failure falls on a shared edge, so all shots are kept.
    mask_edge = selector.compute_mask(result_mixed, strategy="edge", mode="both")
    expected_edge = np.ones(outer_shape, dtype=bool)
    assert np.all(mask_edge == expected_edge)


def test_mode_errors():
    """A mode requiring checks the circuit lacks raises."""
    qreg = QuantumRegister(3, "q")
    creg = ClassicalRegister(3, "alpha")
    creg_ps = ClassicalRegister(3, "alpha_ps")

    # Post-check only: pre/both modes should raise.
    circuit_ps_only = QuantumCircuit(qreg, creg, creg_ps)
    circuit_ps_only.measure(qreg, creg)
    circuit_ps_only.measure(qreg, creg_ps)

    selector_ps = PostSelector.from_circuit(circuit_ps_only, [(0, 1), (1, 2)])

    result = {
        "alpha": np.zeros((5, 3), dtype=bool),
        "alpha_ps": np.ones((5, 3), dtype=bool),
    }
    mask = selector_ps.compute_mask(result, mode="post")
    assert mask.shape == (5,)

    with pytest.raises(ValueError, match="No pre-check measurements"):
        selector_ps.compute_mask(result, mode="pre")

    with pytest.raises(ValueError, match="No pre-check measurements"):
        selector_ps.compute_mask(result, mode="both")

    # Pre-check only: post/both modes should raise.
    creg_pre = ClassicalRegister(3, "alpha_pre")
    circuit_pre_only = QuantumCircuit(qreg, creg, creg_pre)
    circuit_pre_only.measure(qreg, creg_pre)
    circuit_pre_only.barrier()
    circuit_pre_only.measure(qreg, creg)

    selector_pre = PostSelector.from_circuit(
        circuit_pre_only, [(0, 1), (1, 2)], pre_check_suffix="_pre"
    )

    result_pre = {
        "alpha": np.zeros((5, 3), dtype=bool),
        "alpha_pre": np.zeros((5, 3), dtype=bool),
    }
    mask = selector_pre.compute_mask(result_pre, mode="pre")
    assert mask.shape == (5,)

    with pytest.raises(ValueError, match="No post-check measurements"):
        selector_pre.compute_mask(result_pre, mode="post")

    with pytest.raises(ValueError, match="No post-check measurements"):
        selector_pre.compute_mask(result_pre, mode="both")


def test_validation_errors_pre_check():
    """Malformed pre-check registers are rejected when building the selector/summary."""
    qreg = QuantumRegister(3, "q")
    creg = ClassicalRegister(3, "alpha")
    creg_pre = ClassicalRegister(3, "alpha_pre")

    circuit = QuantumCircuit(qreg, creg, creg_pre)
    circuit.measure(qreg, creg_pre)
    circuit.barrier()
    circuit.measure(qreg, creg)

    selector = PostSelector.from_circuit(circuit, [(0, 1), (1, 2)], pre_check_suffix="_pre")

    result_missing = {"alpha": np.zeros((5, 3), dtype=bool)}
    with pytest.raises(ValueError, match="Result does not contain creg 'alpha_pre'"):
        selector.compute_mask(result_missing, mode="pre")

    result_inconsistent = {
        "alpha": np.zeros((5, 3), dtype=bool),
        "alpha_pre": np.zeros((3, 3), dtype=bool),  # mismatched shape
    }
    with pytest.raises(ValueError, match="arrays of inconsistent shapes"):
        selector.compute_mask(result_inconsistent, mode="pre")


def test_post_selector_forwards_spectator_cregs():
    """``PostSelector.from_circuit`` forwards ``spectator_cregs`` to the summary."""
    qreg = QuantumRegister(3, "q")
    creg_data = ClassicalRegister(2, "c")
    creg_data_ps = ClassicalRegister(2, "c_ps")
    creg_spec = ClassicalRegister(1, "spec")
    creg_spec_ps = ClassicalRegister(1, "spec_ps")
    circuit = QuantumCircuit(qreg, creg_data, creg_data_ps, creg_spec, creg_spec_ps)
    circuit.measure(qreg[0], creg_data[0])
    circuit.measure(qreg[1], creg_data[1])
    circuit.measure(qreg[2], creg_spec[0])
    circuit.measure(qreg[0], creg_data_ps[0])
    circuit.measure(qreg[1], creg_data_ps[1])
    circuit.measure(qreg[2], creg_spec_ps[0])

    # Default: ``spec`` recognised as spectator; empty list opts out.
    assert PostSelector.from_circuit(circuit, [(0, 1), (1, 2)]).summary.spectator_cregs == {"spec"}
    selector_off = PostSelector.from_circuit(circuit, [(0, 1), (1, 2)], spectator_cregs=[])
    assert selector_off.summary.spectator_cregs == set()
