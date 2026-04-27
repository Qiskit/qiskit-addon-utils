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
"""Tests for ``PostSelectionSummary``."""

from copy import deepcopy

import pytest
from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_addon_utils.noise_management.post_selection import PostSelectionSummary


def test_init():
    """Test the init constructors."""
    post_selection_suffix = "my_suffix"
    summary = PostSelectionSummary(
        primary_cregs := {"alpha", "beta"},
        measure_map := {
            0: ("alpha", 0),
            1: ("alpha", 1),
            2: ("alpha", 2),
            3: ("beta", 0),
            4: ("beta", 1),
        },
        edges := {frozenset(pair) for pair in [(0, 1), (1, 2), (2, 3), (3, 4)]},
        post_selection_suffix=post_selection_suffix,
    )

    assert summary.primary_cregs == primary_cregs
    assert summary.measure_map == measure_map
    assert summary.edges == edges
    assert summary.post_selection_suffix == post_selection_suffix


def test_constructor_from_circuit():
    """Test the constructor from circuit."""
    qreg = QuantumRegister(5, "q")
    creg0 = ClassicalRegister(3, "alpha")
    creg1 = ClassicalRegister(2, "beta")
    creg0_ps = ClassicalRegister(3, "alpha_ps")
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

    post_selection_suffix = "_ps"
    summary = PostSelectionSummary.from_circuit(
        circuit,
        coupling_map := [(0, 1), (1, 2), (2, 3), (3, 4)],
        post_selection_suffix=post_selection_suffix,
    )

    assert summary.primary_cregs == {"alpha", "beta"}
    assert summary.measure_map == {
        0: ("alpha", 0),
        1: ("alpha", 1),
        2: ("alpha", 2),
        3: ("beta", 0),
        4: ("beta", 1),
    }
    assert summary.edges == {frozenset(pair) for pair in coupling_map}
    assert summary.post_selection_suffix == post_selection_suffix


def test_eq():
    """Test equality between summaries."""
    post_selection_suffix = "my_suffix"
    summary = PostSelectionSummary(
        primary_cregs := {"alpha", "beta"},
        measure_map := {
            0: ("alpha", 0),
            1: ("alpha", 1),
            2: ("alpha", 2),
            3: ("beta", 0),
            4: ("beta", 1),
        },
        edges := {frozenset(pair) for pair in [(0, 1), (1, 2), (2, 3), (3, 4)]},
        post_selection_suffix=post_selection_suffix,
    )

    assert summary == deepcopy(summary)
    assert summary != 3
    assert summary != PostSelectionSummary(
        {"alpha"}, measure_map, edges, post_selection_suffix=post_selection_suffix
    )
    assert summary != PostSelectionSummary(
        primary_cregs, {0: ("alpha", 0)}, edges, post_selection_suffix=post_selection_suffix
    )
    assert summary != PostSelectionSummary(
        primary_cregs, measure_map, {frozenset([0, 1])}, post_selection_suffix=post_selection_suffix
    )
    assert summary != PostSelectionSummary(
        primary_cregs, measure_map, edges, post_selection_suffix="ciao"
    )


def test_invalid_cregs_raises():
    """Test that the constructor from circuits raises when the cregs are invalid."""
    qreg = QuantumRegister(5, "q")
    creg0 = ClassicalRegister(3, "alpha")
    creg1 = ClassicalRegister(2, "beta")
    creg0_ps = ClassicalRegister(3, "alpha_ps")

    circuit = QuantumCircuit(qreg, creg0, creg0_ps, creg1)
    with pytest.raises(ValueError):
        PostSelectionSummary.from_circuit(circuit, [])

    creg1_ps_invalid = ClassicalRegister(3, "beta_ps")
    circuit = QuantumCircuit(qreg, creg1, creg1_ps_invalid)
    with pytest.raises(ValueError):
        PostSelectionSummary.from_circuit(circuit, [])


def test_invalid_measure_maps_raises():
    """Test that the constructor from circuits raises when the cregs are invalid."""
    qreg = QuantumRegister(3, "q")
    creg0 = ClassicalRegister(3, "alpha")
    creg0_ps = ClassicalRegister(3, "alpha_ps")

    circuit = QuantumCircuit(qreg, creg0, creg0_ps)
    circuit.measure(qreg[0], creg0[0])
    circuit.measure(qreg, creg0_ps)
    with pytest.raises(ValueError, match="1 measurements and 3 post selection measurements"):
        PostSelectionSummary.from_circuit(circuit, [])

    circuit = QuantumCircuit(qreg, creg0, creg0_ps)
    circuit.measure(qreg[0], creg0[0])
    circuit.measure(qreg[1], creg0_ps[0])
    with pytest.raises(ValueError, match="Missing post selection measurement on qubit 0"):
        PostSelectionSummary.from_circuit(circuit, [])

    circuit = QuantumCircuit(qreg, creg0, creg0_ps)
    circuit.measure(qreg[0], creg0[0])
    circuit.measure(qreg[0], creg0_ps[1])
    with pytest.raises(ValueError, match="Expected measurement on qubit 0 writing to bit 0"):
        PostSelectionSummary.from_circuit(circuit, [])


def test_pre_selection_from_circuit():
    """Test the constructor from circuit with pre-selection measurements."""
    qreg = QuantumRegister(5, "q")
    creg0 = ClassicalRegister(3, "alpha")
    creg1 = ClassicalRegister(2, "beta")
    creg0_pre = ClassicalRegister(3, "alpha_pre")
    creg1_pre = ClassicalRegister(2, "beta_pre")

    circuit = QuantumCircuit(qreg, creg0, creg0_pre, creg1, creg1_pre)
    # Pre-selection measurements at the start
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

    pre_selection_suffix = "_pre"
    summary = PostSelectionSummary.from_circuit(
        circuit,
        coupling_map := [(0, 1), (1, 2), (2, 3), (3, 4)],
        pre_selection_suffix=pre_selection_suffix,
    )

    assert summary.primary_cregs == {"alpha", "beta"}
    assert summary.measure_map == {
        0: ("alpha", 0),
        1: ("alpha", 1),
        2: ("alpha", 2),
        3: ("beta", 0),
        4: ("beta", 1),
    }
    assert summary.measure_map_pre == {
        0: ("alpha_pre", 0),
        1: ("alpha_pre", 1),
        2: ("alpha_pre", 2),
        3: ("beta_pre", 0),
        4: ("beta_pre", 1),
    }
    assert summary.measure_map_ps == {}
    assert summary.edges == {frozenset(pair) for pair in coupling_map}
    assert summary.pre_selection_suffix == pre_selection_suffix


def test_combined_pre_and_post_selection_from_circuit():
    """Test the constructor from circuit with both pre and post-selection measurements."""
    qreg = QuantumRegister(3, "q")
    creg = ClassicalRegister(3, "alpha")
    creg_pre = ClassicalRegister(3, "alpha_pre")
    creg_ps = ClassicalRegister(3, "alpha_ps")

    circuit = QuantumCircuit(qreg, creg, creg_pre, creg_ps)
    # Pre-selection measurements at the start
    circuit.measure(qreg, creg_pre)
    circuit.barrier()
    # Some gates here...
    circuit.barrier()
    # Terminal measurements
    circuit.measure(qreg, creg)
    # Post-selection measurements
    circuit.measure(qreg, creg_ps)

    summary = PostSelectionSummary.from_circuit(
        circuit,
        coupling_map := [(0, 1), (1, 2)],
        post_selection_suffix="_ps",
        pre_selection_suffix="_pre",
    )

    assert summary.primary_cregs == {"alpha"}
    assert summary.measure_map == {
        0: ("alpha", 0),
        1: ("alpha", 1),
        2: ("alpha", 2),
    }
    assert summary.measure_map_ps == {
        0: ("alpha_ps", 0),
        1: ("alpha_ps", 1),
        2: ("alpha_ps", 2),
    }
    assert summary.measure_map_pre == {
        0: ("alpha_pre", 0),
        1: ("alpha_pre", 1),
        2: ("alpha_pre", 2),
    }
    assert summary.edges == {frozenset(pair) for pair in coupling_map}
    assert summary.post_selection_suffix == "_ps"
    assert summary.pre_selection_suffix == "_pre"


def test_pre_selection_only_partial_measurements():
    """Test pre-selection with only some qubits having terminal measurements (lenient mode)."""
    qreg = QuantumRegister(3, "q")
    creg = ClassicalRegister(2, "alpha")  # Only 2 bits
    creg_pre = ClassicalRegister(2, "alpha_pre")

    circuit = QuantumCircuit(qreg, creg, creg_pre)
    # Pre-selection measurements on 2 qubits
    circuit.measure(qreg[0], creg_pre[0])
    circuit.measure(qreg[1], creg_pre[1])
    circuit.barrier()
    # Terminal measurements on same 2 qubits
    circuit.measure(qreg[0], creg[0])
    circuit.measure(qreg[1], creg[1])

    summary = PostSelectionSummary.from_circuit(
        circuit,
        [(0, 1), (1, 2)],
        pre_selection_suffix="_pre",
    )

    assert summary.primary_cregs == {"alpha"}
    assert summary.measure_map == {
        0: ("alpha", 0),
        1: ("alpha", 1),
    }
    assert summary.measure_map_pre == {
        0: ("alpha_pre", 0),
        1: ("alpha_pre", 1),
    }
    assert summary.measure_map_ps == {}


def test_init_with_all_measure_maps():
    """Test initialization with all three measure maps."""
    summary = PostSelectionSummary(
        primary_cregs := {"alpha"},
        measure_map := {0: ("alpha", 0), 1: ("alpha", 1)},
        edges := {frozenset([0, 1])},
        measure_map_ps={0: ("alpha_ps", 0), 1: ("alpha_ps", 1)},
        measure_map_pre={0: ("alpha_pre", 0), 1: ("alpha_pre", 1)},
        post_selection_suffix="_ps",
        pre_selection_suffix="_pre",
    )

    assert summary.primary_cregs == primary_cregs
    assert summary.measure_map == measure_map
    assert summary.measure_map_ps == {0: ("alpha_ps", 0), 1: ("alpha_ps", 1)}
    assert summary.measure_map_pre == {0: ("alpha_pre", 0), 1: ("alpha_pre", 1)}
    assert summary.edges == edges
    assert summary.post_selection_suffix == "_ps"
    assert summary.pre_selection_suffix == "_pre"


def test_eq_with_all_properties():
    """Test equality with all properties including pre and post selection."""
    summary1 = PostSelectionSummary(
        {"alpha"},
        {0: ("alpha", 0)},
        {frozenset([0, 1])},
        measure_map_ps={0: ("alpha_ps", 0)},
        measure_map_pre={0: ("alpha_pre", 0)},
        post_selection_suffix="_ps",
        pre_selection_suffix="_pre",
    )

    summary2 = PostSelectionSummary(
        {"alpha"},
        {0: ("alpha", 0)},
        {frozenset([0, 1])},
        measure_map_ps={0: ("alpha_ps", 0)},
        measure_map_pre={0: ("alpha_pre", 0)},
        post_selection_suffix="_ps",
        pre_selection_suffix="_pre",
    )

    assert summary1 == summary2

    # Different measure_map_ps
    summary3 = PostSelectionSummary(
        {"alpha"},
        {0: ("alpha", 0)},
        {frozenset([0, 1])},
        measure_map_ps={0: ("alpha_ps", 1)},  # Different bit
        measure_map_pre={0: ("alpha_pre", 0)},
        post_selection_suffix="_ps",
        pre_selection_suffix="_pre",
    )
    assert summary1 != summary3

    # Different measure_map_pre
    summary4 = PostSelectionSummary(
        {"alpha"},
        {0: ("alpha", 0)},
        {frozenset([0, 1])},
        measure_map_ps={0: ("alpha_ps", 0)},
        measure_map_pre={},  # Empty
        post_selection_suffix="_ps",
        pre_selection_suffix="_pre",
    )
    assert summary1 != summary4

    # Different pre_selection_suffix
    summary5 = PostSelectionSummary(
        {"alpha"},
        {0: ("alpha", 0)},
        {frozenset([0, 1])},
        measure_map_ps={0: ("alpha_ps", 0)},
        measure_map_pre={0: ("alpha_pre", 0)},
        post_selection_suffix="_ps",
        pre_selection_suffix="_different",
    )
    assert summary1 != summary5


def test_pre_selection_invalid_measure_map_raises():
    """Test that pre-selection validation raises when measure maps don't match."""
    qreg = QuantumRegister(3, "q")
    creg = ClassicalRegister(3, "alpha")
    creg_pre = ClassicalRegister(3, "alpha_pre")

    # Test case: pre-selection measurement writes to wrong bit
    circuit = QuantumCircuit(qreg, creg, creg_pre)
    circuit.measure(qreg[0], creg_pre[1])  # Wrong bit index
    circuit.barrier()
    circuit.measure(qreg[0], creg[0])

    with pytest.raises(
        ValueError, match="Pre selection measurement on qubit 0 writes to bit 1 of creg alpha_pre"
    ):
        PostSelectionSummary.from_circuit(circuit, [], pre_selection_suffix="_pre")

    # Test case: pre-selection measurement writes to wrong register (different base name)
    # This tests the case where the pre-selection register name doesn't match the primary register
    creg_beta = ClassicalRegister(3, "beta")
    creg_alpha_pre2 = ClassicalRegister(3, "alpha_pre")
    creg_beta_pre = ClassicalRegister(3, "beta_pre")
    circuit2 = QuantumCircuit(qreg, creg, creg_beta, creg_alpha_pre2, creg_beta_pre)
    circuit2.measure(qreg[0], creg_beta_pre[0])  # Measuring to beta_pre instead of alpha_pre
    circuit2.barrier()
    circuit2.measure(qreg[0], creg[0])  # Primary measurement to alpha
    circuit2.measure(qreg[1], creg_beta[0])  # Another primary measurement to beta

    with pytest.raises(
        ValueError, match="Pre selection measurement on qubit 0 writes to bit 0 of creg beta_pre"
    ):
        PostSelectionSummary.from_circuit(circuit2, [], pre_selection_suffix="_pre")
