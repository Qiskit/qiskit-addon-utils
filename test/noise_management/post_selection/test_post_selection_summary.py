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
