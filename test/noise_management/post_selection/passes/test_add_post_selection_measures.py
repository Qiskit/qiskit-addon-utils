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
"""Test the `AddPostSelectionMeasures` pass."""

import numpy as np
import pytest
from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RXGate
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passmanager import PassManager
from qiskit_addon_utils.noise_management.post_selection.transpiler.passes import (
    AddPostSelectionMeasures,
    XSlowGate,
)


def test_empty_circuit():
    """Test the pass on an empty circuit."""
    circuit = QuantumCircuit(1)
    assert circuit == PassManager([AddPostSelectionMeasures()]).run(circuit)


def test_circuit_with_final_layer_of_measurements():
    """Test the pass on a circuit with a final layer of measurements."""
    qreg = QuantumRegister(4, "q")
    creg = ClassicalRegister(3, "c")
    creg_ps = ClassicalRegister(3, "c_ps")

    circuit = QuantumCircuit(qreg, creg)
    circuit.h(0)
    circuit.cz(0, 1)
    circuit.cz(1, 2)
    circuit.cz(2, 3)
    circuit.measure(1, creg[0])
    circuit.measure(2, creg[1])
    circuit.measure(3, creg[2])

    expected_circuit = QuantumCircuit(qreg, creg, creg_ps)
    expected_circuit.h(0)
    expected_circuit.cz(0, 1)
    expected_circuit.cz(1, 2)
    expected_circuit.cz(2, 3)
    expected_circuit.measure(1, creg[0])
    expected_circuit.measure(2, creg[1])
    expected_circuit.measure(3, creg[2])
    expected_circuit.barrier([1, 2, 3])
    expected_circuit.append(XSlowGate(), [1])
    expected_circuit.append(XSlowGate(), [2])
    expected_circuit.append(XSlowGate(), [3])
    expected_circuit.measure(1, creg_ps[0])
    expected_circuit.measure(2, creg_ps[1])
    expected_circuit.measure(3, creg_ps[2])

    assert expected_circuit == PassManager([AddPostSelectionMeasures()]).run(circuit)


def test_circuit_with_measurements_in_a_box():
    """Test the pass on a circuit with measurements inside a box."""
    qreg = QuantumRegister(4, "q")
    creg = ClassicalRegister(3, "c")
    creg_ps = ClassicalRegister(3, "c_ps")

    circuit = QuantumCircuit(qreg, creg)
    circuit.h(0)
    circuit.cz(0, 1)
    circuit.cz(1, 2)
    circuit.cz(2, 3)
    circuit.measure(1, creg[0])
    with circuit.box():
        circuit.measure(2, creg[1])
    circuit.measure(3, creg[2])

    expected_circuit = QuantumCircuit(qreg, creg, creg_ps)
    expected_circuit.h(0)
    expected_circuit.cz(0, 1)
    expected_circuit.cz(1, 2)
    expected_circuit.cz(2, 3)
    expected_circuit.measure(1, creg[0])
    with expected_circuit.box():
        expected_circuit.measure(2, creg[1])
    expected_circuit.measure(3, creg[2])
    expected_circuit.barrier([1, 2, 3])
    expected_circuit.append(XSlowGate(), [1])
    expected_circuit.append(XSlowGate(), [2])
    expected_circuit.append(XSlowGate(), [3])
    expected_circuit.measure(1, creg_ps[0])
    expected_circuit.measure(2, creg_ps[1])
    expected_circuit.measure(3, creg_ps[2])

    assert expected_circuit == PassManager([AddPostSelectionMeasures()]).run(circuit)


def test_if_else():
    """Test the pass for circuits with if/else statements."""
    qreg = QuantumRegister(5, "q")
    creg = ClassicalRegister(2, "c")
    creg_ps = ClassicalRegister(2, "c_ps")

    circuit = QuantumCircuit(qreg, creg)
    circuit.barrier(0)
    circuit.measure(0, creg[0])
    with circuit.if_test((creg[0], 0)) as else_:
        circuit.measure(1, creg[1])
    with else_:
        circuit.measure(2, creg[1])
    with circuit.if_test((creg[0], 0)) as else_:
        circuit.measure(3, creg[1])
    with else_:
        circuit.x(1)
        circuit.measure(3, creg[1])

    expected_circuit = QuantumCircuit(qreg, creg, creg_ps)
    expected_circuit.barrier(0)
    expected_circuit.measure(0, creg[0])
    with expected_circuit.if_test((creg[0], 0)) as else_:
        expected_circuit.measure(1, creg[1])
    with else_:
        expected_circuit.measure(2, creg[1])
    with expected_circuit.if_test((creg[0], 0)) as else_:
        expected_circuit.measure(3, creg[1])
    with else_:
        expected_circuit.x(1)
        expected_circuit.measure(3, creg[1])
    expected_circuit.barrier([0, 3])
    expected_circuit.append(XSlowGate(), [0])
    expected_circuit.append(XSlowGate(), [3])
    expected_circuit.measure(0, creg_ps[0])
    expected_circuit.measure(3, creg_ps[1])

    assert expected_circuit == PassManager([AddPostSelectionMeasures()]).run(circuit)


def test_circuit_with_mid_circuit_measurements():
    """Test the pass on a circuit with mid-circuit measurements."""
    qreg = QuantumRegister(3, "q")
    creg = ClassicalRegister(2, "c")
    creg_ps = ClassicalRegister(2, "c_ps")

    circuit = QuantumCircuit(qreg, creg)
    circuit.measure(1, 0)
    circuit.measure(2, 1)
    with circuit.box():
        circuit.x(1)

    expected_circuit = QuantumCircuit(qreg, creg, creg_ps)
    expected_circuit.measure(1, creg[0])
    expected_circuit.measure(2, creg[1])
    with expected_circuit.box():
        expected_circuit.x(1)
    expected_circuit.barrier([2])
    expected_circuit.append(XSlowGate(), [2])
    expected_circuit.measure(2, creg_ps[1])

    assert expected_circuit == PassManager([AddPostSelectionMeasures()]).run(circuit)


def test_circuit_with_multiple_cregs():
    """Test for a circuit with multiple cregs."""
    qreg = QuantumRegister(4, "q")
    creg1 = ClassicalRegister(1, "c1")
    creg2 = ClassicalRegister(2, "c2")
    creg1_ps = ClassicalRegister(1, "c1_ps")
    creg2_ps = ClassicalRegister(2, "c2_ps")

    circuit = QuantumCircuit(qreg, creg1, creg2)
    circuit.h(0)
    circuit.cz(0, 1)
    circuit.cz(1, 2)
    circuit.cz(2, 3)
    circuit.measure(1, creg1[0])
    circuit.measure(2, creg2[0])
    circuit.measure(3, creg2[1])

    expected_circuit = QuantumCircuit(qreg, creg1, creg2, creg1_ps, creg2_ps)
    expected_circuit.h(0)
    expected_circuit.cz(0, 1)
    expected_circuit.cz(1, 2)
    expected_circuit.cz(2, 3)
    expected_circuit.measure(1, creg1[0])
    expected_circuit.measure(2, creg2[0])
    expected_circuit.measure(3, creg2[1])
    expected_circuit.barrier([1, 2, 3])
    expected_circuit.append(XSlowGate(), [1])
    expected_circuit.append(XSlowGate(), [2])
    expected_circuit.append(XSlowGate(), [3])
    expected_circuit.measure(1, creg1_ps[0])
    expected_circuit.measure(2, creg2_ps[0])
    expected_circuit.measure(3, creg2_ps[1])

    assert expected_circuit == PassManager([AddPostSelectionMeasures()]).run(circuit)


def test_custom_post_selection_suffix():
    """Test the pass for a custom register suffix."""
    qreg = QuantumRegister(1, "q")
    creg = ClassicalRegister(1, "c")
    creg_ps = ClassicalRegister(1, "c_ciao")

    circuit = QuantumCircuit(qreg, creg)
    circuit.measure(0, 0)

    expected_circuit = QuantumCircuit(qreg, creg, creg_ps)
    expected_circuit.measure(0, [creg[0]])
    expected_circuit.barrier([0])
    expected_circuit.append(XSlowGate(), [0])
    expected_circuit.measure(0, creg_ps[0])

    pm = PassManager([AddPostSelectionMeasures(post_selection_suffix="_ciao")])
    assert expected_circuit == pm.run(circuit)


def test_x_pulse_type():
    """Test the pass for non-default X-pulse types."""
    qreg = QuantumRegister(1, "q")
    creg = ClassicalRegister(1, "c")
    creg_ps = ClassicalRegister(1, "c_ps")

    circuit = QuantumCircuit(qreg, creg)
    circuit.measure(0, 0)

    expected_circuit_rx = QuantumCircuit(qreg, creg, creg_ps)
    expected_circuit_rx.measure(0, [creg[0]])
    expected_circuit_rx.barrier([0])
    for _ in range(20):
        expected_circuit_rx.append(RXGate(np.pi / 20), [0])
    expected_circuit_rx.measure(0, creg_ps[0])

    pm = PassManager([AddPostSelectionMeasures(x_pulse_type="rx")])
    assert expected_circuit_rx == pm.run(circuit)


def test_raises():
    """Test that the pass raises."""
    with pytest.raises(ValueError):
        AddPostSelectionMeasures(x_pulse_type="rz")

    pm = PassManager([AddPostSelectionMeasures(x_pulse_type="rx")])
    circuit = QuantumCircuit(1)
    circuit.reset(0)
    with pytest.raises(TranspilerError, match="``'reset'`` is not supported"):
        pm.run(circuit)
