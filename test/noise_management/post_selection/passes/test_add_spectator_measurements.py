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
"""Test the `AddSpectatorMeasures` pass."""

import unittest

import pytest
from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.dagcircuit.exceptions import DAGCircuitError
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passmanager import PassManager
from qiskit_addon_utils.noise_management.post_selection.transpiler.passes import (
    AddSpectatorMeasures,
)


class TestAddSpectatorMeasures(unittest.TestCase):
    """Tests the AddSpectatorMeasures pass."""

    # pylint: disable=no-self-use

    def setUp(self):
        """Setup."""
        super().setUp()

        # The same coupling map as FakeBrisbane, but trimmed down
        self.coupling_map = [
            (6, 7),
            (7, 8),
            (8, 9),
            (11, 12),
            (12, 17),
            (13, 12),
            (16, 8),
            (17, 30),
            (26, 25),
            (27, 26),
            (28, 29),
            (28, 35),
        ]
        self.num_qubits = 127
        self.pm = PassManager([AddSpectatorMeasures(self.coupling_map)])

    def test_empty_circuit(self):
        """Test the pass on an empty circuit."""
        circuit = QuantumCircuit(1)
        self.assertEqual(circuit, self.pm.run(circuit))

    def test_circuit_with_final_layer_of_measurements(self):
        """Test the pass on a circuit with a final layer of measurements."""
        circuit_qubits = [7, 17, 27, 28]
        spectator_qubits = [6, 8, 12, 26, 29, 30, 35]

        qreg = QuantumRegister(self.num_qubits, "q")
        creg = ClassicalRegister(len(circuit_qubits), "c")
        creg_spec = ClassicalRegister(len(spectator_qubits), "spec")

        circuit = QuantumCircuit(qreg, creg)
        circuit.h(7)
        circuit.cz(7, 17)
        circuit.cz(17, 27)
        circuit.barrier(17)
        circuit.cz(27, 28)
        circuit.measure(7, creg[0])
        circuit.measure(17, creg[1])
        circuit.measure(27, creg[2])
        circuit.measure(28, creg[3])

        expected_circuit = QuantumCircuit(qreg, creg, creg_spec)
        expected_circuit.h(7)
        expected_circuit.cz(7, 17)
        expected_circuit.cz(17, 27)
        expected_circuit.barrier(17)
        expected_circuit.cz(27, 28)
        for idx, qubit in enumerate(circuit_qubits):
            expected_circuit.measure(qubit, creg[idx])
        expected_circuit.barrier(circuit_qubits + spectator_qubits)
        for idx, spec_qubit in enumerate(spectator_qubits):
            expected_circuit.measure(spec_qubit, creg_spec[idx])

        self.assertEqual(expected_circuit, self.pm.run(circuit))

    def test_circuit_with_measurements_in_a_box(self):
        """Test the pass on a circuit with measurements inside a box."""
        circuit_qubits = [7, 17, 27, 28]
        spectator_qubits = [6, 8, 12, 26, 29, 30, 35]

        qreg = QuantumRegister(self.num_qubits, "q")
        creg = ClassicalRegister(len(circuit_qubits), "c")
        creg_spec = ClassicalRegister(len(spectator_qubits), "spec")

        circuit = QuantumCircuit(qreg, creg)
        circuit.h(7)
        circuit.cz(7, 17)
        with circuit.box():
            circuit.cz(17, 27)
        circuit.cz(27, 28)
        circuit.measure(7, creg[0])
        circuit.measure(17, creg[1])
        with circuit.box():
            circuit.measure(27, creg[2])
            circuit.measure(28, creg[3])

        expected_circuit = QuantumCircuit(qreg, creg, creg_spec)
        expected_circuit.h(7)
        expected_circuit.cz(7, 17)
        with expected_circuit.box():
            expected_circuit.cz(17, 27)
        expected_circuit.cz(27, 28)
        expected_circuit.measure(7, creg[0])
        expected_circuit.measure(17, creg[1])
        with expected_circuit.box():
            expected_circuit.measure(27, creg[2])
            expected_circuit.measure(28, creg[3])
        expected_circuit.barrier(circuit_qubits + spectator_qubits)
        for idx, spec_qubit in enumerate(spectator_qubits):
            expected_circuit.measure(spec_qubit, creg_spec[idx])

        self.assertEqual(expected_circuit, self.pm.run(circuit))

    def test_circuit_with_mid_circuit_measurements(self):
        """Test the pass on a circuit with mid-circuit measurements."""
        qreg = QuantumRegister(self.num_qubits, "q")
        creg = ClassicalRegister(2, "c")
        creg_spec = ClassicalRegister(2, "spec")

        circuit = QuantumCircuit(qreg, creg)
        circuit.measure(7, creg[0])
        circuit.h(7)
        circuit.measure(7, creg[1])

        expected_circuit = QuantumCircuit(qreg, creg, creg_spec)
        expected_circuit.measure(7, creg[0])
        expected_circuit.h(7)
        expected_circuit.measure(7, creg[1])
        expected_circuit.barrier([7, 6, 8])
        expected_circuit.measure(6, creg_spec[0])
        expected_circuit.measure(8, creg_spec[1])

        self.assertEqual(expected_circuit, self.pm.run(circuit))

    def test_if_else(self):
        """Test the pass for circuits with if/else statements."""
        qreg = QuantumRegister(5, "q")
        creg = ClassicalRegister(2, "c")
        creg_spec = ClassicalRegister(3, "spec")

        circuit = QuantumCircuit(qreg, creg)
        circuit.measure(0, creg[0])
        with circuit.if_test((creg[0], 0)) as else_:
            circuit.measure(1, creg[1])
        with else_:
            circuit.measure(2, creg[1])
        with circuit.if_test((creg[0], 0)) as else_:
            circuit.measure(1, creg[1])
        with else_:
            circuit.x(1)
            circuit.measure(1, creg[1])

        expected_circuit = QuantumCircuit(qreg, creg, creg_spec)
        expected_circuit.measure(0, creg[0])
        with expected_circuit.if_test((creg[0], 0)) as else_:
            expected_circuit.measure(1, creg[1])
        with else_:
            expected_circuit.measure(2, creg[1])
        with expected_circuit.if_test((creg[0], 0)) as else_:
            expected_circuit.measure(1, creg[1])
        with else_:
            expected_circuit.x(1)
            expected_circuit.measure(1, creg[1])
        expected_circuit.barrier([0, 1, 2, 3, 4])
        expected_circuit.measure(2, creg_spec[0])
        expected_circuit.measure(3, creg_spec[1])
        expected_circuit.measure(4, creg_spec[2])

        pm = PassManager([AddSpectatorMeasures(coupling_map=[(1, 3), (2, 4)])])
        self.assertEqual(expected_circuit, pm.run(circuit))

    def test_include_unmeasured(self):
        """Test the ``include_unmeasured`` argument of the pass."""
        qreg = QuantumRegister(self.num_qubits, "q")
        creg = ClassicalRegister(2, "c")
        creg_spec = ClassicalRegister(1, "spec")

        circuit = QuantumCircuit(qreg, creg)
        circuit.measure(7, creg[0])
        circuit.measure(17, creg[1])
        with circuit.box():
            circuit.h(7)

        expected_circuit_true = QuantumCircuit(qreg, creg, creg_spec)
        expected_circuit_true.measure(7, creg[0])
        expected_circuit_true.measure(17, creg[1])
        with expected_circuit_true.box():
            expected_circuit_true.h(7)
        expected_circuit_true.barrier([7, 17])
        expected_circuit_true.measure(7, creg_spec[0])

        pm = PassManager([AddSpectatorMeasures(coupling_map=[], include_unmeasured=True)])
        self.assertEqual(expected_circuit_true, pm.run(circuit))

        pm = PassManager([AddSpectatorMeasures(coupling_map=[], include_unmeasured=False)])
        self.assertEqual(circuit, pm.run(circuit))

    def test_custom_spectator_creg_name(self):
        """Test the pass for a custom register name."""
        qreg = QuantumRegister(self.num_qubits, "q")
        creg = ClassicalRegister(4, "c")

        circuit = QuantumCircuit(qreg, creg)
        circuit.h(7)
        circuit.cz(7, 17)
        circuit.cz(17, 27)
        circuit.cz(27, 28)
        circuit.measure(7, creg[0])
        circuit.measure(17, creg[1])
        circuit.measure(27, creg[2])
        circuit.measure(28, creg[3])

        pm = PassManager([AddSpectatorMeasures(self.coupling_map, spectator_creg_name="my_name")])
        self.assertIn(ClassicalRegister(7, "my_name"), pm.run(circuit).cregs)

    def test_custom_coupling_map(self):
        """Test the pass for a custom coupling map."""
        qreg = QuantumRegister(self.num_qubits, "q")
        creg = ClassicalRegister(4, "c")
        creg_spec = ClassicalRegister(3, "spec")

        circuit = QuantumCircuit(qreg, creg)
        circuit.h(7)
        circuit.cz(7, 17)
        circuit.cz(17, 27)
        circuit.cz(27, 28)
        circuit.measure(7, creg[0])
        circuit.measure(17, creg[1])
        circuit.measure(27, creg[2])
        circuit.measure(28, creg[3])

        coupling_map = [(7, 17), (7, 1), (7, 89), (90, 7)]

        spectator_qubits = [1, 89, 90]
        expected_circuit = QuantumCircuit(qreg, creg, creg_spec)
        expected_circuit.h(7)
        expected_circuit.cz(7, 17)
        expected_circuit.cz(17, 27)
        expected_circuit.cz(27, 28)
        expected_circuit.measure(7, creg[0])
        expected_circuit.measure(17, creg[1])
        expected_circuit.measure(27, creg[2])
        expected_circuit.measure(28, creg[3])
        expected_circuit.barrier([1, 7, 17, 27, 28, 89, 90])
        expected_circuit.measure(spectator_qubits[0], creg_spec[0])
        expected_circuit.measure(spectator_qubits[1], creg_spec[1])
        expected_circuit.measure(spectator_qubits[2], creg_spec[2])

        pm = PassManager([AddSpectatorMeasures(coupling_map)])
        self.assertEqual(expected_circuit, pm.run(circuit))

    def test_add_barrier_false(self):
        """Test for ``add_barrier=False``."""
        circuit_qubits = [7, 17, 27, 28]
        spectator_qubits = [6, 8, 12, 26, 29, 30, 35]

        qreg = QuantumRegister(self.num_qubits, "q")
        creg = ClassicalRegister(len(circuit_qubits), "c")
        creg_spec = ClassicalRegister(len(spectator_qubits), "spec")

        circuit = QuantumCircuit(qreg, creg)
        circuit.h(7)
        circuit.cz(7, 17)
        circuit.cz(17, 27)
        circuit.barrier(17)
        circuit.cz(27, 28)
        circuit.measure(7, creg[0])
        circuit.measure(17, creg[1])
        circuit.measure(27, creg[2])
        circuit.measure(28, creg[3])

        expected_circuit = QuantumCircuit(qreg, creg, creg_spec)
        expected_circuit.h(7)
        expected_circuit.cz(7, 17)
        expected_circuit.cz(17, 27)
        expected_circuit.barrier(17)
        expected_circuit.cz(27, 28)
        for idx, qubit in enumerate(circuit_qubits):
            expected_circuit.measure(qubit, creg[idx])
        for idx, spec_qubit in enumerate(spectator_qubits):
            expected_circuit.measure(spec_qubit, creg_spec[idx])

        pm = PassManager([AddSpectatorMeasures(self.coupling_map, add_barrier=False)])
        self.assertEqual(expected_circuit, pm.run(circuit))

    def test_conflicting_creg(self):
        """Test that an error is raise when the circuit already contains a register named ``spectator_creg_name``."""
        pm = PassManager([AddSpectatorMeasures(coupling_map=[], spectator_creg_name="my_name")])

        circuit = QuantumCircuit(QuantumRegister(1), ClassicalRegister(1, "my_name"))
        circuit.x(0)
        with pytest.raises(DAGCircuitError, match="duplicate register"):
            pm.run(circuit)

        circuit = QuantumCircuit(1)
        circuit.reset(0)
        with pytest.raises(TranspilerError, match="``'reset'`` is not supported"):
            pm.run(circuit)
