# This code is a Qiskit project.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import unittest
from pathlib import Path

from qiskit import QuantumCircuit, transpile
from qiskit.qasm3 import load
from qiskit.transpiler import PassManager
from qiskit_addon_utils.slicing.transpiler.passes import CollectOpType


class TestCollectOpType(unittest.TestCase):
    """Tests for CollectOpType transpiler pass"""

    def setUp(self):
        super().setUp()
        self.basis_gates = ["ecr", "rz", "sx", "id"]
        circ_path = (
            Path(__file__).parent.parent.parent.parent / "qasm_circuits" / "circuit_10101042.qasm"
        ).resolve()
        self.circuit = load(circ_path)

    def test_pass_empty_circuit(self):
        """Test pass on an empty circuit."""
        circuit = QuantumCircuit(5)
        pm = PassManager([CollectOpType("ecr")])
        tqc = pm.run(circuit)
        self.assertEqual(tqc, circuit)

    def test_random_circuit(self):
        """Test pass on a random circuit"""
        with self.subTest("ECR slices"):
            # First transpile random circuit to desired basis gates
            circuit = transpile(self.circuit, basis_gates=self.basis_gates)
            pm = PassManager([CollectOpType("ecr")])
            circuit_ops = circuit.count_ops()
            tcircuit = pm.run(circuit)
            count_ops = tcircuit.count_ops()
            self.assertEqual(sorted(count_ops.keys()), sorted(["slice_op", "rz", "sx"]))
            self.assertEqual(count_ops["rz"], circuit_ops["rz"])
            self.assertEqual(count_ops["sx"], circuit_ops["sx"])
        with self.subTest("ECR and RZ slices"):
            # First transpile random circuit to desired basis gates
            circuit = transpile(self.circuit, basis_gates=self.basis_gates)
            pm = PassManager([CollectOpType("ecr"), CollectOpType("rz")])
            circuit_ops = circuit.count_ops()
            tcircuit = pm.run(circuit)
            count_ops = tcircuit.count_ops()
            self.assertEqual(sorted(count_ops.keys()), sorted(["slice_op", "sx"]))
            self.assertEqual(count_ops["sx"], circuit_ops["sx"])
        with self.subTest("Slices for all gate types"):
            # First transpile random circuit to desired basis gates
            circuit = transpile(self.circuit, basis_gates=self.basis_gates)
            pm = PassManager([CollectOpType(op_name) for op_name in self.basis_gates])
            circuit_ops = circuit.count_ops()
            tcircuit = pm.run(circuit)
            count_ops = tcircuit.count_ops()
            self.assertEqual(sorted(count_ops.keys()), sorted(["slice_op"]))
