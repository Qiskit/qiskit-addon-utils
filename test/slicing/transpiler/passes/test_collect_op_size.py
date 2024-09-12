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

from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit_addon_utils.slicing.transpiler.passes import CollectOpSize


class TestCollectOpSize(unittest.TestCase):
    """Tests for CollectOpSize transpiler pass"""

    def test_pass_empty_circuit(self):
        """Test pass on an empty circuit."""
        circuit = QuantumCircuit(5)
        pm = PassManager([CollectOpSize(1)])
        tqc = pm.run(circuit)
        self.assertEqual(tqc, circuit)

    def test_basic(self):
        """Test pass on a basic circuit"""
        circuit = QuantumCircuit(3)
        circuit.x(0)
        circuit.h(1)
        circuit.y(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.x(1)
        circuit.y(2)
        with self.subTest("two-qubit gate slices"):
            pm = PassManager([CollectOpSize(2)])
            circuit_ops = circuit.count_ops()
            tcircuit = pm.run(circuit)
            count_ops = tcircuit.count_ops()
            self.assertEqual(sorted(count_ops.keys()), sorted(["slice_op", "h", "x", "y"]))
            self.assertEqual(count_ops["h"], circuit_ops["h"])
            self.assertEqual(count_ops["x"], circuit_ops["x"])
            self.assertEqual(count_ops["y"], circuit_ops["y"])
        with self.subTest("single-qubit gate slices"):
            pm = PassManager([CollectOpSize(1)])
            circuit_ops = circuit.count_ops()
            tcircuit = pm.run(circuit)
            count_ops = tcircuit.count_ops()
            self.assertEqual(sorted(count_ops.keys()), sorted(["slice_op", "cx"]))
            self.assertEqual(count_ops["cx"], circuit_ops["cx"])
