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
from qiskit_addon_utils.slicing.transpiler.passes import CollectOpColor


class TestCollectOpColor(unittest.TestCase):
    """Tests for CollectOpColor transpiler pass"""

    def setUp(self):
        super().setUp()
        self.colors = {
            0: {(2, 1), (5, 4), (7, 6)},
            1: {(2, 3), (6, 5)},
            2: {(0, 1), (4, 3)},
        }

    def test_pass_empty_circuit(self):
        """Test pass on an empty circuit."""
        circuit = QuantumCircuit(5)
        pm = PassManager([CollectOpColor("color_0", {(0, 1)})])
        tqc = pm.run(circuit)
        self.assertEqual(tqc, circuit)

    def test_basic(self):
        """A basic test case."""
        qc = QuantumCircuit(8)
        qc.rxx(
            1.0,
            # NOTE: the color assignments shown below
            # 2, 1, 0, 0, 0, 2, 1, 2, 1, 0, 0, 0, 2, 1
            [0, 2, 4, 6, 1, 3, 5, 0, 2, 4, 6, 1, 3, 5],
            [1, 3, 5, 7, 2, 4, 6, 1, 3, 5, 7, 2, 4, 6],
        )
        pm = PassManager(
            [CollectOpColor(f"color_{color}", edges) for color, edges in self.colors.items()]
        )
        tcircuit = pm.run(qc)
        count_ops = tcircuit.count_ops()
        self.assertEqual(list(count_ops.keys()), ["slice_op"])
        self.assertEqual(8, count_ops["slice_op"])
