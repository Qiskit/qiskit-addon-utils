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

"""Tests for slicing by gate types."""

import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit_addon_utils.slicing import slice_by_gate_types


class TestSliceByGateTypes(unittest.TestCase):
    def test_slice_by_gate_types(self):
        with self.subTest("Basic test"):
            qc = QuantumCircuit(6)
            qc.h(0)
            qc.ry(np.pi / 6, 1)
            qc.cx(0, 1)
            qc.cx(2, 3)
            qc.cx(1, 2)
            qc.rx(np.pi / 6, 5)
            qc.ry(np.pi / 6, 4)
            slices = slice_by_gate_types(qc)
            self.assertEqual(5, len(slices))
            targets = (("h", 1), ("ry", 2), ("cx", 2), ("cx", 1), ("rx", 1))
            for i, slice_ in enumerate(slices):
                self.assertEqual(targets[i][1], len(slice_.data))
                op_name = slice_.data[0].operation.name
                self.assertEqual(targets[i][0], op_name)
                for inst in slice_.data:
                    self.assertEqual(op_name, inst.operation.name)
        with self.subTest("Arbitrarily scheduled slice."):
            qc = QuantumCircuit(8)
            qc.ry(1.0, 0)
            qc.cy(1, 2)
            qc.cx(3, 4)
            qc.cy(5, 6)
            qc.cy(2, 3)
            qc.cx(4, 5)
            qc.ry(1.0, 7)
            slices = slice_by_gate_types(qc)
            self.assertEqual(5, len(slices))
            targets = (("cx", 1), ("cy", 2), ("cy", 1), ("cx", 1), ("ry", 2))
            for i, slice_ in enumerate(slices):
                self.assertEqual(targets[i][1], len(slice_.data))
                op_name = slice_.data[0].operation.name
                self.assertEqual(targets[i][0], op_name)
                for inst in slice_.data:
                    self.assertEqual(op_name, inst.operation.name)
        with self.subTest("Circuit with classical bits"):
            qc = QuantumCircuit(2)
            qc.x(0)
            qc.x(1)
            qc.cx(0, 1)
            qc.measure_all()  # Add classical bits to the circuit
            slices = slice_by_gate_types(qc)
            self.assertEqual(4, len(slices))
            targets = (("x", 2), ("cx", 1), ("barrier", 1), ("measure", 2))
            for i, slice_ in enumerate(slices):
                self.assertEqual(qc.num_clbits, slice_.num_clbits)
                self.assertEqual(targets[i][1], len(slice_.data))
                op_name = slice_.data[0].operation.name
                self.assertEqual(targets[i][0], op_name)
                for inst in slice_.data:
                    self.assertEqual(op_name, inst.operation.name)
