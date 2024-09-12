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

"""Tests for depth slicing."""

import unittest

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit_addon_utils.slicing import slice_by_depth


class TestSliceByDepth(unittest.TestCase):
    def setUp(self):
        qc = QuantumCircuit(3)
        qc.rxx(np.pi / 3, 0, 1)
        qc.ryy(np.pi / 6, 1, 2)
        qc.rzz(np.pi / 9, 2, 0)
        self.qc = qc

    def test_slice_by_depth(self):
        with self.subTest("Depth-1"):
            slices = slice_by_depth(self.qc, 1)
            self.assertEqual(3, len(slices))
            gates = ["rxx", "ryy", "rzz"]
            for i, slice_ in enumerate(slices):
                op_counts = circuit_to_dag(slice_).count_ops(recurse=False)
                self.assertEqual(1, len(slice_.data))
                self.assertEqual(1, op_counts[gates[i]])
        with self.subTest("Barrier"):
            qc = QuantumCircuit(2)
            qc.h(0)
            qc.barrier()
            qc.x(1)
            qc.cx(0, 1)
            qc.h(0)

            slices = slice_by_depth(qc, 1)
            self.assertEqual(5, len(slices))
            gates = [{"h"}, {"barrier"}, {"x"}, {"cx"}, {"h"}]
            for i, slice_ in enumerate(slices):
                op_counts = circuit_to_dag(slice_).count_ops(recurse=False)
                self.assertEqual(len(gates[i]), len(slice_.data))
                self.assertEqual(set(op_counts.keys()), gates[i])

            # Depth 2
            slices = slice_by_depth(qc, 2)
            self.assertEqual(3, len(slices))
            gates = [{"h", "barrier"}, {"x", "cx"}, {"h"}]
            for i, slice_ in enumerate(slices):
                op_counts = circuit_to_dag(slice_).count_ops(recurse=False)
                self.assertEqual(len(gates[i]), len(slice_.data))
                self.assertEqual(set(op_counts.keys()), gates[i])
        with self.subTest("Invalid max_slice_depth"):
            with pytest.raises(ValueError) as e_info:
                slice_by_depth(self.qc, 0)
            self.assertEqual("max_slice_depth must be > 0.", e_info.value.args[0])
