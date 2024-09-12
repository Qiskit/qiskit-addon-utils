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

"""Tests for slice re-combining."""

import unittest

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit_addon_utils.slicing import combine_slices


class TestCombineSlices(unittest.TestCase):
    def setUp(self):
        qc = QuantumCircuit(3)
        qc.rxx(np.pi / 3, 0, 1)
        qc.ryy(np.pi / 6, 1, 2)
        qc.rzz(np.pi / 9, 2, 0)
        self.qc = qc

    def test_combine_slices(self):
        with self.subTest("Empty"):
            slices = []
            out = combine_slices(slices)
            self.assertEqual(None, out)
        with self.subTest("Circuit with barriers"):
            slices = [self.qc, self.qc, self.qc]
            circ = combine_slices(slices, include_barriers=True)
            op_counts = circuit_to_dag(circ).count_ops(recurse=False)
            self.assertEqual(3, op_counts["rxx"])
            self.assertEqual(3, op_counts["ryy"])
            self.assertEqual(3, op_counts["rzz"])
            self.assertEqual(2, op_counts["barrier"])
        with self.subTest("Circuit no barriers"):
            slices = [self.qc, self.qc, self.qc]
            circ = combine_slices(slices)
            op_counts = circuit_to_dag(circ).count_ops(recurse=False)
            self.assertEqual(3, op_counts["rxx"])
            self.assertEqual(3, op_counts["ryy"])
            self.assertEqual(3, op_counts["rzz"])
            self.assertEqual(9, len(circ.data))
        with self.subTest("Mismatching slices"):
            diff_qc = QuantumCircuit(2)
            diff_qc.rxx(np.pi / 3, 0, 1)
            slices = [self.qc, self.qc, diff_qc]
            with pytest.raises(ValueError) as e_info:
                combine_slices(slices)
            self.assertEqual(
                (
                    "All slices must be defined on the same number of qubits. "
                    "slices[0] contains 3 qubits, but slices[2] contains "
                    "2 qubits."
                ),
                e_info.value.args[0],
            )
