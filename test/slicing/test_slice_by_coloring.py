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

"""Tests for color slicing."""

import copy
import unittest

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit_addon_utils.slicing import slice_by_coloring


class TestSliceByColoring(unittest.TestCase):
    def setUp(self):
        qc = QuantumCircuit(3)
        qc.rxx(np.pi / 3, 0, 1)
        qc.ryy(np.pi / 6, 1, 2)
        qc.rzz(np.pi / 9, 2, 0)
        self.qc = qc

    def test_slice_by_coloring(self):
        with self.subTest("Triangle"):
            slices = slice_by_coloring(self.qc, {(0, 1): 0, (1, 2): 1, (2, 0): 2})
            self.assertEqual(3, len(slices))
        with self.subTest("Unsupported gate"):
            qc = self.qc.copy()
            qc.ccx(0, 1, 2)
            with pytest.raises(ValueError) as e_info:
                slice_by_coloring(qc, {(0, 1): 0, (1, 2): 1, (2, 0): 2})
            self.assertEqual(
                "Could not assign color to circuit instruction: ccx", e_info.value.args[0]
            )
        with self.subTest("Partial edge coloring"):
            qc = self.qc.copy()
            with pytest.raises(ValueError) as e_info:
                slice_by_coloring(qc, {(0, 1): 0, (1, 2): 1})
            self.assertEqual(
                "Could not assign color to circuit instruction: rzz", e_info.value.args[0]
            )
        with self.subTest("Invalid coloring"):
            with pytest.raises(ValueError) as e_info:
                slice_by_coloring(self.qc, {(0, 1): 0, (1, 2): 1, (2, 0): 0})
            self.assertEqual("The input coloring is invalid.", e_info.value.args[0])
        with self.subTest("Triangle w backward edge"):
            slices = slice_by_coloring(self.qc, {(1, 0): 0, (1, 2): 1, (2, 0): 2})
            self.assertEqual(3, len(slices))
        with self.subTest("Triangle w hadamards"):
            qc = copy.deepcopy(self.qc)
            qc.h(range(0, 3))
            qc.rxx(np.pi / 3, 0, 1)
            qc.ryy(np.pi / 6, 1, 2)
            qc.rzz(np.pi / 9, 2, 0)
            slices = slice_by_coloring(qc, {(0, 1): 0, (1, 2): 1, (2, 0): 2})
            self.assertEqual(7, len(slices))
        with self.subTest("Test straggler colored gate"):
            # This tests a case where the first color-0 gate on an edge
            # must wait to be scheduled on the 2nd layer for that color.
            targets = [
                ({0, 1}, 1),
                ({2, 3}, 1),
                ({1, 2, 4, 5, 6, 7}, 3),
                ({0, 1, 3, 4}, 2),
                ({2, 3, 5, 6}, 2),
                ({1, 2, 4, 5, 6, 7}, 3),
                ({3, 4}, 1),
                ({5, 6}, 1),
            ]
            qc = QuantumCircuit(8)
            qc.rxx(
                1.0,
                [0, 2, 4, 6, 1, 3, 5, 0, 2, 4, 6, 1, 3, 5],
                [1, 3, 5, 7, 2, 4, 6, 1, 3, 5, 7, 2, 4, 6],
            )
            coloring = {
                (2, 1): 0,  # The gate on this edge may not be scheduled in first layer
                (5, 4): 0,
                (7, 6): 0,
                (2, 3): 1,
                (6, 5): 1,
                (0, 1): 2,
                (4, 3): 2,
            }
            slices = slice_by_coloring(qc, coloring=coloring)

            self.assertEqual(len(targets), len(slices))
            for i, (target_qargs, target_num_inst) in enumerate(targets):
                slice_qargs = set(
                    slices[i].find_bit(q).index for inst in slices[i].data for q in inst.qubits
                )
                self.assertEqual(target_qargs, slice_qargs)
                self.assertEqual(target_num_inst, len(slices[i].data))
