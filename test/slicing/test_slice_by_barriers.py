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

"""Tests for slicing by barriers."""

import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit_addon_utils.slicing import slice_by_barriers


class TestSliceByBarriers(unittest.TestCase):
    def test_slice_by_barriers(self):
        with self.subTest("Basic test"):
            qc = QuantumCircuit(6)
            qc.h(0)
            qc.ry(np.pi / 6, 1)
            qc.barrier()
            qc.cx(0, 1)
            qc.cx(2, 3)
            qc.cx(1, 2)
            qc.barrier()
            qc.rx(np.pi / 6, 5)
            qc.barrier()
            qc.ry(np.pi / 6, 4)
            slices = slice_by_barriers(qc)
            self.assertEqual(4, len(slices))
            targets = (({"h", "ry"}, 2), ({"cx"}, 3), ({"rx"}, 1), ({"ry"}, 1))
            for i, slice_ in enumerate(slices):
                op_names = set()
                for inst in slice_.data:
                    op_names.add(inst.operation.name)
                self.assertEqual(targets[i][0], op_names)
                self.assertEqual(targets[i][1], len(slice_.data))

        with self.subTest("Mini barriers"):
            qc = QuantumCircuit(6)
            qc.h(0)
            qc.barrier(0)
            qc.ry(np.pi / 6, 1)
            qc.barrier()
            qc.cx(0, 1)
            qc.cx(2, 3)
            qc.barrier([1, 2])
            qc.cx(1, 2)
            qc.barrier()
            qc.barrier([4, 5])
            qc.rx(np.pi / 6, 5)
            qc.barrier()
            qc.ry(np.pi / 6, 4)
            slices = slice_by_barriers(qc)
            self.assertEqual(4, len(slices))
            targets = (
                ({"h", "ry", "barrier"}, 3),
                ({"cx", "barrier"}, 4),
                ({"rx", "barrier"}, 2),
                ({"ry"}, 1),
            )
            for i, slice_ in enumerate(slices):
                op_names = set()
                for inst in slice_.data:
                    op_names.add(inst.operation.name)
                self.assertEqual(targets[i][0], op_names)
                self.assertEqual(targets[i][1], len(slice_.data))

        with self.subTest("Empty circuit"):
            qc = QuantumCircuit(6)
            slices = slice_by_barriers(qc)
            self.assertEqual(0, len(slices))
