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

"""Tests for coloring utilities."""

import unittest

from qiskit_addon_utils.coloring import is_valid_edge_coloring


class TestColoringValidation(unittest.TestCase):
    def test_is_valid_edge_coloring(self):
        with self.subTest("Basic edge coloring"):
            edge_coloring = {
                (2, 1): 0,
                (5, 4): 0,
                (7, 6): 0,
                (2, 3): 1,
                (6, 5): 1,
                (0, 1): 2,
                (4, 3): 2,
            }
            self.assertTrue(is_valid_edge_coloring(edge_coloring))
        with self.subTest("Invalid edge coloring"):
            edge_coloring = {(0, 1): 0, (1, 2): 1, (2, 3): 0, (2, 4): 1}
            self.assertFalse(is_valid_edge_coloring(edge_coloring))
