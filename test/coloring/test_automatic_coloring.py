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

from qiskit_addon_utils.coloring import auto_color_edges


class TestAutomaticColoring(unittest.TestCase):
    def test_auto_color_edges(self):
        with self.subTest("Basic two colors"):
            edges = ((2, 3), (0, 1), (3, 4), (1, 2))
            edge_coloring = auto_color_edges(edges)
            target_coloring = {
                (0, 1): 1,
                (1, 2): 0,
                (2, 3): 1,
                (3, 4): 0,
            }
            self.assertEqual(target_coloring, edge_coloring)
        with self.subTest("Three colors"):
            edges = ((2, 3), (0, 1), (2, 14), (3, 4), (1, 2))
            edge_coloring = auto_color_edges(edges)
            target_coloring = {
                (0, 1): 1,
                (1, 2): 0,
                (2, 3): 1,
                (2, 14): 2,
                (3, 4): 0,
            }
            self.assertEqual(target_coloring, edge_coloring)
