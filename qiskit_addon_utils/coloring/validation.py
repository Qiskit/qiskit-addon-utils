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

"""Coloring validation methods."""

from __future__ import annotations

from collections import defaultdict


def is_valid_edge_coloring(coloring: dict[tuple[int, int], int]) -> bool:
    """Check whether an edge coloring scheme is valid.

    An edge coloring is valid if no two edges of the same color share a node.

    Args:
        coloring: A mapping from edges to integer representations of colors.

    Returns:
        A boolean indicating whether the input coloring is valid.
    """
    node_colors: defaultdict[int, set[int]] = defaultdict(set)
    for (n1, n2), color in coloring.items():
        if color in node_colors[n1] or color in node_colors[n2]:
            return False
        node_colors[n1].add(color)
        node_colors[n2].add(color)

    return True
