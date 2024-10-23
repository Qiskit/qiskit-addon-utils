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

"""Automatic coloring methods."""

from __future__ import annotations

from collections.abc import Sequence

import rustworkx as rx


def auto_color_edges(edges: Sequence[tuple[int, int]]) -> dict[tuple[int, int], int]:
    """Color the input edges of an undirected graph such that no two incident edges share a color.

    Args:
        edges: The edges describing an undirected graph.

    Returns:
        A dictionary mapping each edge to an integer representation of a color.
    """
    coupling_graph: rx.PyGraph = rx.PyGraph()
    coupling_graph.extend_from_edge_list(sorted(edges, key=lambda x: min(x)))
    edge_coloring_by_id = rx.graph_greedy_edge_color(coupling_graph)

    coloring_out = {}
    for i, edge in enumerate(coupling_graph.edge_list()):
        coloring_out[edge] = edge_coloring_by_id[i]

    # This function should always return a color for each unique input edge
    assert len(coloring_out.keys()) == len(set(edges))

    return coloring_out
