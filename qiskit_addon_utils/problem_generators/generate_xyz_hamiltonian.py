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

"""Utility functions for generating "XYZ model"-like Hamiltonians."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from enum import Enum, auto

import numpy as np
import rustworkx as rx
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import CouplingMap

from qiskit_addon_utils.coloring import auto_color_edges


class PauliOrderStrategy(Enum):
    """Enumeration of different Pauli-orderings.

    When constructing a Hamiltonian on a colored set of edges, the generated Pauli terms can be
    ordered in different ways. This order of terms in the Hamiltonian is preserved during its time
    evolution and, thus, directly impacts the resulting quantum circuit.
    """

    ColorThenInteraction = auto()
    """This strategy first iterates all edges (sorted by their color value) and then the
    interactions (sorted as ``X``, ``Y``, ``Z``).
    """

    InteractionThenColor = auto()
    """This strategy is the inverse to ``ColorThenInteraction``. It first iterates the interactions
    (sorted as ``X``, ``Y``, ``Z``) and then all edges (sorted by their color value).
    """

    InteractionThenColorZigZag = auto()
    """This strategy is similar to the ``InteractionThenColor`` one. However, it alternates between
    iterating the edges by incrementing and decrementing color values as it jumps from one
    interaction to the next. For example, if only ``X`` and ``Y`` interactions are included and
    three color values are used (``{1, 2, 3}``), this will result in the following order:
    ``["X on 1", "X on 2", "X on 3", "Y on 3", "Y on 2", "Y on 1"]``.
    """


def generate_xyz_hamiltonian(
    coupling: CouplingMap | rx.PyGraph | rx.PyDiGraph,
    *,
    coupling_constants: Sequence[float] = (1.0, 1.0, 1.0),
    ext_magnetic_field: Sequence[float] = (0.0, 0.0, 0.0),
    pauli_order_strategy: PauliOrderStrategy = PauliOrderStrategy.ColorThenInteraction,
    coloring: dict[tuple[int, int], int] | None = None,
) -> SparsePauliOp:
    r"""Generate a connectivity-aware qubit operator representing a quantum XYZ-type model.

    This function implements the following Hamiltonian:

    .. math::
        \hat{H} = \sum_{(j,k)\in E} (J_{x} \sigma_j^{x} \sigma_{k}^{x} +
        J_{y} \sigma_j^{y} \sigma_{k}^{y} + J_{z} \sigma_j^{z} \sigma_{k}^{z}) +
        \sum_{j\in V} (h_{x} \sigma_j^{x} + h_{y} \sigma_j^{y} + h_{z} \sigma_j^{z})

    Where G(V,E) is the graph of the provided ``coupling`` map.

    .. note::

       There is often a :math:`-\frac{1}{2}` factor included outside the summation of this
       equation. This factor is not applied internally, so it should be accounted for
       in the ``coupling_constants`` and ``ext_magnetic_field`` inputs.

    .. code-block:: python

        >>> from qiskit.transpiler import CouplingMap
        >>> from qiskit_addon_utils.problem_generators import generate_xyz_hamiltonian

        >>> coupling_map = CouplingMap.from_line(10)
        >>> hamiltonian = generate_xyz_hamiltonian(
        ...     coupling_map,
        ...     coupling_constants=(0.4, 0.4, 0.0),
        ...     ext_magnetic_field=(0.0, 0.0, 0.6),
        ... )
        >>> print(hamiltonian)
        SparsePauliOp(['IIIIIIIXXI', 'IIIIIIIYYI', 'IIIIIXXIII', 'IIIIIYYIII',
                       'IIIXXIIIII', 'IIIYYIIIII', 'IXXIIIIIII', 'IYYIIIIIII',
                       'IIIIIIIIXX', 'IIIIIIIIYY', 'IIIIIIXXII', 'IIIIIIYYII',
                       'IIIIXXIIII', 'IIIIYYIIII', 'IIXXIIIIII', 'IIYYIIIIII',
                       'XXIIIIIIII', 'YYIIIIIIII', 'IIIIIIIIIZ', 'IIIIIIIIZI',
                       'IIIIIIIZII', 'IIIIIIZIII', 'IIIIIZIIII', 'IIIIZIIIII',
                       'IIIZIIIIII', 'IIZIIIIIII', 'IZIIIIIIII', 'ZIIIIIIIII'],
                    coeffs=[0.4+0.j, 0.4+0.j, 0.4+0.j, 0.4+0.j, 0.4+0.j, 0.4+0.j, 0.4+0.j, 0.4+0.j,
                            0.4+0.j, 0.4+0.j, 0.4+0.j, 0.4+0.j, 0.4+0.j, 0.4+0.j, 0.4+0.j, 0.4+0.j,
                            0.4+0.j, 0.4+0.j, 0.6+0.j, 0.6+0.j, 0.6+0.j, 0.6+0.j, 0.6+0.j, 0.6+0.j,
                            0.6+0.j, 0.6+0.j, 0.6+0.j, 0.6+0.j])

    Args:
        coupling: The qubit subgraph on which to map the Hamiltonian. Directionality of graph edges
            will be ignored, and parallel edges will be treated as a single edge during generation
            of the operator.
        coupling_constants: The real-valued coupling constants, :math:`J_i`, in each Cartesian axis.
        ext_magnetic_field: The coefficients, :math:`h_i`, representing a magnetic field along each
            Cartesian axis.
        pauli_order_strategy: Indicates the iteration strategy in which the Pauli terms will be
            generated. See :class:`.PauliOrderStrategy` for more details.
        coloring: An optional dictionary encoding the graph coloring that is used to sort the
            Hamiltonian terms. This dictionary maps edge labels (in the form of integer pairs) to
            color values (simple integers). Hamiltonian interaction terms will be added by
            increasing color value. Within each color, edges are sorted which does not change
            anything physically but results in easier to read results.

    Returns:
        A qubit operator describing a quantum XYZ-type model. The ``i``-th qubit in the operator
        corresponds to the node in index ``i`` on the coupling map.

    Raises:
        ValueError: The coupling constants must be specified by a length-3 sequence of floating
            point values.
        ValueError: The external magnetic field must be specified by a length-3 sequence of floating
            point values.
    """
    if len(coupling_constants) != 3:
        raise ValueError(
            "Coupling constants must be specified by a length-3 sequence of floating point values."
        )
    if len(ext_magnetic_field) != 3:
        raise ValueError(
            "External magnetic field must be specified by a length-3 sequence of floating point values."
        )

    if coloring is None:
        # Specify the coupling as an undirected rx.PyGraph so we can color the edges
        undirected_graph = _make_undirected_graph(coupling)
        coloring = auto_color_edges(undirected_graph.edge_list())

    # Sort edges by color to make for easier visualization
    colored_edges = sorted(coloring.items(), key=lambda pair: pair[1])

    # Generate Hamiltonian
    num_qubits = coupling.size() if isinstance(coupling, CouplingMap) else coupling.num_nodes()
    ham_sparse_list = []
    if pauli_order_strategy == PauliOrderStrategy.ColorThenInteraction:
        for edge, _ in colored_edges:
            for p, J in zip(("XX", "YY", "ZZ"), coupling_constants):
                if not np.isclose(J, 0.0):
                    ham_sparse_list.append((p, [edge[0], edge[1]], J))
        for qubit in range(num_qubits):
            for p, h in zip("XYZ", ext_magnetic_field):
                if not np.isclose(h, 0.0):
                    ham_sparse_list.append((p, [qubit], h))
    elif pauli_order_strategy == PauliOrderStrategy.InteractionThenColor:
        for p, J in zip(("XX", "YY", "ZZ"), coupling_constants):
            if not np.isclose(J, 0.0):
                for edge, _ in colored_edges:
                    ham_sparse_list.append((p, [edge[0], edge[1]], J))
        for p, h in zip("XYZ", ext_magnetic_field):
            if not np.isclose(h, 0.0):
                for qubit in range(num_qubits):
                    ham_sparse_list.append((p, [qubit], h))
    elif pauli_order_strategy == PauliOrderStrategy.InteractionThenColorZigZag:
        zig_zag_state = False
        for p, J in zip(("XX", "YY", "ZZ"), coupling_constants):
            if not np.isclose(J, 0.0):
                edges = reversed(colored_edges) if zig_zag_state else iter(colored_edges)
                zig_zag_state = not zig_zag_state
                for edge, _ in edges:
                    ham_sparse_list.append((p, [edge[0], edge[1]], J))
        for p, h in zip("XYZ", ext_magnetic_field):
            if not np.isclose(h, 0.0):
                for qubit in range(num_qubits):
                    ham_sparse_list.append((p, [qubit], h))
    else:  # pragma: no cover
        # NOTE: PauliOrderStrategy is an Enum so we cannot get here. Once Python 3.10 becomes the
        # minimum supported version, we can change this to a match statement which will remove the
        # need for this branch coverage exception.
        pass

    hamiltonian = SparsePauliOp.from_sparse_list(ham_sparse_list, num_qubits=num_qubits)

    return hamiltonian


def _make_undirected_graph(
    coupling: CouplingMap | rx.PyDiGraph | rx.PyGraph,
) -> rx.PyGraph:
    """Transform the coupling graph into an undirected graph with no parallel edges."""
    # Get the underlying graph
    input_graph = coupling.graph if isinstance(coupling, CouplingMap) else coupling

    # The output of PyDiGraph.to_undirected (below) has references to original node data,
    # and it is possible we may prune edges later, so it's best to just copy the input
    # structure to avoid modifying user data structures directly
    input_graph = copy.deepcopy(input_graph)

    # Get an undirected graph from the input graph
    if isinstance(input_graph, rx.PyDiGraph):
        undirected_graph = input_graph.to_undirected(multigraph=False)
    else:
        undirected_graph = input_graph

    # Prune parallel edges, as they interfere with coloring and circuit creation
    if undirected_graph.has_parallel_edges():
        undirected_edges = set()
        for edge in undirected_graph.edge_list():
            if edge[::-1] in undirected_edges:
                undirected_graph.remove_edge(edge[0], edge[1])
            else:
                undirected_edges.add(edge)

    return undirected_graph
