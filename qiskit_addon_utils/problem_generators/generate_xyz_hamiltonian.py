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
    coupling_constants: float | Sequence[float] | dict[tuple[int, int], float | Sequence[float]] = (
        1.0,
        1.0,
        1.0,
    ),
    ext_magnetic_field: float | Sequence[float] | dict[int, float | Sequence[float]] = (
        0.0,
        0.0,
        0.0,
    ),
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
            May be a single scalar, a length-3 sequence, or a dict mapping edge tuples
            `(i, j)` to scalars or length-3 sequences.
        ext_magnetic_field: The coefficients, :math:`h_i`, representing a magnetic field
            along each Cartesian axis. May be a single scalar, a length-3 sequence,
            or a dict mapping qubit indices to scalars or length-3 sequences.
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
        ValueError: Coupling constants must be a scalar or length-3 sequence of floating point values.
        ValueError: External magnetic field must be a scalar or length-3 sequence of floating point values.
        ValueError: Edge keys must be tuples of two integer qubit indices.
        ValueError: Coupling constants contains conflicting values for an edge.
        ValueError: Magnetic field keys must be integer qubit indices.
    """
    # Validate inputs
    _validate_xyz_input(coupling_constants, name="Coupling constants")
    _validate_xyz_input(ext_magnetic_field, name="External magnetic field")
    _validate_edge_dict_keys(coupling_constants)

    if coloring is None:
        # Specify the coupling as an undirected rx.PyGraph so we can color the edges
        undirected_graph = _make_undirected_graph(coupling)
        coloring = auto_color_edges(undirected_graph.edge_list())

    # Sort edges by color to make for easier visualization
    colored_edges = sorted(coloring.items(), key=lambda pair: pair[1])

    # Generate Hamiltonian
    num_qubits = coupling.size() if isinstance(coupling, CouplingMap) else coupling.num_nodes()

    # Normalize the coupling constants and magnetic field values to a per-edge and per-qubit mapping
    edge_couplings = _normalize_coupling_constants(coupling_constants, colored_edges)
    site_fields = _normalize_ext_magnetic_field(ext_magnetic_field, num_qubits)

    ham_sparse_list = []
    if pauli_order_strategy == PauliOrderStrategy.ColorThenInteraction:
        for edge, _ in colored_edges:
            Jxx, Jyy, Jzz = edge_couplings[edge]
            for p, J in zip(("XX", "YY", "ZZ"), (Jxx, Jyy, Jzz)):
                if not np.isclose(J, 0.0):
                    ham_sparse_list.append((p, [edge[0], edge[1]], J))
        for qubit in range(num_qubits):
            hx, hy, hz = site_fields[qubit]
            for p, h in zip("XYZ", (hx, hy, hz)):
                if not np.isclose(h, 0.0):
                    ham_sparse_list.append((p, [qubit], h))
    elif pauli_order_strategy == PauliOrderStrategy.InteractionThenColor:
        for p_index, p in enumerate(("XX", "YY", "ZZ")):
            for edge, _ in colored_edges:
                Jxx, Jyy, Jzz = edge_couplings[edge]
                J = (Jxx, Jyy, Jzz)[p_index]
                if not np.isclose(J, 0.0):
                    ham_sparse_list.append((p, [edge[0], edge[1]], J))
        for p_index, p in enumerate("XYZ"):
            for qubit in range(num_qubits):
                hx, hy, hz = site_fields[qubit]
                h = (hx, hy, hz)[p_index]
                if not np.isclose(h, 0.0):
                    ham_sparse_list.append((p, [qubit], h))
    elif pauli_order_strategy == PauliOrderStrategy.InteractionThenColorZigZag:
        zig_zag_state = False
        for p_index, p in enumerate(("XX", "YY", "ZZ")):
            edges = reversed(colored_edges) if zig_zag_state else colored_edges
            zig_zag_state = not zig_zag_state
            for edge, _ in edges:
                Jxx, Jyy, Jzz = edge_couplings[edge]
                J = (Jxx, Jyy, Jzz)[p_index]
                if not np.isclose(J, 0.0):
                    ham_sparse_list.append((p, [edge[0], edge[1]], J))
        for p_index, p in enumerate("XYZ"):
            for qubit in range(num_qubits):
                hx, hy, hz = site_fields[qubit]
                h = (hx, hy, hz)[p_index]
                if not np.isclose(h, 0.0):
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


def _validate_xyz_input(value: float | Sequence[float] | dict, *, name: str) -> None:
    """Validate top-level scalar/sequence inputs before normalization."""
    if isinstance(value, dict):
        return

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) != 3:
        raise ValueError(
            f"{name} must be specified by a length-3 sequence of scalar or floating point values."
        )


def _validate_edge_dict_keys(
    coupling_constants: float | Sequence[float] | dict[tuple[int, int], float | Sequence[float]],
) -> None:
    """Validate that dict-based edge couplings use valid edge keys."""
    if not isinstance(coupling_constants, dict):
        return

    for edge in coupling_constants:
        _normalize_edge_key(edge)


def _normalize_xyz_triplet(
    value: float | Sequence[float],
) -> tuple[float, float, float]:
    """Normalize a scalar or 3-element sequence to a length-3 tuple."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 3:
            raise ValueError(
                "Value must be a scalar or length-3 sequence of floating point values."
            )
        # If the value is a scalar, return a tuple with the same value repeated three times.
        return (float(value[0]), float(value[1]), float(value[2]))
    return (float(value), float(value), float(value))


def _normalize_edge_key(edge: tuple[int, int]) -> tuple[int, int]:
    """Canonicalize an undirected edge key so (i, j) and (j, i) match."""
    # Validate the edge key is a tuple of two integers
    if (
        not isinstance(edge, tuple)
        or len(edge) != 2
        or not all(isinstance(idx, int) for idx in edge)
    ):
        raise ValueError("Edge keys must be tuples of two integer qubit indices.")
    return tuple(sorted(edge))


def _normalize_coupling_constants(
    coupling_constants: float | Sequence[float] | dict[tuple[int, int], float | Sequence[float]],
    colored_edges: list[tuple[tuple[int, int], int]],
) -> dict[tuple[int, int], tuple[float, float, float]]:
    """Return a per-edge (Jx, Jy, Jz) mapping for every edge."""
    if isinstance(coupling_constants, dict):
        normalized: dict[tuple[int, int], tuple[float, float, float]] = {}
        for edge, value in coupling_constants.items():
            canonical_edge = _normalize_edge_key(edge)
            triplet = _normalize_xyz_triplet(value)

            if canonical_edge in normalized:
                if normalized[canonical_edge] != triplet:
                    raise ValueError(
                        f"Coupling_constants contains conflicting values for edge {canonical_edge}."
                    )
                continue

            normalized[canonical_edge] = triplet

        edge_map = {}
        for edge, _ in colored_edges:
            edge_map[edge] = normalized.get(_normalize_edge_key(edge), (0.0, 0.0, 0.0))
        return edge_map

    triple = _normalize_xyz_triplet(coupling_constants)
    return {edge: triple for edge, _ in colored_edges}


def _normalize_ext_magnetic_field(
    ext_magnetic_field: float | Sequence[float] | dict[int, float | Sequence[float]],
    num_qubits: int,
) -> dict[int, tuple[float, float, float]]:
    """Return a per-qubit (hx, hy, hz) mapping for every qubit."""
    if isinstance(ext_magnetic_field, dict):
        normalized: dict[int, tuple[float, float, float]] = {}
        for qubit, value in ext_magnetic_field.items():
            if not isinstance(qubit, int):
                raise ValueError("Magnetic field keys must be integer qubit indices.")
            normalized[qubit] = _normalize_xyz_triplet(value)
        return {qubit: normalized.get(qubit, (0.0, 0.0, 0.0)) for qubit in range(num_qubits)}

    triple = _normalize_xyz_triplet(ext_magnetic_field)
    return {qubit: triple for qubit in range(num_qubits)}
