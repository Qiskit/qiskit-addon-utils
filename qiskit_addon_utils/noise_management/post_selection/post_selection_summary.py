# This code is a Qiskit project.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Reminder: update the RST file in docs/apidocs when adding new interfaces.
"""Post selection summary."""

from __future__ import annotations

from typing import Any

from qiskit.circuit import ClassicalRegister, QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import CouplingMap

from ..constants import DEFAULT_POST_SELECTION_SUFFIX


class PostSelectionSummary:
    """A helper class to store the properties of a quantum circuit required to post select the results."""

    def __init__(
        self,
        primary_cregs: set[str],
        measure_map: dict[int, tuple[str, int]],
        edges: set[frozenset[int]],
        *,
        post_selection_suffix: str = DEFAULT_POST_SELECTION_SUFFIX,
    ):
        """Initialize a ``PostSelectionSummary`` object.

        Args:
            primary_cregs: The names of the "primary" classical registers, namely those that do not end with
                the post selection suffix.
            measure_map: A map between qubit indices to the register and clbits that uniquely define a
                measurement on those qubits.
            edges: A list of tuples defining pairs of neighboring qubits.
            post_selection_suffix: The suffix of the post selection registers.
        """
        self._primary_cregs = primary_cregs
        self._measure_map = measure_map
        self._edges = edges
        self._post_selection_suffix = post_selection_suffix

    @property
    def measure_map(self) -> dict[int, tuple[str, int]]:
        """A map from qubit indices to the register and clbit index used to measure those qubits."""
        return self._measure_map

    @property
    def edges(self) -> set[frozenset[int]]:
        """A set of edges to consider for edge-based post selection."""
        return self._edges

    @property
    def primary_cregs(self) -> set[str]:
        """The names of the "primary" classical registers."""
        return self._primary_cregs

    @property
    def post_selection_suffix(self) -> str:
        """The suffix of the post selection registers."""
        return self._post_selection_suffix

    @classmethod
    def from_circuit(
        cls,
        circuit: QuantumCircuit,
        coupling_map: CouplingMap | list[tuple[int, int]],
        *,
        post_selection_suffix: str = DEFAULT_POST_SELECTION_SUFFIX,
    ) -> PostSelectionSummary:
        """Initialize from quantum circuits.

        Args:
            circuit: The circuit to create a summary of.
            coupling_map: A coupling map or a list of tuples indicating pairs of neighboring qubits.
            post_selection_suffix: A fixed suffix to append to the names of the classical registers when
                copying them.
        """
        coupling_map = (
            coupling_map
            if isinstance(coupling_map, CouplingMap)
            else CouplingMap(couplinglist=coupling_map)
        )

        cregs = (dag := circuit_to_dag(circuit)).cregs
        primary_cregs, ps_cregs = _get_primary_and_ps_cregs(cregs, post_selection_suffix)
        _validate_cregs(primary_cregs, ps_cregs, post_selection_suffix)

        measure_map, measure_map_ps = _get_measure_maps(dag, primary_cregs, ps_cregs)
        _validate_measure_maps(measure_map, measure_map_ps, post_selection_suffix)

        return PostSelectionSummary(
            set(primary_cregs),
            measure_map,
            _get_edges(coupling_map, measure_map),
            post_selection_suffix=post_selection_suffix,
        )

    def __eq__(self, other: Any) -> bool:  # noqa: D105
        return (
            isinstance(other, PostSelectionSummary)
            and self.primary_cregs == other.primary_cregs
            and self.edges == other.edges
            and self.measure_map == other.measure_map
            and self.post_selection_suffix == other.post_selection_suffix
        )


def _get_primary_and_ps_cregs(
    cregs: dict[str, ClassicalRegister],
    post_selection_suffix: str,
) -> tuple[dict[str, ClassicalRegister], dict[str, ClassicalRegister]]:
    """Split a dictionary of registers into primary and post selection registers.

    Args:
        cregs: The dictionary of registers.
        post_selection_suffix: The suffix of the post selection registers.
    """
    primary_cregs = {
        name: creg for name, creg in cregs.items() if not name.endswith(post_selection_suffix)
    }

    ps_cregs = {name: creg for name, creg in cregs.items() if name.endswith(post_selection_suffix)}

    return primary_cregs, ps_cregs


def _validate_cregs(
    primary_cregs: dict[str, ClassicalRegister],
    ps_cregs: dict[str, ClassicalRegister],
    post_selection_suffix: str,
) -> None:
    """Validate primary and post selection registers.

    This function checks that every primary register has a corresponding post selection register with
    matching names (expect for the suffix at the end of the posts election register's name) and the same
    number of clbits.s

    Args:
        primary_cregs: The primary cregs.
        ps_cregs: The post selection cregs.
        post_selection_suffix: The suffix of the post selection registers.

    Raise:
        ValueError: If the names do not match.
        ValueError: If the sizes do not match.
    """
    expected_ps_names = {name + post_selection_suffix for name in primary_cregs}
    if expected_ps_names != set(ps_cregs):
        sorted_primary_names = ", ".join(sorted(list(primary_cregs)))
        sorted_ps_names = ", ".join(sorted(list(ps_cregs)))
        raise ValueError(
            f"Cannot apply post selection for circuit with primary registers {sorted_primary_names} "
            f"and post selection registers {sorted_ps_names}. Every primary register must correspond "
            f"to a post selection register with the same name and suffix {post_selection_suffix}."
        )

    for name, primary_creg in primary_cregs.items():
        if len(primary_creg) != len(ps_creg := ps_cregs[name + post_selection_suffix]):
            raise ValueError(
                f"Primary register {name} has {len(primary_creg)} clbits, but post selection register "
                f"{name + post_selection_suffix} has {len(ps_creg)} clbits."
            )


def _get_measure_maps(
    dag: DAGCircuit,
    primary_cregs: dict[str, ClassicalRegister],
    ps_cregs: dict[str, ClassicalRegister],
) -> tuple[dict[int, tuple[str, int]], dict[int, tuple[str, int]]]:
    """Map the qubits in ``dag`` to the registers and clbits used to measure them.

    Args:
        dag: The dag circuit.
        primary_cregs: The primary cregs.
        ps_cregs: The post selection cregs.
    """
    # A map between clbits in the primary registers to the register that owns them and the
    # positions that they occupy in those registers
    clbit_map = {
        clbit: (name, clbit_idx)
        for name, creg in primary_cregs.items()
        for clbit_idx, clbit in enumerate(creg)
    }

    # A map between clbits in the post selection registers to the register that owns them
    # and the positions that they occupy in those registers
    clbit_map_ps = {
        clbit: (name, clbit_idx)
        for name, creg in ps_cregs.items()
        for clbit_idx, clbit in enumerate(creg)
    }

    qubit_map = {qubit: idx for idx, qubit in enumerate(dag.qubits)}

    measure_map: dict[int, tuple[str, int]] = {}
    measure_map_ps: dict[int, tuple[str, int]] = {}

    for node in dag.topological_op_nodes():
        if node.op.name == "measure":  # pragma: no cover
            if clbit := clbit_map.get(node.cargs[0], None):
                measure_map[qubit_map[node.qargs[0]]] = clbit
            elif clbit_ps := clbit_map_ps.get(node.cargs[0], None):
                measure_map_ps[qubit_map[node.qargs[0]]] = clbit_ps
            else:  # pragma: no cover
                raise ValueError(f"Clbit {node.cargs[0]} does not belong to any valid register.")

    return measure_map, measure_map_ps


def _validate_measure_maps(
    measure_map: dict[int, tuple[str, int]],
    measure_map_ps: dict[int, tuple[str, int]],
    post_selection_suffix: str,
) -> None:
    """Validate measurement maps.

    This function checks that the measurement maps of the primary registers and those of the post selection
    registers are compatible, i.e., that they are of the same length, that they contain the same qubits, and that
    qubits are mapped to matching bits.

    Args:
        measure_map: The measurement map for the primary measurements.
        measure_map_ps: The measurement map for the post selection measurements.
        post_selection_suffix: The suffix of the post selection registers.

    Raise:
        ValueError: If the two measurement maps are of different length.
        ValueError: If a qubit is present in a map but not in the other one.
        ValueError: If the same qubit is mapped to different bits.
    """
    if len(measure_map) != len(measure_map_ps):
        raise ValueError(
            f"Found {len(measure_map)} measurements and {len(measure_map_ps)} "
            "post selection measurements."
        )

    for qubit_idx, (name, clbit_idx) in measure_map.items():
        try:
            name_ps, clbit_idx_ps = measure_map_ps[qubit_idx]
        except KeyError as key_err:
            raise ValueError(
                f"Missing post selection measurement on qubit {qubit_idx}."
            ) from key_err

        if name_ps != (expected_name := name + post_selection_suffix) or clbit_idx != clbit_idx_ps:
            raise ValueError(
                f"Expected measurement on qubit {qubit_idx} writing to bit {clbit_idx} of creg "
                f"{expected_name}, found measurement writing to bit {clbit_idx_ps} of creg "
                f"{name_ps}."
            )


def _get_edges(
    coupling_map: CouplingMap,
    measure_map: dict[int, tuple[str, int]],
) -> set[frozenset[int]]:
    """Get the set of edges that are relevant for edge-based post selection."""
    return {
        frozenset(edge)
        for edge in coupling_map.get_edges()
        if edge[0] in measure_map and edge[1] in measure_map
    }
