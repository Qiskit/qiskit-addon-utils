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
"""Pre selection summary."""

from __future__ import annotations

from typing import Any

from qiskit.circuit import ClassicalRegister, QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import CouplingMap

from ..constants import DEFAULT_PRE_SELECTION_SUFFIX


class PreSelectionSummary:
    """A helper class to store the properties of a quantum circuit required to pre select the results."""

    def __init__(
        self,
        primary_cregs: set[str],
        measure_map: dict[int, tuple[str, int]],
        edges: set[frozenset[int]],
        *,
        measure_map_pre: dict[int, tuple[str, int]] | None = None,
        pre_selection_suffix: str = DEFAULT_PRE_SELECTION_SUFFIX,
    ):
        """Initialize a ``PreSelectionSummary`` object.

        Args:
            primary_cregs: The names of the "primary" classical registers, namely those that do not end with
                the pre selection suffix.
            measure_map: A map between qubit indices to the register and clbits that uniquely define a
                measurement on those qubits.
            edges: A list of tuples defining pairs of neighboring qubits.
            measure_map_pre: A map between qubit indices to the register and clbits for pre-selection measurements.
            pre_selection_suffix: The suffix of the pre selection registers.
        """
        self._primary_cregs = primary_cregs
        self._measure_map = measure_map
        self._measure_map_pre = measure_map_pre if measure_map_pre is not None else {}
        self._edges = edges
        self._pre_selection_suffix = pre_selection_suffix

    @property
    def measure_map(self) -> dict[int, tuple[str, int]]:
        """A map from qubit indices to the register and clbit index used to measure those qubits."""
        return self._measure_map

    @property
    def measure_map_pre(self) -> dict[int, tuple[str, int]]:
        """A map from qubit indices to the register and clbit index for pre-selection measurements."""
        return self._measure_map_pre

    @property
    def edges(self) -> set[frozenset[int]]:
        """A set of edges to consider for edge-based pre selection."""
        return self._edges

    @property
    def primary_cregs(self) -> set[str]:
        """The names of the "primary" classical registers."""
        return self._primary_cregs

    @property
    def pre_selection_suffix(self) -> str:
        """The suffix of the pre selection registers."""
        return self._pre_selection_suffix

    @classmethod
    def from_circuit(
        cls,
        circuit: QuantumCircuit,
        coupling_map: CouplingMap | list[tuple[int, int]],
        *,
        pre_selection_suffix: str = DEFAULT_PRE_SELECTION_SUFFIX,
    ) -> PreSelectionSummary:
        """Initialize from quantum circuits.

        Args:
            circuit: The circuit to create a summary of.
            coupling_map: A coupling map or a list of tuples indicating pairs of neighboring qubits.
            pre_selection_suffix: A fixed suffix to append to the names of the classical registers when
                copying them.
        """
        coupling_map = (
            coupling_map
            if isinstance(coupling_map, CouplingMap)
            else CouplingMap(couplinglist=coupling_map)
        )

        cregs = (dag := circuit_to_dag(circuit)).cregs
        primary_cregs, pre_cregs = _get_primary_and_pre_cregs(cregs, pre_selection_suffix)
        _validate_cregs(primary_cregs, pre_cregs, pre_selection_suffix)

        measure_map, measure_map_pre = _get_measure_maps(dag, primary_cregs, pre_cregs)
        _validate_measure_maps(measure_map, measure_map_pre, pre_selection_suffix)

        return PreSelectionSummary(
            set(primary_cregs),
            measure_map,
            _get_edges(coupling_map, measure_map),
            measure_map_pre=measure_map_pre,
            pre_selection_suffix=pre_selection_suffix,
        )

    def __eq__(self, other: Any) -> bool:  # noqa: D105
        return (
            isinstance(other, PreSelectionSummary)
            and self.primary_cregs == other.primary_cregs
            and self.edges == other.edges
            and self.measure_map == other.measure_map
            and self.measure_map_pre == other.measure_map_pre
            and self.pre_selection_suffix == other.pre_selection_suffix
        )


def _get_primary_and_pre_cregs(
    cregs: dict[str, ClassicalRegister],
    pre_selection_suffix: str,
) -> tuple[dict[str, ClassicalRegister], dict[str, ClassicalRegister]]:
    """Split a dictionary of registers into primary and pre selection registers.

    Args:
        cregs: The dictionary of registers.
        pre_selection_suffix: The suffix of the pre selection registers.
    """
    # Exclude post-selection registers (ending with _ps) from primary registers
    # that need pre-selection validation, since post-selection happens after the circuit
    primary_cregs = {
        name: creg
        for name, creg in cregs.items()
        if not name.endswith(pre_selection_suffix) and not name.endswith("_ps")
    }

    pre_cregs = {name: creg for name, creg in cregs.items() if name.endswith(pre_selection_suffix)}

    return primary_cregs, pre_cregs


def _validate_cregs(
    primary_cregs: dict[str, ClassicalRegister],
    pre_cregs: dict[str, ClassicalRegister],
    pre_selection_suffix: str,
) -> None:
    """Validate primary and pre selection registers.

    This function checks that every primary register has a corresponding pre selection register with
    matching names (expect for the suffix at the end of the pre selection register's name) and the same
    number of clbits.

    Args:
        primary_cregs: The primary cregs.
        pre_cregs: The pre selection cregs.
        pre_selection_suffix: The suffix of the pre selection registers.

    Raise:
        ValueError: If the names do not match.
        ValueError: If the sizes do not match.
    """
    expected_pre_names = {name + pre_selection_suffix for name in primary_cregs}
    if expected_pre_names != set(pre_cregs):
        sorted_primary_names = ", ".join(sorted(list(primary_cregs)))
        sorted_pre_names = ", ".join(sorted(list(pre_cregs)))
        raise ValueError(
            f"Cannot apply pre selection for circuit with primary registers {sorted_primary_names} "
            f"and pre selection registers {sorted_pre_names}. Every primary register must correspond "
            f"to a pre selection register with the same name and suffix {pre_selection_suffix}."
        )

    for name, primary_creg in primary_cregs.items():
        if len(primary_creg) != len(pre_creg := pre_cregs[name + pre_selection_suffix]):
            raise ValueError(
                f"Primary register {name} has {len(primary_creg)} clbits, but pre selection register "
                f"{name + pre_selection_suffix} has {len(pre_creg)} clbits."
            )


def _get_measure_maps(
    dag: DAGCircuit,
    primary_cregs: dict[str, ClassicalRegister],
    pre_cregs: dict[str, ClassicalRegister],
) -> tuple[dict[int, tuple[str, int]], dict[int, tuple[str, int]]]:
    """Map the qubits in ``dag`` to the registers and clbits used to measure them.

    Args:
        dag: The dag circuit.
        primary_cregs: The primary cregs.
        pre_cregs: The pre selection cregs.
    """
    # A map between clbits in the primary registers to the register that owns them and the
    # positions that they occupy in those registers
    clbit_map = {
        clbit: (name, clbit_idx)
        for name, creg in primary_cregs.items()
        for clbit_idx, clbit in enumerate(creg)
    }

    # A map between clbits in the pre selection registers to the register that owns them
    # and the positions that they occupy in those registers
    clbit_map_pre = {
        clbit: (name, clbit_idx)
        for name, creg in pre_cregs.items()
        for clbit_idx, clbit in enumerate(creg)
    }

    qubit_map = {qubit: idx for idx, qubit in enumerate(dag.qubits)}

    measure_map: dict[int, tuple[str, int]] = {}
    measure_map_pre: dict[int, tuple[str, int]] = {}

    for node in dag.topological_op_nodes():
        if node.op.name == "measure":  # pragma: no cover
            if clbit := clbit_map.get(node.cargs[0]):
                measure_map[qubit_map[node.qargs[0]]] = clbit
            elif clbit_pre := clbit_map_pre.get(node.cargs[0]):
                measure_map_pre[qubit_map[node.qargs[0]]] = clbit_pre
            # Skip measurements into post-selection registers (ending with _ps)
            # as they are not relevant for pre-selection analysis

    return measure_map, measure_map_pre


def _validate_measure_maps(
    measure_map: dict[int, tuple[str, int]],
    measure_map_pre: dict[int, tuple[str, int]],
    pre_selection_suffix: str,
) -> None:
    """Validate measurement maps.

    This function checks that the measurement maps of the primary registers and those of the pre selection
    registers are compatible, i.e., that they contain the same qubits, and that
    qubits are mapped to matching bits.

    Args:
        measure_map: The measurement map for the primary measurements.
        measure_map_pre: The measurement map for the pre selection measurements.
        pre_selection_suffix: The suffix of the pre selection registers.

    Raise:
        ValueError: If a qubit is present in a map but not in the other one.
        ValueError: If the same qubit is mapped to different bits.
    """
    for qubit_idx, (name, clbit_idx) in measure_map_pre.items():
        # For pre-selection, we only check qubits that have pre-selection measurements
        # It's okay if some qubits don't have primary measurements yet
        if qubit_idx in measure_map:
            name_primary, clbit_idx_primary = measure_map[qubit_idx]
            expected_pre_name = name_primary + pre_selection_suffix
            if name != expected_pre_name or clbit_idx != clbit_idx_primary:
                raise ValueError(
                    f"Pre-selection measurement on qubit {qubit_idx} writes to bit {clbit_idx} of creg "
                    f"{name}, but expected to write to bit {clbit_idx_primary} of creg "
                    f"{expected_pre_name}."
                )


def _get_edges(
    coupling_map: CouplingMap,
    measure_map: dict[int, tuple[str, int]],
) -> set[frozenset[int]]:
    """Get the set of edges that are relevant for edge-based pre selection."""
    return {
        frozenset(edge)
        for edge in coupling_map.get_edges()
        if edge[0] in measure_map and edge[1] in measure_map
    }


# Made with Bob
