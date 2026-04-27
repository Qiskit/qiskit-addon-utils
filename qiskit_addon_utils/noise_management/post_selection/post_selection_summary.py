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

from typing import Any, Literal

from qiskit.circuit import ClassicalRegister, QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import CouplingMap

from ..constants import (
    DEFAULT_POST_SELECTION_SUFFIX,
    DEFAULT_PRE_SELECTION_SUFFIX,
    DEFAULT_SPECTATOR_CREG_NAME,
)


class PostSelectionSummary:
    """A helper class to store the properties of a quantum circuit required to post select the results.

    This class handles both traditional post-selection (measurements at the end of the circuit) and
    pre-selection (measurements at the beginning of the circuit to verify initialization).

    A primary register may either hold "data" measurements that the user cares about
    (a problem-specific register such as ``c``) or hold the first half of a spectator
    parity check (a register such as ``spec`` produced by
    :class:`.AddSpectatorMeasures`). The :attr:`spectator_cregs` property labels which
    primary registers fall into the spectator category. The mask computation treats
    both the same — every primary register participates in the parity check — but
    downstream code that wants to extract observables from data measurements only
    can use ``primary_cregs - spectator_cregs`` to find the data registers.

    .. note::
        With :class:`.AddSpectatorMeasures` configured with ``include_unmeasured=True``,
        active-but-unterminated qubits are also written into the spectator register.
        :attr:`spectator_cregs` therefore reflects the *register identity*, not a
        guarantee that every qubit writing into it was an idle neighbour.
    """

    def __init__(
        self,
        primary_cregs: set[str],
        measure_map: dict[int, tuple[str, int]],
        edges: set[frozenset[int]],
        *,
        measure_map_ps: dict[int, tuple[str, int]] | None = None,
        measure_map_pre: dict[int, tuple[str, int]] | None = None,
        post_selection_suffix: str = DEFAULT_POST_SELECTION_SUFFIX,
        pre_selection_suffix: str = DEFAULT_PRE_SELECTION_SUFFIX,
        spectator_cregs: set[str] | None = None,
    ):
        """Initialize a ``PostSelectionSummary`` object.

        Args:
            primary_cregs: The names of the "primary" classical registers, namely those that do not end with
                the post selection or pre selection suffix.
            measure_map: A map between qubit indices to the register and clbits that uniquely define a
                measurement on those qubits (primary measurements).
            edges: A list of tuples defining pairs of neighboring qubits.
            measure_map_ps: An optional map for post-selection measurements (at end of circuit).
                If None, defaults to empty dict.
            measure_map_pre: An optional map for pre-selection measurements (at start of circuit).
                If None, defaults to empty dict.
            post_selection_suffix: The suffix of the post selection registers.
            pre_selection_suffix: The suffix of the pre selection registers.
            spectator_cregs: Names of primary registers that hold spectator measurements
                (the first half of the spectator parity check produced by
                :class:`.AddSpectatorMeasures`). Stored intersected with ``primary_cregs``;
                names not present in the circuit are silently dropped. Defaults to the
                empty set.
        """
        self._primary_cregs = primary_cregs
        self._measure_map = measure_map
        self._measure_map_ps = measure_map_ps if measure_map_ps is not None else {}
        self._measure_map_pre = measure_map_pre if measure_map_pre is not None else {}
        self._edges = edges
        self._post_selection_suffix = post_selection_suffix
        self._pre_selection_suffix = pre_selection_suffix
        self._spectator_cregs = set(spectator_cregs) & primary_cregs if spectator_cregs else set()

    @property
    def measure_map(self) -> dict[int, tuple[str, int]]:
        """A map from qubit indices to the register and clbit index used to measure those qubits."""
        return self._measure_map

    @property
    def measure_map_ps(self) -> dict[int, tuple[str, int]]:
        """A map from qubit indices to the register and clbit index for post-selection measurements."""
        return self._measure_map_ps

    @property
    def measure_map_pre(self) -> dict[int, tuple[str, int]]:
        """A map from qubit indices to the register and clbit index for pre-selection measurements."""
        return self._measure_map_pre

    @property
    def edges(self) -> set[frozenset[int]]:
        """A set of edges to consider for edge-based post selection."""
        return self._edges

    @property
    def primary_cregs(self) -> set[str]:
        """The names of the "primary" classical registers."""
        return self._primary_cregs

    @property
    def spectator_cregs(self) -> set[str]:
        """Subset of ``primary_cregs`` that hold spectator measurements.

        Use ``primary_cregs - spectator_cregs`` to get the data-only primary
        registers, e.g. when computing observables from filtered shots.
        """
        return self._spectator_cregs

    @property
    def post_selection_suffix(self) -> str:
        """The suffix of the post selection registers."""
        return self._post_selection_suffix

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
        post_selection_suffix: str = DEFAULT_POST_SELECTION_SUFFIX,
        pre_selection_suffix: str = DEFAULT_PRE_SELECTION_SUFFIX,
        spectator_cregs: set[str] | list[str] | None = None,
    ) -> PostSelectionSummary:
        """Initialize from quantum circuits.

        Args:
            circuit: The circuit to create a summary of.
            coupling_map: A coupling map or a list of tuples indicating pairs of neighboring qubits.
            post_selection_suffix: A fixed suffix for post-selection classical registers.
            pre_selection_suffix: A fixed suffix for pre-selection classical registers.
            spectator_cregs: Names of primary registers that hold spectator
                measurements. Defaults to ``[DEFAULT_SPECTATOR_CREG_NAME]``
                (i.e. ``["spec"]``), matching the default name used by
                :class:`.AddSpectatorMeasures`. Names absent from the circuit
                are silently dropped.
        """
        if spectator_cregs is None:
            spectator_cregs = [DEFAULT_SPECTATOR_CREG_NAME]
        coupling_map = (
            coupling_map
            if isinstance(coupling_map, CouplingMap)
            else CouplingMap(couplinglist=coupling_map)
        )

        cregs = (dag := circuit_to_dag(circuit)).cregs
        primary_cregs, ps_cregs, pre_cregs = _get_primary_ps_and_pre_cregs(
            cregs, post_selection_suffix, pre_selection_suffix
        )

        # Validate that primary registers have corresponding selection registers
        if ps_cregs:
            _validate_cregs(primary_cregs, ps_cregs, post_selection_suffix)
        if pre_cregs:
            _validate_cregs(primary_cregs, pre_cregs, pre_selection_suffix)

        measure_map, measure_map_ps, measure_map_pre = _get_measure_maps(
            dag, primary_cregs, ps_cregs, pre_cregs
        )

        # Post-selection requires a strict 1:1 mapping between primary and ``_ps``
        # measurements: ``_compute_post_mask_by_*`` indexes ``measure_map_ps`` by
        # every primary qubit, so a missing entry would raise at mask time.
        # Pre-selection is intrinsically partial (only terminally-measured qubits
        # are pre-selected), so it's validated leniently.
        if measure_map_ps:
            _validate_measure_maps(measure_map, measure_map_ps, post_selection_suffix, "strict")
        if measure_map_pre:
            _validate_measure_maps(measure_map, measure_map_pre, pre_selection_suffix, "lenient")

        return PostSelectionSummary(
            set(primary_cregs),
            measure_map,
            _get_edges(coupling_map, measure_map, measure_map_ps, measure_map_pre),
            measure_map_ps=measure_map_ps,
            measure_map_pre=measure_map_pre,
            post_selection_suffix=post_selection_suffix,
            pre_selection_suffix=pre_selection_suffix,
            spectator_cregs=set(spectator_cregs),
        )

    def __eq__(self, other: Any) -> bool:  # noqa: D105
        return (
            isinstance(other, PostSelectionSummary)
            and self.primary_cregs == other.primary_cregs
            and self.edges == other.edges
            and self.measure_map == other.measure_map
            and self.measure_map_ps == other.measure_map_ps
            and self.measure_map_pre == other.measure_map_pre
            and self.post_selection_suffix == other.post_selection_suffix
            and self.pre_selection_suffix == other.pre_selection_suffix
            and self.spectator_cregs == other.spectator_cregs
        )


def _get_primary_ps_and_pre_cregs(
    cregs: dict[str, ClassicalRegister],
    post_selection_suffix: str,
    pre_selection_suffix: str,
) -> tuple[
    dict[str, ClassicalRegister], dict[str, ClassicalRegister], dict[str, ClassicalRegister]
]:
    """Split a dictionary of registers into primary, post-selection, and pre-selection registers.

    Args:
        cregs: The dictionary of registers.
        post_selection_suffix: The suffix of the post selection registers.
        pre_selection_suffix: The suffix of the pre selection registers.
    """
    # Primary registers exclude both post-selection and pre-selection registers
    primary_cregs = {
        name: creg
        for name, creg in cregs.items()
        if not name.endswith(post_selection_suffix) and not name.endswith(pre_selection_suffix)
    }

    ps_cregs = {name: creg for name, creg in cregs.items() if name.endswith(post_selection_suffix)}
    pre_cregs = {name: creg for name, creg in cregs.items() if name.endswith(pre_selection_suffix)}

    return primary_cregs, ps_cregs, pre_cregs


def _validate_cregs(
    primary_cregs: dict[str, ClassicalRegister],
    selection_cregs: dict[str, ClassicalRegister],
    selection_suffix: str,
) -> None:
    """Validate primary and selection registers.

    Every primary register must have a corresponding selection register (same
    name + ``selection_suffix``, same number of clbits). Selection registers
    that don't correspond to any primary are *allowed* — they are treated as
    spectator selection registers (e.g. ``spec_pre`` produced by
    :class:`.AddSpectatorMeasuresPreSelection` when no spec primary exists).

    Args:
        primary_cregs: The primary cregs.
        selection_cregs: The selection cregs (either post or pre).
        selection_suffix: The suffix of the selection registers.

    Raise:
        ValueError: If a primary register is missing its matching selection register.
        ValueError: If a primary/selection size pair does not match.
    """
    expected_selection_names = {name + selection_suffix for name in primary_cregs}
    missing = expected_selection_names - set(selection_cregs)
    if missing:
        sorted_primary_names = ", ".join(sorted(list(primary_cregs)))
        sorted_selection_names = ", ".join(sorted(list(selection_cregs)))
        sorted_missing = ", ".join(sorted(missing))
        selection_type = "post selection" if selection_suffix == "_ps" else "pre selection"
        raise ValueError(
            f"Cannot apply {selection_type} for circuit with primary registers {sorted_primary_names} "
            f"and {selection_type} registers {sorted_selection_names}: missing matching {selection_type} "
            f"register(s) {sorted_missing} for the primary register(s)."
        )

    for name, primary_creg in primary_cregs.items():
        if len(primary_creg) != len(selection_creg := selection_cregs[name + selection_suffix]):
            selection_type = "post selection" if selection_suffix == "_ps" else "pre selection"
            raise ValueError(
                f"Primary register {name} has {len(primary_creg)} clbits, but {selection_type} register "
                f"{name + selection_suffix} has {len(selection_creg)} clbits."
            )


def _get_measure_maps(
    dag: DAGCircuit,
    primary_cregs: dict[str, ClassicalRegister],
    ps_cregs: dict[str, ClassicalRegister],
    pre_cregs: dict[str, ClassicalRegister],
) -> tuple[dict[int, tuple[str, int]], dict[int, tuple[str, int]], dict[int, tuple[str, int]]]:
    """Map the qubits in ``dag`` to the registers and clbits used to measure them.

    Args:
        dag: The dag circuit.
        primary_cregs: The primary cregs.
        ps_cregs: The post selection cregs.
        pre_cregs: The pre selection cregs.
    """
    # A map between clbits in the primary registers to the register that owns them and the
    # positions that they occupy in those registers
    clbit_map = {
        clbit: (name, clbit_idx)
        for name, creg in primary_cregs.items()
        for clbit_idx, clbit in enumerate(creg)
    }

    # A map between clbits in the post selection registers
    clbit_map_ps = {
        clbit: (name, clbit_idx)
        for name, creg in ps_cregs.items()
        for clbit_idx, clbit in enumerate(creg)
    }

    # A map between clbits in the pre selection registers
    clbit_map_pre = {
        clbit: (name, clbit_idx)
        for name, creg in pre_cregs.items()
        for clbit_idx, clbit in enumerate(creg)
    }

    qubit_map = {qubit: idx for idx, qubit in enumerate(dag.qubits)}

    measure_map: dict[int, tuple[str, int]] = {}
    measure_map_ps: dict[int, tuple[str, int]] = {}
    measure_map_pre: dict[int, tuple[str, int]] = {}

    for node in dag.topological_op_nodes():
        if node.op.name == "measure":  # pragma: no cover
            if clbit := clbit_map.get(node.cargs[0]):
                measure_map[qubit_map[node.qargs[0]]] = clbit
            elif clbit_ps := clbit_map_ps.get(node.cargs[0]):
                measure_map_ps[qubit_map[node.qargs[0]]] = clbit_ps
            elif clbit_pre := clbit_map_pre.get(node.cargs[0]):
                measure_map_pre[qubit_map[node.qargs[0]]] = clbit_pre

    return measure_map, measure_map_ps, measure_map_pre


def _validate_measure_maps(
    measure_map: dict[int, tuple[str, int]],
    selection_measure_map: dict[int, tuple[str, int]],
    selection_suffix: str,
    validation_mode: Literal["strict", "lenient"],
) -> None:
    """Validate measurement maps.

    This function checks that the measurement maps of the primary registers and those of the selection
    registers are compatible.

    Args:
        measure_map: The measurement map for the primary measurements.
        selection_measure_map: The measurement map for the selection measurements.
        selection_suffix: The suffix of the selection registers.
        validation_mode: The validation mode. "strict" requires equal length measure maps,
            while "lenient" only validates qubits that exist in both maps.

    Raise:
        ValueError: If validation fails based on the mode.
    """
    selection_type = "post selection" if selection_suffix == "_ps" else "pre selection"

    if validation_mode == "strict":
        # Post-selection mode: require equal length and all qubits present in both
        if len(measure_map) != len(selection_measure_map):
            raise ValueError(
                f"Found {len(measure_map)} measurements and {len(selection_measure_map)} "
                f"{selection_type} measurements."
            )

        for qubit_idx, (name, clbit_idx) in measure_map.items():
            try:
                name_selection, clbit_idx_selection = selection_measure_map[qubit_idx]
            except KeyError as key_err:
                raise ValueError(
                    f"Missing {selection_type} measurement on qubit {qubit_idx}."
                ) from key_err

            if (
                name_selection != (expected_name := name + selection_suffix)
                or clbit_idx != clbit_idx_selection
            ):
                raise ValueError(
                    f"Expected measurement on qubit {qubit_idx} writing to bit {clbit_idx} of creg "
                    f"{expected_name}, found measurement writing to bit {clbit_idx_selection} of creg "
                    f"{name_selection}."
                )
    else:
        # Pre-selection mode: only validate qubits that have selection measurements
        for qubit_idx, (name, clbit_idx) in selection_measure_map.items():
            if qubit_idx in measure_map:
                name_primary, clbit_idx_primary = measure_map[qubit_idx]
                expected_selection_name = name_primary + selection_suffix
                if name != expected_selection_name or clbit_idx != clbit_idx_primary:
                    raise ValueError(
                        f"{selection_type.capitalize()} measurement on qubit {qubit_idx} writes to bit {clbit_idx} of creg "
                        f"{name}, but expected to write to bit {clbit_idx_primary} of creg "
                        f"{expected_selection_name}."
                    )


def _get_edges(
    coupling_map: CouplingMap,
    *measure_maps: dict[int, tuple[str, int]],
) -> set[frozenset[int]]:
    """Get the set of edges that are relevant for edge-based post selection.

    An edge participates if both of its qubits appear in at least one of the
    supplied ``measure_maps`` (typically ``measure_map``, ``measure_map_ps``,
    ``measure_map_pre``). The mask functions filter further to the subset of
    edges where the specific measurement they need is available on both
    endpoints.
    """
    measured: set[int] = set()
    for mm in measure_maps:
        measured.update(mm.keys())
    return {
        frozenset(edge)
        for edge in coupling_map.get_edges()
        if edge[0] in measured and edge[1] in measured
    }


# Made with Bob
