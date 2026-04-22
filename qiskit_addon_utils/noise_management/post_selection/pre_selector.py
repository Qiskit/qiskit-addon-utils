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
"""Pre selector."""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from ..constants import DEFAULT_PRE_SELECTION_SUFFIX
from .post_selection_summary import PostSelectionSummary


class PreSelectionStrategy(str, Enum):
    """The supported pre selection strategies."""

    NODE = "node"
    """Discard every shot where one or more pre-selection measurements returned 0. Keep every other shot."""

    EDGE = "edge"
    """Discard every shot where there exists a pair of neighbouring qubits for which both of
    the pre-selection measurements returned 0. Keep every other shot."""


class PreSelector:
    """A class to process the results of quantum programs based on the outcome of pre selection measurements."""

    def __init__(self, summary: PostSelectionSummary):
        """Initialize a ``PreSelector`` object.

        Args:
            summary: A summary of the circuit being pre selected.
        """
        self._summary = summary

    @property
    def summary(self) -> PostSelectionSummary:
        """A summary of the circuit being pre selected."""
        return self._summary

    @classmethod
    def from_circuit(  # noqa: D417
        cls,
        circuit: QuantumCircuit,
        coupling_map: CouplingMap | list[tuple[int, int]],
        *,
        pre_selection_suffix: str = DEFAULT_PRE_SELECTION_SUFFIX,
    ) -> PreSelector:
        """Initialize from quantum circuits.

        Args:
            circuits: The circuits to process the results of.
            coupling_map: A coupling map or a list of tuples indicating pairs of neighboring qubits.
            pre_selection_suffix: A fixed suffix to append to the names of the classical registers when
                copying them.
        """
        coupling_map = (
            coupling_map
            if isinstance(coupling_map, CouplingMap)
            else CouplingMap(couplinglist=coupling_map)
        )

        summary = PostSelectionSummary.from_circuit(
            circuit,
            coupling_map,
            pre_selection_suffix=pre_selection_suffix,
            validation_mode="lenient",
        )
        return PreSelector(summary)

    def compute_mask(
        self,
        result: dict[str, NDArray[np.bool]],
        strategy: str | PreSelectionStrategy = PreSelectionStrategy.NODE,
    ) -> NDArray[np.bool]:
        """Compute boolean masks indicating what shots should be kept or discarded for the given result.

        This function examines the pre-selection measurements, identifying all those that returned 0
        (indicating improper initialization). The shots that should be kept are marked as ``True`` in the
        returned mask, those that should be discarded are marked as ``False``.

        By construction, the returned mask has the same shape as the arrays in corresponding result, but with one
        fewer dimension (the last axis of every array, over clbits, is not present in the mask).

        Args:
            result: The result to post-process. It must be a ``QuantumProgramResult`` containing a single item or
                a dictionary.
            strategy: The pre selection strategy used to process the result.
        """
        strategy = PreSelectionStrategy(strategy)
        if strategy == PreSelectionStrategy.NODE:
            _compute_mask = _compute_mask_by_node
        else:
            _compute_mask = _compute_mask_by_edge

        return _compute_mask(result, self.summary)


def _compute_mask_by_node(
    result: dict[str, NDArray[np.bool]], summary: PostSelectionSummary
) -> NDArray[np.bool]:
    """Compute the mask using a node-based pre selection strategy.

    Mark as ``False`` every shot where one or more pre-selection measurements returned 0,
    and as ``True`` every other shot.
    """
    _validate_result(result, summary)

    # Get shape from any primary register
    shape = result[next(iter(summary.primary_cregs))].shape[:-1]
    mask = np.ones(shape, dtype=bool)

    # For pre-selection, we expect the measurements to return 0 (good initialization)
    # Discard shots where any pre-selection measurement is 1 (bad initialization)
    for name_pre, clbit_idx_pre in summary.measure_map_pre.values():
        # Keep shots where pre-selection measurement is 0
        mask &= result[name_pre][..., clbit_idx_pre] == 0

    return mask


def _compute_mask_by_edge(
    result: dict[str, Any], summary: PostSelectionSummary
) -> NDArray[np.bool]:
    """Compute the mask using an edge-based pre selection strategy.

    Mark as ``False`` every shot where there exists a pair of neighbouring qubits for which
    both of the pre-selection measurements returned 0, and as ``True`` every other shot.
    """
    _validate_result(result, summary)

    # Get shape from any primary register
    shape = result[next(iter(summary.primary_cregs))].shape[:-1]
    mask = np.ones(shape, dtype=bool)

    # For each edge, discard shots where both qubits have pre-selection measurement of 1 (bad initialization)
    for qubit0_idx, qubit1_idx in summary.edges:
        # Use measure_map_pre to get the correct register and clbit index for pre-selection measurements
        name0_pre, clbit0_idx_pre = summary.measure_map_pre[qubit0_idx]
        name1_pre, clbit1_idx_pre = summary.measure_map_pre[qubit1_idx]

        # Keep shots where at least one of the pre-selection measurements is 0 (good initialization)
        mask &= (result[name0_pre][..., clbit0_idx_pre] == 0) | (
            result[name1_pre][..., clbit1_idx_pre] == 0
        )

    return mask


def _validate_result(result: dict[str, NDArray[np.bool]], summary: PostSelectionSummary):
    """Validate a result against a summary.

    Args:
        result: A result to post-process.
        summary: A summary to validate the given result.

    Raise:
        ValueError: If ``result`` contains more than one datum.
        ValueError: If ``result`` does not contain all of the required registers.
        ValueError: If ``result`` contains arrays of inconsistent shapes.
    """
    primary_cregs = summary.primary_cregs
    pre_selection_suffix = summary.pre_selection_suffix
    cregs = summary.primary_cregs.union(name + pre_selection_suffix for name in primary_cregs)

    for name in cregs:
        if result.get(name) is None:
            raise ValueError(f"Result does not contain creg '{name}'.")

    if len(set(result[name].shape[:-1] for name in cregs)) > 1:
        raise ValueError("Result contains arrays of inconsistent shapes.")


# Made with Bob
