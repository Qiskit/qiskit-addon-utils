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
"""Post selector."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from ..constants import DEFAULT_POST_SELECTION_SUFFIX, DEFAULT_PRE_SELECTION_SUFFIX
from .post_selection_summary import PostSelectionSummary


class SelectionStrategy(str, Enum):
    """The supported selection strategies."""

    NODE = "node"
    """Discard every shot where one or more checks fail. Keep every other shot."""

    EDGE = "edge"
    """Discard every shot where there exists a pair of neighbouring qubits for which both
    checks fail. Keep every other shot."""


# Backwards compatibility alias
PostSelectionStrategy = SelectionStrategy


class PostSelector:
    """A class to process the results of quantum programs based on selection measurements.

    This class supports both post-selection (measurements at the end of the circuit to verify
    bit flips) and pre-selection (measurements at the beginning to verify initialization).
    It can handle circuits with either type of selection, or both simultaneously.
    """

    def __init__(self, summary: PostSelectionSummary):
        """Initialize a ``PostSelector`` object.

        Args:
            summary: A summary of the circuit being selected.
        """
        self._summary = summary

    @property
    def summary(self) -> PostSelectionSummary:
        """A summary of the circuit being selected."""
        return self._summary

    @classmethod
    def from_circuit(
        cls,
        circuit: QuantumCircuit,
        coupling_map: CouplingMap | list[tuple[int, int]],
        *,
        post_selection_suffix: str = DEFAULT_POST_SELECTION_SUFFIX,
        pre_selection_suffix: str = DEFAULT_PRE_SELECTION_SUFFIX,
        validation_mode: Literal["strict", "lenient"] = "strict",
    ) -> PostSelector:
        """Initialize from quantum circuits.

        Args:
            circuit: The circuit to process the results of.
            coupling_map: A coupling map or a list of tuples indicating pairs of neighboring qubits.
            post_selection_suffix: A fixed suffix for post-selection classical registers.
            pre_selection_suffix: A fixed suffix for pre-selection classical registers.
            validation_mode: The validation mode for post-selection. "strict" requires all qubits
                to have post-selection measurements, "lenient" allows partial coverage.
        """
        coupling_map = (
            coupling_map
            if isinstance(coupling_map, CouplingMap)
            else CouplingMap(couplinglist=coupling_map)
        )

        summary = PostSelectionSummary.from_circuit(
            circuit,
            coupling_map,
            post_selection_suffix=post_selection_suffix,
            pre_selection_suffix=pre_selection_suffix,
            validation_mode=validation_mode,
        )
        return PostSelector(summary)

    def compute_mask(
        self,
        result: dict[str, NDArray[np.bool]],
        strategy: str | SelectionStrategy = SelectionStrategy.NODE,
        *,
        mode: Literal["post", "pre", "both"] = "post",
    ) -> NDArray[np.bool]:
        """Compute boolean masks indicating what shots should be kept or discarded.

        This function examines selection measurements (post-selection, pre-selection, or both)
        and identifies shots that should be discarded based on the specified strategy.

        By construction, the returned mask has the same shape as the arrays in the result, but with one
        fewer dimension (the last axis of every array, over clbits, is not present in the mask).

        Args:
            result: The result to post-process. Must be a dictionary mapping register names to
                boolean arrays.
            strategy: The selection strategy ("node" or "edge").
            mode: Which type of selection to apply:
                - "post": Apply post-selection only (default, checks bit flips)
                - "pre": Apply pre-selection only (checks initialization)
                - "both": Apply both pre and post-selection (combined mask)

        Returns:
            A boolean mask where True indicates shots to keep, False indicates shots to discard.

        Raises:
            ValueError: If the requested mode is not available (e.g., no pre-selection measurements
                in the circuit but mode="pre" was requested).
        """
        strategy = SelectionStrategy(strategy)

        if mode == "post":
            if not self.summary.measure_map_ps:
                raise ValueError("No post-selection measurements found in circuit.")
            if strategy == SelectionStrategy.NODE:
                return _compute_post_mask_by_node(result, self.summary)
            return _compute_post_mask_by_edge(result, self.summary)
        if mode == "pre":
            if not self.summary.measure_map_pre:
                raise ValueError("No pre-selection measurements found in circuit.")
            if strategy == SelectionStrategy.NODE:
                return _compute_pre_mask_by_node(result, self.summary)
            return _compute_pre_mask_by_edge(result, self.summary)
        # mode == "both"
        if not self.summary.measure_map_ps:
            raise ValueError("No post-selection measurements found in circuit.")
        if not self.summary.measure_map_pre:
            raise ValueError("No pre-selection measurements found in circuit.")

        # Compute both masks and combine with logical AND
        if strategy == SelectionStrategy.NODE:
            pre_mask = _compute_pre_mask_by_node(result, self.summary)
            post_mask = _compute_post_mask_by_node(result, self.summary)
        else:
            pre_mask = _compute_pre_mask_by_edge(result, self.summary)
            post_mask = _compute_post_mask_by_edge(result, self.summary)

        return pre_mask & post_mask


def _compute_post_mask_by_node(
    result: dict[str, NDArray[np.bool]], summary: PostSelectionSummary
) -> NDArray[np.bool]:
    """Compute the mask using a node-based post selection strategy.

    Mark as ``False`` every shot where one or more results failed to flip, and as ``True``
    every other shot.
    """
    _validate_post_result(result, summary)

    shape = result[next(iter(summary.primary_cregs))].shape[:-1]
    mask = np.ones(shape, dtype=bool)
    for name, clbit_idx in summary.measure_map.values():
        name_ps = name + summary.post_selection_suffix
        mask &= result[name][..., clbit_idx] != result[name_ps][..., clbit_idx]
    return mask


def _compute_post_mask_by_edge(
    result: dict[str, Any], summary: PostSelectionSummary
) -> NDArray[np.bool]:
    """Compute the mask using an edge-based post selection strategy.

    Mark as ``False`` every shot where there exists a pair of neighbouring qubits for which
    both of the results failed to flip, and as ``True`` every other shot.
    """
    _validate_post_result(result, summary)

    shape = result[next(iter(summary.primary_cregs))].shape[:-1]
    mask = np.ones(shape, dtype=bool)
    for qubit0_idx, qubit1_idx in summary.edges:
        name0, clbit0_idx = summary.measure_map[qubit0_idx]
        name0_ps = name0 + summary.post_selection_suffix

        name1, clbit1_idx = summary.measure_map[qubit1_idx]
        name1_ps = name1 + summary.post_selection_suffix

        mask &= (result[name0][..., clbit0_idx] != result[name0_ps][..., clbit0_idx]) | (
            result[name1][..., clbit1_idx] != result[name1_ps][..., clbit1_idx]
        )
    return mask


def _compute_pre_mask_by_node(
    result: dict[str, NDArray[np.bool]], summary: PostSelectionSummary
) -> NDArray[np.bool]:
    """Compute the mask using a node-based pre selection strategy.

    Mark as ``False`` every shot where one or more pre-selection measurements returned 1 (bad
    initialization), and as ``True`` every other shot.
    """
    _validate_pre_result(result, summary)

    # Get shape from any primary register
    shape = result[next(iter(summary.primary_cregs))].shape[:-1]
    mask = np.ones(shape, dtype=bool)

    # For pre-selection, we expect the measurements to return 0 (good initialization)
    # Discard shots where any pre-selection measurement is 1 (bad initialization)
    for name_pre, clbit_idx_pre in summary.measure_map_pre.values():
        # Keep shots where pre-selection measurement is 0
        mask &= result[name_pre][..., clbit_idx_pre] == 0

    return mask


def _compute_pre_mask_by_edge(
    result: dict[str, Any], summary: PostSelectionSummary
) -> NDArray[np.bool]:
    """Compute the mask using an edge-based pre selection strategy.

    Mark as ``False`` every shot where there exists a pair of neighbouring qubits for which
    both of the pre-selection measurements returned 1 (bad initialization), and as ``True``
    every other shot.
    """
    _validate_pre_result(result, summary)

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


def _validate_post_result(result: dict[str, NDArray[np.bool]], summary: PostSelectionSummary):
    """Validate a result against a summary for post-selection.

    Args:
        result: A result to post-process.
        summary: A summary to validate the given result.

    Raise:
        ValueError: If ``result`` does not contain all of the required registers.
        ValueError: If ``result`` contains arrays of inconsistent shapes.
    """
    primary_cregs = summary.primary_cregs
    post_selection_suffix = summary.post_selection_suffix
    cregs = summary.primary_cregs.union(name + post_selection_suffix for name in primary_cregs)

    for name in cregs:
        if result.get(name) is None:
            raise ValueError(f"Result does not contain creg '{name}'.")

    if len(set(result[name].shape[:-1] for name in cregs)) > 1:
        raise ValueError("Result contains arrays of inconsistent shapes.")


def _validate_pre_result(result: dict[str, NDArray[np.bool]], summary: PostSelectionSummary):
    """Validate a result against a summary for pre-selection.

    Args:
        result: A result to post-process.
        summary: A summary to validate the given result.

    Raise:
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
