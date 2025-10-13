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
from typing import Any

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from ..constants import DEFAULT_POST_SELECTION_SUFFIX
from .post_selection_summary import PostSelectionSummary


class PostSelectionStrategy(str, Enum):
    """The supported post selection strategies."""

    NODE = "node"
    """Discard every shot where one or more bits failed to flip. Keep every other shot."""

    EDGE = "edge"
    """Discard every shot where there exists a pair of neighbouring qubits for which both of
    the bits failed to flip. Keep every other shot."""


class PostSelector:
    """A class to process the results of quantum programs based on the outcome of post selection measurements."""

    def __init__(self, summary: PostSelectionSummary):
        """Initialize a ``PostSelector`` object.

        Args:
            summary: A summary of the circuit being post selected.
        """
        self._summary = summary

    @property
    def summary(self) -> PostSelectionSummary:
        """A summary of the circuit being post selected."""
        return self._summary

    @classmethod
    def from_circuit(  # noqa: D417
        cls,
        circuit: QuantumCircuit,
        coupling_map: CouplingMap | list[tuple[int, int]],
        *,
        post_selection_suffix: str = DEFAULT_POST_SELECTION_SUFFIX,
    ) -> PostSelector:
        """Initialize from quantum circuits.

        Args:
            circuits: The circuits to process the results of.
            coupling_map: A coupling map or a list of tuples indicating pairs of neighboring qubits.
            post_selection_suffix: A fixed suffix to append to the names of the classical registers when
                copying them.
        """
        coupling_map = (
            coupling_map
            if isinstance(coupling_map, CouplingMap)
            else CouplingMap(couplinglist=coupling_map)
        )

        summary = PostSelectionSummary.from_circuit(
            circuit, coupling_map, post_selection_suffix=post_selection_suffix
        )
        return PostSelector(summary)

    def compute_mask(
        self,
        result: dict[str, NDArray[np.bool]],
        strategy: str | PostSelectionStrategy = PostSelectionStrategy.NODE,
    ) -> NDArray[np.bool]:
        """Compute boolean masks indicating what shots should be kept or discarded for the given result.

        This function compares the bits returned by every pair of measurement and post selection measurement,
        identifying all those that failed to flip. The shots that should be kept are marked as ``True`` in the
        returned mask, those that should be discarded are marked as ``False``.

        By construction, the returned mask has the same shape as the arrays in corresponding result, but with one
        fewer dimension (the last axis of every array, over clbits, is not present in the mask).

        Args:
            result: The result to post-process. It must be a ``QuantumProgramResult`` containing a single item or
                a dictionary.
            strategy: The post selection strategy used to process the result.
        """
        strategy = PostSelectionStrategy(strategy)
        if strategy == PostSelectionStrategy.NODE:
            _compute_mask = _compute_mask_by_node
        else:
            _compute_mask = _compute_mask_by_edge

        return _compute_mask(result, self.summary)


def _compute_mask_by_node(result: dict[str, NDArray[np.bool]], summary: PostSelectionSummary):
    """Compute the mask using a node-based post selection strategy.

    Mark as ``False`` every shot where one or more results failed to flip, and as ``True``
    every other shot.
    """
    _validate_result(result, summary)

    shape = result[next(iter(summary.primary_cregs))].shape[:-1]
    mask = np.ones(shape, dtype=bool)
    for name, clbit_idx in summary.measure_map.values():
        name_ps = name + summary.post_selection_suffix
        mask &= result[name][..., clbit_idx] != result[name_ps][..., clbit_idx]
    return mask


def _compute_mask_by_edge(result: dict[str, Any], summary: PostSelectionSummary):
    """Compute the mask using an edge-based post selection strategy.

    Mark as ``False`` every shot where there exists a pair of neighbouring qubits for which
    both of the results failed to flip, and as ``True`` every other shot.
    """
    _validate_result(result, summary)

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
    post_selection_suffix = summary.post_selection_suffix
    cregs = summary.primary_cregs.union(name + post_selection_suffix for name in primary_cregs)

    for name in cregs:
        if result.get(name) is None:
            raise ValueError(f"Result does not contain creg '{name}'.")

    if len(set(result[name].shape[:-1] for name in cregs)) > 1:
        raise ValueError("Result contains arrays of inconsistent shapes.")
