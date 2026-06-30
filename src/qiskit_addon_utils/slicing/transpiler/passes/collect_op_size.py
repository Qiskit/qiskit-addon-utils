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

"""A transformation pass for collecting ops of a given size."""

from __future__ import annotations

from functools import partial

from qiskit.circuit import Instruction, QuantumCircuit
from qiskit.dagcircuit import DAGCircuit, DAGNode
from qiskit.transpiler.passes.optimization.collect_and_collapse import (
    CollectAndCollapse,
    collapse_to_operation,
    collect_using_filter_function,
)


class CollectOpSize(CollectAndCollapse):
    """Collects blocks of the specified size and replaces them by a single block instruction."""

    def __init__(
        self,
        size: int,
        do_commutative_analysis: bool = False,
    ) -> None:
        """Initialize a ``CollectOpSize`` object.

        .. warning::
           The ``do_commutative_analysis`` keyword currently does not work as intended due to an
           open issue. Thus, setting it to ``True`` will not work.

        Args:
            size: The size of operations to collect.
            do_commutative_analysis: If ``True``, exploits commutativity relations between nodes.
                Note also the warning above.
        """
        collect_function = partial(_collect_function, size=size)

        collapse_function = partial(
            collapse_to_operation,
            collapse_function=partial(_collapse_to_instruction, label=f"size_{size}"),
        )

        super().__init__(
            collect_function=collect_function,
            collapse_function=collapse_function,
            do_commutative_analysis=do_commutative_analysis,
        )


def _is_size(size: int, node: DAGNode) -> bool:
    """Specify whether a node holds an instruction of the specified size."""
    return bool(len(node.qargs) == size)


def _collapse_to_instruction(circuit: QuantumCircuit, label: str | None = None) -> Instruction:
    """Collapse function for turning a circuit into an instruction."""
    inst = circuit.to_instruction(label=f"{label}_slice")
    inst.name = "slice_op"
    return inst


def _collect_function(
    dag: DAGCircuit,
    size: int,
):
    """Collect instructions of a given size into blocks."""
    blocks = collect_using_filter_function(
        dag,
        filter_function=partial(_is_size, size),
        split_blocks=False,
        min_block_size=1,
        split_layers=True,
        collect_from_back=False,
    )
    return blocks
