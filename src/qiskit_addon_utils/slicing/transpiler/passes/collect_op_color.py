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

"""A transformation pass for collecting ops of a given edge color."""

from __future__ import annotations

from functools import partial

from qiskit.circuit import Instruction, QuantumCircuit
from qiskit.dagcircuit import DAGCircuit, DAGNode
from qiskit.transpiler.passes.optimization.collect_and_collapse import (
    CollectAndCollapse,
    collapse_to_operation,
    collect_using_filter_function,
)


class CollectOpColor(CollectAndCollapse):
    """Collects blocks of operations which act on the provided edges.

    The collected operations are replaced by a single block instruction.
    """

    def __init__(
        self,
        color_name: str,
        edges: set[tuple[int, int]],
        do_commutative_analysis: bool = False,
    ) -> None:
        """Initialize a ``CollectOpColor`` object.

        .. warning::
           The ``do_commutative_analysis`` keyword currently does not work as intended due to an
           open issue. Thus, setting it to ``True`` will not work.

        Args:
            color_name: The name of the color to consolidate into blocks. This is used to prefix the
                labels on the produced block instructions such that each block is labeled
                ``"{color_name}_slice"``.
            edges: The set of edges belonging to this color.
            do_commutative_analysis: If ``True``, exploits commutativity relations between nodes.
                Note also the warning above.
        """
        collect_function = partial(_collect_function, edges=edges)

        collapse_function = partial(
            collapse_to_operation,
            collapse_function=partial(_collapse_to_instruction, label=color_name),
        )

        super().__init__(
            collect_function=collect_function,
            collapse_function=collapse_function,
            do_commutative_analysis=do_commutative_analysis,
        )


def _is_color(dag: DAGCircuit, edges: set[tuple[int, int]], node: DAGNode) -> bool:
    """Specify whether a node holds an instruction of the specified colored edges."""
    qargs = tuple(dag.find_bit(qubit).index for qubit in node.qargs)
    return (bool(qargs in edges) or bool(qargs[::-1] in edges)) and node.op.name != "slice_op"


def _collapse_to_instruction(circuit: QuantumCircuit, label: str | None = None) -> Instruction:
    """Collapse function for turning a circuit into an instruction."""
    inst = circuit.to_instruction(label=f"{label}_slice")
    inst.name = "slice_op"
    return inst


def _collect_function(
    dag: DAGCircuit,
    edges: set[tuple[int, int]],
):
    """Collect instructions of a given color into blocks."""
    blocks = collect_using_filter_function(
        dag,
        filter_function=partial(_is_color, dag, edges),
        split_blocks=False,
        min_block_size=1,
        split_layers=False,
        collect_from_back=False,
    )
    return blocks
