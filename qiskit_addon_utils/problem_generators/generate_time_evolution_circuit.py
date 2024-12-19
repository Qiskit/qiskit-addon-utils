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

"""Utility functions for generating time evolution circuits."""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import EvolutionSynthesis


def generate_time_evolution_circuit(
    operator: SparsePauliOp,
    *,
    synthesis: EvolutionSynthesis | None = None,
    time: float = 1.0,
) -> QuantumCircuit:
    """Model the time evolution of an operator.

    .. plot::
        :alt: Output from the previous code.
        :include-source:

        >>> from qiskit.quantum_info import SparsePauliOp
        >>> from qiskit.synthesis import SuzukiTrotter
        >>> from qiskit.transpiler import CouplingMap
        >>> from qiskit_addon_utils.problem_generators import (
        ...     PauliOrderStrategy,
        ...     generate_xyz_hamiltonian,
        ...     generate_time_evolution_circuit,
        ... )

        >>> coupling_map = CouplingMap.from_line(6)
        >>> hamiltonian = generate_xyz_hamiltonian(
        ...     coupling_map,
        ...     coupling_constants=(0.4, 0.4, 0.0),
        ...     ext_magnetic_field=(0.0, 0.0, 0.6),
        ...     pauli_order_strategy=PauliOrderStrategy.InteractionThenColorZigZag,
        ... )

        >>> circ = generate_time_evolution_circuit(
        ...     hamiltonian, synthesis=SuzukiTrotter(order=2, reps=2), time=2.0
        ... )
        >>> _ = circ.draw("mpl", fold=-1)

    Args:
        operator: The operator for which to model the time evolution.
        synthesis: A synthesis strategy. If ``None``, the default synthesis is the Lie-Trotter
            product formula with a single repetition.
        time: The evolution time.

    Returns:
        A :class:`~qiskit.circuit.QuantumCircuit` implementing a time-evolved operator.
    """
    # Generate quantum circuit describing the time evolution of Hamiltonian
    circuit = QuantumCircuit(operator.num_qubits)
    circuit.append(
        PauliEvolutionGate(operator, time=time, synthesis=synthesis),
        qargs=circuit.qubits,
    )

    circuit_out = (
        circuit.decompose()  # Experimental Pauli synthesis breaks if not decomposed
    )

    return circuit_out
