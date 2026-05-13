# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Inputs class for various quantum error mitigation methods."""

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.primitives.containers.observables_array import ObservablesArray, ObservablesArrayLike
from qiskit.quantum_info import SparsePauliOp

InputsLike = (
    tuple[QuantumCircuit, ObservablesArrayLike]
    | tuple[QuantumCircuit, ObservablesArrayLike, np.ndarray]
)
"""Types that can be natively converted to an ExecutionInputs object."""


class ExecutionInputs:
    """Inputs class for various quantum error mitigation methods.

    The ExecutionInputs should contain at least a circuit in ISA format and an array of observables as a ObservablesArray object.
    In addition, it can contain an array of parameters to bind the circuit against. The parameters can be specified as a single array-like
    object where the last index is over circuit ``Parameter`` objects, or omitted if the circuit has no ``Parameter`` objects
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        observables: ObservablesArray,
        parameters: np.ndarray | None = None,
    ):
        """Initialize an ExecutionInputs object.

        Args:
            circuit: The circuit to execute, in ISA format.
            observables: The observables which specify the expectation values to estimate.
            parameters: The parameters to bind the circuit against.
        """
        self.circuit = circuit
        # TODO: remove the convertion to SparsePauliOp once all the inner code fully support SparseObservable
        self.observables = [
            SparsePauliOp.from_sparse_observable(sparse_obs)
            for sparse_obs in observables.sparse_observables_array()
        ]
        # clean observable from layout
        if circuit.layout:
            qubits_layout = circuit.layout.final_index_layout()
            new_observables = []
            for observable in self.observables:
                # check if layout should be removed
                if observable.num_qubits == circuit.num_qubits and observable.num_qubits != len(
                    qubits_layout
                ):
                    new_elements = []
                    for element in observable.paulis:
                        new_elements.append(element[circuit.layout.final_index_layout()])
                    new_observables.append(SparsePauliOp(new_elements, observable.coeffs))
                else:
                    new_observables.append(observable)
            self.observables = new_observables

        self.parameters = parameters

    def __eq__(self, other):
        """Check if two ExecutionInputs objects are equal."""
        if self.parameters is not None:
            if other.parameters is None:
                return False
            return (
                self.circuit == other.circuit
                and self.observables == other.observables
                and self.parameters == other.parameters
            )
        if other.parameters is not None and self.parameters is None:
            return False
        return self.circuit == other.circuit and self.observables == other.observables

    @classmethod
    def coerce(cls, inputs: InputsLike):
        """Coerce an ExecutionInputs object from a InputsLike object."""
        if isinstance(inputs, ExecutionInputs):
            return inputs
        if not isinstance(inputs, tuple) or (
            isinstance(inputs, tuple) and not (1 < len(inputs) < 4)
        ):
            raise ValueError("inputs must be a valid InputsLike tuple")
        if len(inputs) < 3:
            return cls(inputs[0], ObservablesArray(inputs[1]))
        return cls(inputs[0], ObservablesArray(inputs[1]), inputs[2])
