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

from collections.abc import Sequence

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.primitives.containers.estimator_pub import EstimatorPub, EstimatorPubLike
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.quantum_info import SparsePauliOp


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
        self.observables = self.remove_layout_from_observables(observables, circuit)
        self.parameters = parameters

    def __eq__(self, other):
        """Check if two ExecutionInputs objects are equal."""
        if self.parameters is not None:
            if other.parameters is None:
                return False
            return (
                self.circuit == other.circuit
                and self.observables.equivalent(other.observables)
                and self.parameters.data == other.parameters.data
            )
        if other.parameters is not None and self.parameters is None:
            return False
        return self.circuit == other.circuit and self.observables.equivalent(other.observables)

    @classmethod
    def coerce(cls, inputs: EstimatorPubLike, parameters_outer_product: bool = True):
        """Coerce an ExecutionInputs object from a EstimatorPubLike object.

        If parameters_outer_product is ``True``, parameters binding variable must be an array with no empty shape. The shape of the observables and parameters will be broadcasted to create an outer-product calculation.
        """
        if isinstance(inputs, ExecutionInputs):
            return inputs

        if not isinstance(inputs, Sequence) or (
            isinstance(inputs, Sequence) and not (1 < len(inputs) < 4)
        ):
            raise ValueError("inputs must be a tuple with length 2 or 3")

        original_parameters_shape: tuple = tuple()
        if parameters_outer_product and len(inputs) == 3:
            observables, parameters = inputs[1], inputs[2]
            parameters = np.array(parameters)
            if parameters.shape == ():
                raise ValueError(
                    "parameters must be an array with non empty shape if broadcast_parameters is True"
                )
            # handle the case of 0 dimension input
            observables_array = (
                ObservablesArray(observables)
                if ObservablesArray(observables).shape != ()
                else ObservablesArray([observables])
            )
            original_parameters_shape = parameters.shape

            # The last index of the parameters binding is the circuit parameters size
            new_obs_arr = observables_array.reshape(
                observables_array.shape + (1,) * len(parameters.shape[:-1])
            )
            new_params = parameters.reshape((1,) * len(observables_array.shape) + parameters.shape)
            inputs = (inputs[0], new_obs_arr, new_params)
        try:
            estimator_pub = EstimatorPub.coerce(inputs)
        except ValueError as err:
            raise ValueError("inputs must be a valid EstimatorPubLike tuple") from err

        # handle the case of 0 dimension input
        observables_array = (
            estimator_pub.observables
            if estimator_pub.observables.shape != ()
            else ObservablesArray([estimator_pub.observables.tolist()])
        )
        if estimator_pub.parameter_values.shape == ():
            return cls(estimator_pub.circuit, observables_array)
        parameters_arr = estimator_pub.parameter_values.as_array()
        if parameters_outer_product:
            parameters_arr = parameters_arr.reshape(original_parameters_shape)
        return cls(estimator_pub.circuit, observables_array, parameters_arr)

    @classmethod
    def remove_layout_from_observables(cls, observables: ObservablesArray, circuit: QuantumCircuit):
        """Remove the layout from the observables in the array."""
        if circuit.layout is None:
            return observables

        qubits_layout = circuit.layout.final_index_layout()
        # check if the observables have not been applied the circuit layout
        if observables.num_qubits != circuit.num_qubits or observables.num_qubits == len(
            qubits_layout
        ):
            return observables

        new_observables = []
        for observable in observables.ravel():
            new_obs = {}
            for pauli, coeff in observable.items():
                # pauli string order is reversed
                new_pauli = "".join([pauli[::-1][q] for q in qubits_layout])
                new_obs[new_pauli] = coeff
            new_observables.append(new_obs)

        return ObservablesArray(new_observables).reshape(observables.shape)

    @classmethod
    def observables_array_to_1d_sparse_obs(cls, observables_array: ObservablesArray):
        """Convert an observables array to a 1D SparsePauliOp."""
        # TODO: Convert to SparseObservable instead of SparsePauliOp once all the inner code fully support SparseObservable
        return [
            SparsePauliOp.from_sparse_observable(sparse_obs)
            for sparse_obs in observables_array.ravel().sparse_observables_array()
        ]


InputsLike = EstimatorPubLike | ExecutionInputs
"""Types that can be natively converted to an ExecutionInputs object."""
