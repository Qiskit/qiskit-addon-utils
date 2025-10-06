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

"""Utility functions for calculating expectation values of observables."""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import Pauli, PauliLindbladMap, PauliList, SparsePauliOp


def calculate_expectation_values(
    meas_results: np.typing.NDArray[np.bool_],
    reverser: dict[Pauli, list[SparsePauliOp]],
    signs: np.typing.NDArray[np.bool_] | None = None,
    gamma: float | None = None,
) -> tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]]:
    r"""Return the expectation values of the observables represented by the reverser.

    If signs are given, mitigated expectation values of observables are calculated.
    For an observable O, the mitigated expectation value is calculated as:
    :math:`<\tilde{O}> = -1^k * \gamma * <O>`
    while k refers to the number of layered noise injections into the circuit.
    If gamma is not given, calculate it using only the signs.

    Args:
        meas_results: measured state for each qubit, for each shot, for all the parameters and bases measured.
        reverser: Map for each measured basis to the relevant Paulis and their coefficients in each observable.
        signs: Whether a noise was injected for each layer at each sampling batch.
        gamma: The gamma factor associated with the mitigation.

    Returns:
        The expectation value and variance for each observable and each parameter vector.
    """
    bases = list(reverser.keys())

    meas_res_reshape, signs_reshape, param_shape_flat, param_shape = _flatten_parameters_tensor(
        meas_results, signs, bases
    )

    # get number of observables from the reverser
    num_obs = len(reverser[bases[0]])
    exp_vals = np.zeros((num_obs, param_shape_flat), dtype=float)
    exp_vars = np.zeros((num_obs, param_shape_flat), dtype=float)
    # if signs is supplied but gamma is not, calculate gamma using the signs
    if signs is not None and gamma is None:
        gamma_per_base_per_parameter = calculate_gamma_by_signs(signs_reshape)

    for i, (results_per_base, base) in enumerate(zip(meas_res_reshape, bases)):
        measured_pauli_group = reverser[base]
        for j, results_per_parameter in enumerate(results_per_base):
            counts = _bits2counts(results_per_parameter)
            sign = (-1 if np.sum(signs_reshape[i, j]) % 2 == 1 else 1) if signs is not None else 1
            for obs_index in range(num_obs):
                obs_pauli_op = measured_pauli_group[obs_index]
                if obs_pauli_op:
                    expval, var = calculate_expectation_values_for_observable(counts, obs_pauli_op)
                    # If signs are given, calculate the mitigated expectation value
                    if signs is not None:
                        if gamma is not None:
                            expval *= gamma
                        else:
                            expval *= gamma_per_base_per_parameter[i, j]
                    exp_vals[obs_index, j] += sign * expval.real
                    exp_vars[obs_index, j] += var.real

    # reshape back to the original parameters shape
    output_shape = list(param_shape)
    output_shape.insert(0, num_obs)
    exp_vals = exp_vals.reshape(output_shape)
    exp_vars = exp_vars.reshape(output_shape)
    return exp_vals, exp_vars


def calculate_gamma_by_noise_map(
    noise_map: dict[str, PauliLindbladMap], layer_map: dict[str, str]
) -> float:
    """Calculate the gamma factor for a given noise and layers of a circuit.

    Args:
        noise_map: The learned noise for each unique gates layer in the circuit.
        layer_map: A map for each gates layer instance in the circuit to its layer type id.

    Returns:
        The gamma factor.
    """
    layers_gamma = {}
    for layer_key, layer_noise in noise_map.items():
        layers_gamma[layer_key] = layer_noise.inverse().gamma()

    gamma = 1
    for layer_type in layer_map.values():
        gamma *= layers_gamma[layer_type]
    return gamma


def calculate_gamma_by_signs(signs: np.typing.NDArray[np.bool_]) -> np.typing.NDArray[np.float64]:
    r"""Calculate the gamma factor for a given invocation of noise injection.

     Gamma is calculated by:
    :math:`\gamma = \frac{k}{(k_{even} - k_{odd})}`
    while k refers to the total number of layered noise injections into the circuit, and :math:`k_{even},k_{odd}`
    refers to the number of invocations where an even (or odd) number of layered noise was injected into the circuit.

    Args:
        signs: Whether a noise was injection for each layer in each circuit randomization run.
                assumes that the array is ordered as [bases, parameters, randomizations, layers].

    Returns:
        The gamma factor for each base and each parameter.
    """
    gamma_per_base_per_parameter = np.zeros((signs.shape[0], signs.shape[1]), dtype=float)

    for base_index in range(len(signs)):
        for parameter_index in range(len(signs[base_index])):
            count_parameter_even_injections = 0
            count_parameter_odd_injections = 0
            for randomization in signs[base_index, parameter_index]:
                sign = -1 if np.sum(randomization) % 2 == 1 else 1
                if sign == 1:
                    count_parameter_even_injections += 1
                else:
                    count_parameter_odd_injections += 1
            total_samples = count_parameter_even_injections + count_parameter_odd_injections
            gamma = total_samples / (
                count_parameter_even_injections - count_parameter_odd_injections
            )
            gamma_per_base_per_parameter[base_index, parameter_index] = gamma
    return gamma_per_base_per_parameter


def calculate_expectation_values_for_observable(
    counts: dict[str, int], pauli_op: SparsePauliOp
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate expectation value of a SparsePauliOp.

    Args:
        counts: The measurements outcome formatted as a dict with the states as keys and the number of times each state was measured as values.
        pauli_op: The operation to calculate the expectation value of.

    Returns:
        The expectation value and variance for each Pauli operation of the observable.
    """
    expvals, variances = pauli_expval_with_variance(counts, pauli_op.paulis)
    # Multiply by the operator coefficients
    expval = np.dot(expvals, pauli_op.coeffs)
    var = np.dot(variances, pauli_op.coeffs**2)
    return expval, var


def pauli_expval_with_variance(
    counts: dict[str, int], paulis: PauliList
) -> tuple[np.ndarray, np.ndarray]:
    """Return array of expval and variance pairs for input Paulis.

    Note: All non-identity Pauli's are treated as Z-paulis, assuming
    that basis rotations have been applied to convert them to the
    diagonal basis.
    """
    # Diag indices
    size = len(paulis)
    diag_inds = _paulis2inds(paulis)

    expvals = np.zeros(size, dtype=float)
    denom = 0  # Total shots for counts dict
    for bin_outcome, freq in counts.items():
        outcome = int(bin_outcome, 2)
        denom += freq
        for k in range(size):
            coeff = (-1) ** _parity(diag_inds[k] & outcome)
            expvals[k] += freq * coeff

    # Divide by total shots
    expvals /= denom

    # Compute variance
    variances = 1 - expvals**2
    return expvals, variances


def _flatten_parameters_tensor(meas_results, signs, bases):
    """Fix the dimensions of the measurements and sign results objects to the following format.

    for the measurement results: [bases, parameters, randomizations, shots, measurement instructions].
    for the sign results: [bases, parameters, randomizations, layers].
    """
    if len(bases) == 1:
        meas_results = np.array([meas_results])
        signs = np.array([signs]) if signs is not None else None

    reshaped_signs = None
    if len(meas_results.shape) >= 5:
        parameter_shape = meas_results.shape[2:-2]
        parameters_flatten_size = np.prod(parameter_shape)
        reshaped_meas_results = meas_results.reshape(
            meas_results.shape[0],  # bases
            parameters_flatten_size,  # parameters
            meas_results.shape[1],  # randomizations
            meas_results.shape[-2],  # shots
            meas_results.shape[-1],  # measurement instructions
        )
        if signs is not None:
            reshaped_signs = signs.reshape(
                signs.shape[0],  # bases
                parameters_flatten_size,  # parameters
                signs.shape[1],  # randomizations
                signs.shape[-1],  # layers
            )
    elif len(meas_results.shape) == 4:
        # no parameters were used
        parameters_flatten_size = 1
        parameter_shape = [1]
        reshaped_meas_results = np.expand_dims(meas_results, axis=1)
        if signs is not None:
            reshaped_signs = np.expand_dims(signs, axis=1)
    else:
        parameters_flatten_size = 1
        parameter_shape = [1]
        # only a single randomization was used and no parameters were used
        reshaped_meas_results = np.expand_dims(meas_results, axis=1)
        reshaped_meas_results = np.expand_dims(reshaped_meas_results, axis=2)
        if signs is not None:
            reshaped_signs = np.expand_dims(signs, axis=1)
            reshaped_signs = np.expand_dims(reshaped_signs, axis=2)

    return reshaped_meas_results, reshaped_signs, parameters_flatten_size, parameter_shape


def _bits2counts(meas_results: list) -> dict[str, int]:
    """Convert the results represented by a boolean to counts dictionary.

    The conversion if done for each bit in each shot, for each randomization.
    """
    counts: dict[str, int] = {}
    for randomization in meas_results:
        for shot in randomization:
            # skip post-selected shots
            if shot[0] is not None:
                meas_state = "".join(["1" if meas_res else "0" for meas_res in shot])
                if meas_state in counts:
                    counts[meas_state] += 1
                else:
                    counts[meas_state] = 1
    return counts


def _paulis2inds(paulis: PauliList) -> list[int]:
    """Convert PauliList to diagonal integers.

    These are integer representations of the binary string with a
    1 where there are Paulis, and 0 where there are identities.
    """
    # Treat Z, X, Y the same
    nonid = paulis.z | paulis.x

    inds = [0] * paulis.size
    # bits are packed into uint8 in little endian
    # e.g., i-th bit corresponds to coefficient 2^i
    packed_vals = np.packbits(nonid, axis=1, bitorder="little")
    for i, vals in enumerate(packed_vals):
        for j, val in enumerate(vals):
            inds[i] += val.item() * (1 << (8 * j))
    return inds


def _parity(integer: int) -> int:
    """Return the parity of an integer."""
    return bin(integer).count("1") % 2
