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

"""Utility functions for calculating expectation values of observables."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from qiskit.primitives import BitArray
from qiskit.quantum_info import Pauli, SparseObservable, SparsePauliOp


def executor_expectation_values(
    # positional-only arguments: these canNOT be specified as keyword arguments, meaning we can
    # rename them without breaking API
    bool_array: np.ndarray[tuple[int, ...], np.dtype[np.bool]],
    basis_dict: dict[Pauli, list[SparsePauliOp | None]],
    /,
    # positional or keyword arguments
    meas_basis_axis: int | None = None,
    *,
    # keyword-only arguments: these can ONLY be specified as keyword arguments. Renaming them breaks
    # API, but their order does not matter.
    avg_axis: int | tuple[int, ...] | None = None,
    measurement_flips: np.ndarray[tuple[int, ...], np.dtype[np.bool]] | None = None,
    pauli_signs: np.ndarray[tuple[int, ...], np.dtype[np.bool]] | None = None,
    postselect_mask: np.ndarray[tuple[int, ...], np.dtype[np.bool]] | None = None,
    gamma_factor: float | None = None,
    rescale_factors: Sequence[Sequence[Sequence[float]]] | None = None,
):
    """Computes expectation values from boolean data and metadata, aiming for compatibility with the components of a result from ``Executor`` in ``qiskit_ibm_runtime`` .

    Uses data in `bool_array`, acquired with measurement bases as ordered in keys of `basis_dict`, to compute observables encoded in values of `basis_dict`.

    Optionally allows averaging over additional axes of `bool_array`, as when twirling.

    Optionally supports measurement twirling, PEC, and postselection.

    Args:
        bool_array: Boolean array, presumably representing data from measured qubits.
            The last two axes are the number of shots and number of classical bits, respectively.
            The least significant bit is assumed to be at index `0` of the bits axis.
            If `meas_basis_axis` is given, that axis of `bool_array` indexes the measurement bases, with length `len(basis_dict)`.
        basis_dict: This dict encodes how the data in `bool_array` should be used to estimate the desired list of Pauli observables.
            The ith key is a measurement basis assumed to correspond to the ith slice of `bool_array` along the `meas_basis_axis` axis.
            Each dict value is a list of length equal to the number of desired observables.
            The jth element of this list is a `SparsePauliOp` assumed to be compatible (qubit-wise commuting) with the measurement-basis key.
            In place of a `SparsePauliOp`, `None` may be used to represent the zero operator, when a basis is not used to compute an observable.
            The jth observable is defined as the sum of the jth element of each dict value (contribution from each meas basis).
            - Note the order of dict entries is relied on here for indexing; the dict keys are never used.
            - Assumes each Pauli term (in dict values) is compatible with each measurement basis (in keys).
            - Assumes each term in each observable appears for exactly one basis (TODO: remove this assumption).
        meas_basis_axis: Axis of bool_array that indexes measurement bases. Ordering must match ordering in `basis_dict`. If `None`,
            then `len(basis_dict)` must be 1, and `bool_array` is assumed to correspond to the only measurement basis.
        avg_axis: Optional axis or axes of bool_array to average over when computing expectation values. Usually this is the "twirling" axis.
            Must be nonnegative. (The shots axis, assumed to be at index -2 in the boolean array, is always averaged over).
        measurement_flips: Optional boolean array used with measurement twirling. Indicates which bits were acquired with measurements preceded by bit-flip gates.
            Data processing will use the result of `xor`ing this array with `bool_array`. Must be same shape as `bool_array`.
        pauli_signs: Optional boolean array used with probabilistic error cancellation (PEC). Final axis is assumed to index all noisy boxes in circuit. Value of `True` indicates an overall sign of `-1` should be associated with the noisy box, typically because an odd number of inverse-noise errors were inserted in that box for the specified circuit randomization. The final axis is immediately collapsed as a sum mod 2 to obtain the overall sign associated with each circuit randomization.
            Remaining shape must be `pauli_signs.shape[:-1] == bool_array.shape[:-2]`. Note this array does not have a shots axis.
        postselect_mask: Optional boolean array used for postselection. `True` (`False`) indicates a shot accepted (rejected) by postselection.
            Shape must be `bool_array.shape[:-1]`.
        gamma_factor: Rescaling factor gamma to be applied to PEC mitigated expectation values. If `None`, rescaling factors will be computed as the
            number of positive samples minus the number of negative samples, computed as `1/(np.sum(~pauli_signs, axis=avg_axis) - np.sum(pauli_signs, axis=avg_axis))`.
            This can fail due to division by zero if there are an equal number of positive and negative samples. Also note this rescales each expectation value
            by a different factor. (TODO: allow specifying an array of gamma values).
        rescale_factors: Scale factor for each Pauli term in each observable in each basis in the given ``basis_dict``.
            Typically used for readout mitigation ("TREX") correction factors.
            Each item in the list corresponds to a different basis, and contains a list of lists of factors for each term in each observable related to that basis.
            The order of the bases and the observables inside each basis should be the same as in `basis_dict`.
            For empty observables for some of the bases, keep an empty list. If `None`, scaling factor will not be applied.

    Returns:
        A list of (exp. val, variance) 2-tuples, one for each desired observable.

        Note: Covariances between summed terms in each observable are not currently accounted for in the
            returned variances. (TODO)

    Raises:
        ValueError if `avg_axis` contains negative values.
        ValueError if `meas_basis_axis` is `None` but `len(basis_dict) != 1`.
        ValueError if the number of entries in `basis_dict` does not equal the length of `bool_array` along `meas_basis_axis`.
    """
    ##### VALIDATE INPUTS:
    if avg_axis is None:
        avg_axis = tuple()
    elif isinstance(avg_axis, int):
        avg_axis = (avg_axis,)
    else:
        avg_axis = tuple(avg_axis)

    if any(a < 0 for a in avg_axis):
        raise ValueError("`avg_axis` must be nonnegative")

    if meas_basis_axis is None:
        if len(basis_dict) != 1:
            raise ValueError(
                f"`meas_basis_axis` cannot be `None` unless there is only one measurement basis, but {len(basis_dict) = }. "
            )
        bool_array = bool_array.reshape((1, *bool_array.shape))
        if measurement_flips is not None:
            measurement_flips = measurement_flips.reshape((1, *measurement_flips.shape))
        if pauli_signs is not None:
            pauli_signs = pauli_signs.reshape((1, *pauli_signs.shape))
        if postselect_mask is not None:
            postselect_mask = postselect_mask.reshape((1, *postselect_mask.shape))
        meas_basis_axis = 0
        avg_axis = tuple(a + 1 for a in avg_axis)

    if len(basis_dict) != bool_array.shape[meas_basis_axis]:
        raise ValueError(
            f"{len(basis_dict) = } does not match {bool_array.shape[meas_basis_axis] = }."
        )

    for i, v in enumerate(basis_dict.values()):
        if i == 0:
            num_observables = len(v)
            continue
        if len(v) != num_observables:
            raise ValueError(
                f"Entry 0 in `basis_dict` indicates {num_observables} observables, but entry {i} indicates {len(v)} observables."
            )

    ##### APPLY MEAS FLIPS:
    if measurement_flips is not None:
        bool_array = np.logical_xor(bool_array, measurement_flips)

    ##### FORMAT OBSERVABLES:
    original_num_bits = bool_array.shape[-1]

    # Convert SparsePauliOps to SparseObservables
    basis_dict_ = {}
    for basis, spo_list in basis_dict.items():
        diag_obs_list = []
        for spo in spo_list:
            if isinstance(spo, SparseObservable):
                diag_obs_list.append(spo)
            elif spo is None:
                diag_obs_list.append(SparseObservable.zero(original_num_bits))
            else:
                diag_obs_list.append(SparseObservable(spo))
        basis_dict_[basis] = diag_obs_list
    basis_dict = basis_dict_

    ##### POSTSELECTION:
    bool_array, basis_dict, num_shots_kept = _apply_postselect_mask(
        bool_array, basis_dict, postselect_mask
    )
    # We will need to correct the shot counts later when computing expectation values.

    ##### PEC SIGNS:
    bool_array, basis_dict, net_signs = _apply_pec_signs(bool_array, basis_dict, pauli_signs)
    # For PEC, we will need to apply a rescaling factor gamma later when computing expectation values.

    ##### ACCUMULATE CONTRIBUTIONS FROM EACH MEAS BASIS:
    barray = BitArray.from_bool_array(bool_array, "little")
    output_shape_each_obs = np.delete(barray.shape, (meas_basis_axis, *avg_axis))
    mean_each_observable = np.zeros((num_observables, *output_shape_each_obs), dtype=float)
    var_each_observable = np.zeros((num_observables, *output_shape_each_obs), dtype=float)

    skip_axes = list([slice(None) for _ in range(meas_basis_axis)])

    for meas_basis_idx, (_, observables) in enumerate(basis_dict.items()):
        # Take element `meas_basis_idx` along axis `meas_basis_axis` of BitArray:
        idx = tuple([*skip_axes, meas_basis_idx])
        barray_this_basis = barray[idx]
        num_kept = num_shots_kept[idx]
        signs = net_signs[idx]
        basis_rescale_factors = (
            rescale_factors[meas_basis_idx] if rescale_factors is not None else None
        )

        ## AVERAGE OVER SHOTS:
        (means, standard_errs) = _bitarray_expectation_value(
            barray_this_basis,
            observables,
            shots=num_kept,
            rescale_each_observable=basis_rescale_factors,
        )

        variances = standard_errs**2
        del standard_errs

        ## AVERAGE OVER SPECIFIED AXES ("TWIRLS"):
        # Update indexing since we already sliced away meas_basis axis:
        avg_axis_ = tuple(a if a < meas_basis_axis else a - 1 for a in avg_axis)

        if gamma_factor is not None:
            rescaling = gamma_factor
        else:
            num_minus = np.count_nonzero(signs, axis=avg_axis_)
            num_plus = np.count_nonzero(~signs, axis=avg_axis_)
            num_twirls = num_plus + num_minus
            rescaling = num_twirls / (num_plus - num_minus)

        # Will weight each twirl by its fraction of kept shots.
        # If no postselection, weighting reduces to dividing by num_twirls:
        weights = num_kept[..., np.newaxis] / np.sum(num_kept, axis=avg_axis_)
        means = rescaling * np.sum(means * weights, axis=avg_axis_)
        # Propagate uncertainties:
        variances = rescaling**2 * np.sum(variances * weights**2, axis=avg_axis_)
        mean_each_observable += means
        var_each_observable += variances

    mean_and_var_each_observable = list(
        zip(mean_each_observable.tolist(), var_each_observable.tolist())
    )

    return mean_and_var_each_observable


def _apply_postselect_mask(
    bool_array: np.ndarray[tuple[int, ...], np.dtype[np.bool]],
    basis_dict: dict[Pauli, list[SparseObservable]],
    postselect_mask: np.ndarray[tuple[int, ...], np.dtype[np.bool]] | None,
):
    """Applies postselection mask in preparation for computing expectation values.

    Args:
        bool_array: Boolean array, presumably representing data from measured qubits.
            The last two axes are the number of shots and number of classical bits, respectively.
        basis_dict: This dict encodes how the data in `bool_array` should be used to estimate the desired list of Pauli observables.
            Similar to `basis_dict` arg of `executor_expectation_values()`, but here Pauli components and coefficients must be represented as
            `SparseObservable` instead of `SparsePauliOp`, and `None` may not be used as a placeholder for the zero operator.
        postselect_mask: Boolean array used for postselection. `True` (`False`) indicates a shot accepted (rejected) by postselection.
            Shape must be `bool_array.shape[:-1]`.

    Returns:
        - A copy of `bool_array` with an extra classical bit appended indicating the postselection mask.
        - A copy of `basis_dict` where each observable term has had an extra `1` appended.
        - An array tabulating how many shots were kept for each circuit configuration. When computing expectation values,
            this must be used to correct for the number of shots included in each average.
    """
    if postselect_mask is not None:
        # Projector will ignore shots where ps bit is 0 when computing expectation
        # (though we will need to correct the shot counts later on)
        basis_dict = {
            # Append a `1` projector to the observable, which will act on the postselection bit:
            basis: [obs.expand("1") for obs in diag_obs_list]
            for basis, diag_obs_list in basis_dict.items()
        }
        # Append ps bit to classical bits:
        bool_array = np.concatenate((bool_array, postselect_mask[..., np.newaxis]), axis=-1)
    else:
        postselect_mask = np.ones(bool_array.shape[:-1], dtype=bool)
        bool_array = bool_array.copy()

    num_shots_kept = np.sum(postselect_mask, axis=-1)

    return bool_array, basis_dict, num_shots_kept


def _apply_pec_signs(
    bool_array: np.ndarray[tuple[int, ...], np.dtype[np.bool]],
    basis_dict: dict[Pauli, list[SparseObservable | SparsePauliOp]],
    pauli_signs: np.ndarray[tuple[int, ...], np.dtype[np.bool]] | None,
):
    """Applies PEC signs in preparation for computing expectation values.

    Args:
        bool_array: Boolean array, presumably representing data from measured qubits.
            The last two axes are the number of shots and number of classical bits, respectively.
        basis_dict: This dict encodes how the data in `bool_array` should be used to estimate the desired list of Pauli observables.
            Similar to `basis_dict` arg of `expectation_values()`, but here `None` may not be used as a placeholder for the zero operator.
        pauli_signs: Optional boolean array used with probabilistic error cancellation (PEC). Indicates which errors were inserted in each
                circuit randomization. Final axis, assumed to index all error generators in circuit, is immediately collapsed as a sum mod 2.
                Remaining shape must be `pauli_signs.shape[:-1] == bool_array.shape[:-2]`. Note this array does not have a shots axis.

    Returns:
        - A copy of `bool_array` with an extra classical bit appended indicating the PEC sign.
        - A copy of `basis_dict` where each observable term has had an extra `Z` appended.
        - An array indicating the net sign of each circuit randomization. When computing expectation values,
            this may be used to compute an approximation of the PEC rescaling factor gamma.
    """
    if pauli_signs is not None:
        # signs axes are [..., error_generator]
        # Append sign bit to classical bits:
        net_signs = np.asarray(np.sum(pauli_signs, axis=-1) % 2, dtype=bool)
        #   Broadcast signs over shots axis:
        net_signs_bc = np.broadcast_to(net_signs[..., np.newaxis], shape=bool_array.shape[:-1])
        bool_array = np.concatenate((bool_array, net_signs_bc[..., np.newaxis]), axis=-1)
        # Pauli Z negates shots where sign bit is 1:
        basis_dict = {
            basis: [obs.expand("Z") for obs in diag_obs_list]
            for basis, diag_obs_list in basis_dict.items()
        }
    else:
        net_signs = np.zeros(bool_array.shape[:-2], dtype=bool)
        bool_array = bool_array.copy()

    return bool_array, basis_dict, net_signs


def _bitarray_expectation_value(
    outcomes: BitArray,
    observables: list[SparseObservable],
    shots: int | np.ndarray[tuple[int, ...], np.dtype[np.int64]] | None = None,
    rescale_each_observable: Sequence[Sequence[float]] | None = None,
):
    """Calculate expectation value of observables on the BitArray data.

    Observables are assumed to be diagonal in the measured bases.

    Args:
        outcomes: BitArray containing the classical data.
        observables: List of `SparseObservable`s to evaluate. These are assumed
            to be diagonal in the measured bases, so X, Y, Z are all treated as Z;
            1, -, l are treated as 1; and 0, +, r are treated as 0.
        shots: If `None` (default), results will be averaged over shots in the data
            as usual. If `shots` is specified, it will be used in the denominator when
            computing the mean instead of the number of shots in the data. This
            permits vectorized processing of postselected data despite the tendency
            of postselection to produce ragged arrays. See `apply_postselect_mask`.
        rescale_each_observable: list of lists of rescale factors for each term in each observable in ``observables``.
            The calculated expectation value of each Pauli term will be multiplied by the matching rescale factor.
            If `None`, rescale factors will not be applied to the calculated expectation values.

    Returns:
        The means and standard errors for the observable expectation values.

        Note: Covariances between summed Paulis are not currently accounted for in the
            returned variances. (TODO)
    """
    num_obs = len(observables)
    obs_lengths = [len(obs) for obs in observables]
    num_bits = observables[0].num_qubits

    # Sum to create flat list of all terms:
    obs_tot = sum(observables, start=SparseObservable.zero(num_bits))
    term_lengths = np.diff(obs_tot.boundaries)
    num_terms = len(obs_tot)

    all_coeffs = obs_tot.coeffs
    all_coeffs = np.array(all_coeffs)
    if not np.allclose(all_coeffs.imag, 0):
        raise ValueError("Nonzero imaginary parts of observable coeffs not supported.")
    all_coeffs = all_coeffs.real

    # We only care whether each (non-id) bit is Z-like, 0-like, or 1-like:
    bit_term_types = np.array(obs_tot.bit_terms) & 0b1100

    mask_z = np.zeros((num_terms, num_bits), dtype=bool)
    mask_0 = mask_z.copy()
    mask_1 = mask_z.copy()
    # fancy indexing:
    term_idx = np.repeat(np.arange(num_terms), np.asarray(term_lengths, dtype=int))
    mask_z[term_idx, obs_tot.indices] = bit_term_types == 0b0000
    mask_0[term_idx, obs_tot.indices] = bit_term_types == 0b1000
    mask_1[term_idx, obs_tot.indices] = bit_term_types == 0b0100

    # Observables have least significant bit at zeroth index in array:
    mask_z = BitArray.from_bool_array(mask_z[:, np.newaxis, :], "little")
    mask_0 = BitArray.from_bool_array(mask_0[:, np.newaxis, :], "little")
    mask_1 = BitArray.from_bool_array(mask_1[:, np.newaxis, :], "little")
    # BitArray shape: terms, shots (1 for broadcasting), bits.

    # append terms axis to outcomes BitArray shape:
    outcomes = outcomes.reshape((*outcomes.shape, 1))

    # Compute parities (0 or 1) of Z components
    parities = (outcomes & mask_z).bitcount() % 2

    # Compute the coefficients of 0 and 1 components.
    nulled_by_0_projector = np.logical_or.reduce((outcomes & mask_0).array, axis=-1)
    nulled_by_1_projector = np.logical_or.reduce(((outcomes & mask_1) ^ mask_1).array, axis=-1)
    coeffs_01 = ~np.logical_or(nulled_by_0_projector, nulled_by_1_projector)

    # Compute expectation values
    shape = np.broadcast_shapes(outcomes.shape, mask_z.shape)
    expvals_each_term = np.zeros(shape, dtype=float)
    sq_expvals_each_term = np.zeros_like(expvals_each_term)

    # Combine masks to get coeff for each shot (-1, 0, or 1)
    if np.all(coeffs_01):
        # We can do a faster computation of pure Pauli parities
        expvals_each_term += outcomes.num_shots - 2 * np.sum(parities, axis=-1, dtype=int)
        sq_expvals_each_term += outcomes.num_shots
    else:
        samples = coeffs_01 * ((-1) ** parities.astype(np.int8))
        expvals_each_term += np.sum(samples, axis=-1)
        sq_expvals_each_term += np.sum(np.abs(samples), axis=-1)

    # Divide by total shots. May be less than nominal number in array if
    # we are postselecting via projector in observable terms:
    denom = np.asarray(outcomes.num_shots if shots is None else shots)

    # Edge case of counts dict containing outcomes but with total shots, eg {"0": 0}.
    no_shots = denom == 0

    expvals_each_term[~no_shots] /= denom[..., np.newaxis]
    sq_expvals_each_term[~no_shots] /= denom[..., np.newaxis]
    expvals_each_term[no_shots] = np.nan
    sq_expvals_each_term[no_shots] = np.nan
    variances_each_term = (
        np.clip(sq_expvals_each_term - expvals_each_term**2, 0, None) / denom[..., np.newaxis]
    )

    # all_coeffs == number of bit terms == means.shape[-1], so broadcasts automatically:
    expvals_each_term *= all_coeffs
    variances_each_term *= all_coeffs**2

    # Divide by rescale factors if supplied
    if rescale_each_observable is not None:
        # combine the rescale factors of all the terms of all observables into a single array
        rescale_each_term = np.array(
            [term for observable in rescale_each_observable for term in observable]
        )

        expvals_each_term *= rescale_each_term
        variances_each_term *= rescale_each_term**2

    ### We have the expectation value and variance for each term.
    ### Next, we sum these back into their original observables:
    expval_each_observable = np.zeros((*expvals_each_term.shape[:-1], num_obs))
    variance_each_observable = np.zeros((*expvals_each_term.shape[:-1], num_obs))
    start = 0
    for obs_idx, obs_len in enumerate(obs_lengths):
        stop = start + obs_len
        expval_each_observable[..., obs_idx] += np.sum(expvals_each_term[..., start:stop], axis=-1)
        variance_each_observable[..., obs_idx] += np.sum(
            variances_each_term[..., start:stop], axis=-1
        )
        start = stop

    stderr_each_observable = np.sqrt(variance_each_observable)

    # Observables are along last axis.
    # Shots, bits axes have been collapsed.

    return expval_each_observable, stderr_each_observable
