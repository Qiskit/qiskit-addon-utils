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
from qiskit.quantum_info import Pauli, PauliList, SparseObservable, SparsePauliOp


def executor_expectation_values(
    # positional-only arguments: these canNOT be specified as keyword arguments, meaning we can
    # rename them without breaking API
    bool_array: np.ndarray[tuple[int, ...], np.dtype[np.bool_]],
    basis_mapping: dict[Pauli, list[SparsePauliOp | None]]
    | tuple[Sequence[SparsePauliOp], Sequence[str] | PauliList],
    /,
    # positional or keyword arguments
    meas_basis_axis: int | None = None,
    *,
    # keyword-only arguments: these can ONLY be specified as keyword arguments. Renaming them breaks
    # API, but their order does not matter.
    avg_axis: int | tuple[int, ...] | None = None,
    measurement_flips: np.ndarray[tuple[int, ...], np.dtype[np.bool_]] | None = None,
    pauli_signs: np.ndarray[tuple[int, ...], np.dtype[np.bool_]] | None = None,
    postselect_mask: np.ndarray[tuple[int, ...], np.dtype[np.bool_]] | None = None,
    gamma_factor: float | None = None,
    rescale_factors: Sequence[Sequence[Sequence[float]]] | None = None,
):
    """Computes expectation values from shot data from ``qiskit_ibm_runtime.Executor`` and metadata.

    Uses data in ``bool_array``, acquired with measurement bases as ordered in keys of ``basis_dict``, to compute observables encoded in values of ``basis_dict``.

    Optionally allows averaging over additional axes of ``bool_array``, as when twirling.

    Optionally supports measurement twirling, PEC, and postselection.

    Args:
        bool_array: Boolean array, presumably representing data from measured qubits.
            The last two axes are the number of shots and number of classical bits, respectively.
            The least significant bit is assumed to be at index ``0`` of the bits axis.
            If ``meas_basis_axis`` is given, that axis of ``bool_array`` indexes the measurement bases, with length ``len(basis_mapping)``.
        basis_mapping: The Pauli observables and associated bases which were measured. Can be a ``tuple``, ``(observables, msmt_bases)``, or a
            ``dict``, ``{basis: commuting_observables}``.

            - **tuple**: A length-2 tuple containing ``(observables, msmt_bases)``, where ``observables`` is a sequence of ``SparsePauliOp`` instances for
              which individual expectation values should be calculated, and ``msmt_bases`` is a sequence of Pauli observables. The ``i`` th Pauli in
              ``msmt_bases`` is expected to correspond to the ``i`` th slice of ``bool_array`` along the ``meas_basis_axis``.

            - **dict**: The ``i`` th key is a measurement basis assumed to correspond to the ``i`` th slice of ``bool_array`` along the ``meas_basis_axis`` axis.
              The values are lists of observables (``SparsePauliOp``) with length equal to the number of observables. ``None`` values are used
              when an observable does not qubit-wise commute with the basis. This method assumes each observable appears only once in the values, even if it
              commutes with more than one basis.

        meas_basis_axis: Axis of bool_array that indexes measurement bases. Ordering must match ordering in ``basis_mapping``. If ``None``,
            then ``len(basis_mapping)`` must be ``1``, and ``bool_array`` is assumed to correspond to the only measurement basis.
        avg_axis: Optional axis or axes of bool_array to average over when computing expectation values. Usually this is the "twirling" axis.
            Must be nonnegative. (The shots axis, assumed to be at index ``-2`` in the boolean array, is always averaged over).
        measurement_flips: Optional boolean array used with measurement twirling. Indicates which bits were acquired with measurements preceded by bit-flip gates.
            Data processing will use the result of XOR'ing this array with ``bool_array``. Must be same shape as ``bool_array``.
        pauli_signs: Optional boolean array used with probabilistic error cancellation (PEC). Final axis is assumed to index all noisy boxes in circuit. Value of ``True`` indicates an overall sign of ``-1`` should be associated with the noisy box, typically because an odd number of inverse-noise errors were inserted in that box for the specified circuit randomization. The final axis is immediately collapsed as a sum mod 2 to obtain the overall sign associated with each circuit randomization.
            Remaining shape must be ``pauli_signs.shape[:-1] == bool_array.shape[:-2]``. Note this array does not have a shots axis.
        postselect_mask: Optional boolean array used for postselection. ``True`` (``False``) indicates a shot accepted (rejected) by postselection.
            Shape must be ``bool_array.shape[:-1]``.
        gamma_factor: Rescaling factor gamma to be applied to PEC mitigated expectation values. If ``None``, rescaling factors will be computed as the
            number of positive samples minus the number of negative samples, computed as ``1/(np.sum(~pauli_signs, axis=avg_axis) - np.sum(pauli_signs, axis=avg_axis))``.
            This can fail due to division by zero if there are an equal number of positive and negative samples. Also note this rescales each expectation value
            by a different factor. (TODO: allow specifying an array of gamma values).
        rescale_factors: Scale factor for each Pauli term in each observable in each basis in the given ``basis_mapping``.
            Typically used for readout mitigation ("TREX") correction factors.
            Each item in the list corresponds to a different basis, and contains a list of lists of factors for each term in each observable related to that basis.
            The order of the bases and the observables inside each basis should be the same as in ``basis_mapping``.
            For empty observables for some of the bases, keep an empty list. If ``None``, scaling factor will not be applied.

    Returns:
        A list of (exp. val, variance) 2-tuples, one for each desired observable.

        Note: Covariances between summed terms in each observable are not currently accounted for in the
            returned variances. # TODO

    Raises:
        ValueError: ``avg_axis`` contains negative values.
        ValueError: ``meas_basis_axis`` is ``None`` but ``len(basis_mapping) != 1``.
        ValueError: The number of entries in ``basis_mapping`` does not equal the length of ``bool_array`` along ``meas_basis_axis``.
        ValueError: An observable is not covered by the measurement bases.
    """
    ##### VALIDATE INPUTS:
    avg_axis = _validate_avg_axis(avg_axis, len(bool_array.shape))

    if meas_basis_axis is None:
        if (isinstance(basis_mapping, dict) and len(basis_mapping) != 1) or (isinstance(basis_mapping, tuple) and len(basis_mapping[1]) != 1):
            raise ValueError(
                f"`meas_basis_axis` cannot be `None` unless there is only one measurement basis, but {len(basis_mapping) = }. "
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
    elif meas_basis_axis < 0:
        raise ValueError("meas_basis_axis must be nonnegative.")

    basis_dict, num_observables = _validate_and_format_basis_mapping(
        basis_mapping, bool_array, meas_basis_axis
    )

    ##### APPLY MEAS FLIPS:
    if measurement_flips is not None:
        bool_array = np.logical_xor(bool_array, measurement_flips)

    ##### POSTSELECTION:
    if postselect_mask is not None:
        bool_array, basis_dict, num_shots_kept = _apply_postselect_mask(
            bool_array, basis_dict, postselect_mask
        )
    else:
        num_shots_kept = np.full(bool_array.shape[:-2], bool_array.shape[-2])
    # We will need to correct the shot counts later when computing expectation values.

    ##### PEC SIGNS:
    if pauli_signs is not None:
        bool_array, basis_dict, net_signs = _apply_pec_signs(bool_array, basis_dict, pauli_signs)
        # For PEC, we will need to multiply by gamma later when computing expectation values.
        if gamma_factor is None:
            # If gamma not provided, estimate it empirically, for each requested expectation value:
            gamma_factor = 1 / (1 - 2 * np.mean(net_signs, axis=avg_axis))
    elif gamma_factor is None:
        gamma_factor = 1.0

    ##### If other axes are to be averaged over, do so by first absorbing them into the shots axis:
    if avg_axis:
        # move avg_axis just before shots axis (just before axis -2):
        axis_positions_before_shots = -2 - np.arange(len(avg_axis))
        bool_array = np.moveaxis(bool_array, avg_axis, axis_positions_before_shots)
        # flatten into shots axis (preserve sizes of other axes, including bits axis):
        bool_array = np.reshape(
            bool_array, (*bool_array.shape[: -2 - len(avg_axis)], -1, bool_array.shape[-1])
        )

        # update others to match:
        num_shots_kept = np.sum(num_shots_kept, avg_axis)
        meas_basis_axis -= int(np.sum(np.asarray(avg_axis) < meas_basis_axis))

    ##### ACCUMULATE CONTRIBUTIONS FROM EACH MEAS BASIS:
    barray = BitArray.from_bool_array(bool_array, "little")
    output_shape_each_obs = np.delete(barray.shape, meas_basis_axis)
    mean_each_observable = np.zeros((num_observables, *output_shape_each_obs), dtype=float)
    var_each_observable = np.zeros((num_observables, *output_shape_each_obs), dtype=float)

    # Programmatic way of indexing along `meas_basis_axis`:
    skip_axes = list([slice(None) for _ in range(meas_basis_axis)])

    for meas_basis_idx, (_, observables) in enumerate(basis_dict.items()):
        # Take element `meas_basis_idx` along axis `meas_basis_axis` of BitArray:
        idx = tuple([*skip_axes, meas_basis_idx])
        barray_this_basis = barray[idx]
        num_kept = num_shots_kept[idx]
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

        # Append axis for observables being evaluated (to match axis in `means`):
        if not isinstance(gamma_factor, float):
            gamma_factor = gamma_factor[..., np.newaxis]
        means = gamma_factor * means
        # Propagate uncertainties:
        variances = (gamma_factor * standard_errs) ** 2
        # Move observable axis from end to front:
        mean_each_observable += np.moveaxis(means, -1, 0)
        var_each_observable += np.moveaxis(variances, -1, 0)

    # TODO: Return list of tuples of arrays, not list of tuples of list of list of list of ...
    mean_and_var_each_observable = list(
        zip(mean_each_observable.tolist(), var_each_observable.tolist())
    )

    return mean_and_var_each_observable


def _apply_postselect_mask(
    bool_array: np.ndarray[tuple[int, ...], np.dtype[np.bool_]],
    basis_dict: dict[Pauli, list[SparseObservable]],
    postselect_mask: np.ndarray[tuple[int, ...], np.dtype[np.bool_]],
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
    # Projector will ignore shots where ps bit is 0 when computing expectation
    # (though we will need to correct the shot counts later on)
    basis_dict = {
        # Append a `1` projector to the observable, which will act on the postselection bit:
        basis: [obs.expand("1") for obs in diag_obs_list]
        for basis, diag_obs_list in basis_dict.items()
    }
    # Append ps bit to classical bits:
    bool_array = np.concatenate((bool_array, postselect_mask[..., np.newaxis]), axis=-1)
    num_shots_kept = np.sum(postselect_mask, axis=-1)

    return bool_array, basis_dict, num_shots_kept


def _validate_avg_axis(avg_axis: int | tuple[int, ...] | None, num_dims: int) -> tuple[int, ...]:
    if avg_axis is None:
        avg_axis = tuple()
    elif isinstance(avg_axis, int):
        avg_axis = (avg_axis,)
    else:
        avg_axis = tuple(avg_axis)
    if any(a > (num_dims - 3) for a in avg_axis):
        raise ValueError(
            "Cannot average over the last two dimensions of `bool_array`, which are associated with shots and qubits."
        )
    if any(a < 0 for a in avg_axis):
        raise ValueError("`avg_axis` must be nonnegative")

    return avg_axis


def _apply_pec_signs(
    bool_array: np.ndarray[tuple[int, ...], np.dtype[np.bool_]],
    basis_dict: dict[Pauli, list[SparseObservable | SparsePauliOp]],
    pauli_signs: np.ndarray[tuple[int, ...], np.dtype[np.bool_]],
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

    return bool_array, basis_dict, net_signs


def _convert_to_pauli(basis):
    """Converts a basis in various formats into a Pauli object.

    Can convert a string or a list of integers representing the Paulis using this convention:
    0=I, 1=Z, 2=X, 3=Y

    Args:
        basis: the basis to convert.

    Returns:
        The Pauli represented as a Pauli object.

    Raises:
        ValueError: if the basis is in invalid format.
    """
    int_mapping = {0: "I", 1: "Z", 2: "X", 3: "Y"}
    if isinstance(basis, Pauli):
        return basis
    if isinstance(basis, str):
        return Pauli(basis)
    if isinstance(basis, (list, np.ndarray, tuple)) and isinstance(
        basis[0], (np.unsignedinteger, int, np.integer)
    ):
        return Pauli("".join([int_mapping[int_val] for int_val in basis]))

    raise ValueError("basis must be a Pauli instance, str or a list of ints.")


def _find_measure_basis_to_observable_mapping(
    observables: Sequence[SparsePauliOp], measure_bases: Sequence[str] | Sequence[int] | PauliList
) -> dict[Pauli, list[SparsePauliOp | None]]:
    """Maps each term for each observable to the first basis it qubit-wise commutes with from the given measure_bases.

    Args:
        observables: list of observables.
        measure_bases: list of Pauli bases that the observables are measured with.

    Returns:
        A dictionary mapping from basis to observables terms that commutes with them.

    Raises:
        ValueError: If there is an observable with a term that does not qubit-wise commute with any basis from the given measure_bases.
    """
    measure_paulis = PauliList([_convert_to_pauli(basis) for basis in measure_bases])
    measurement_dict: dict[Pauli, list[SparsePauliOp]] = {}
    observables_elements_basis_found = []
    for basis in measure_paulis:
        measurement_dict[basis] = [[] for _ in range(len(observables))]

    for observable_index, observable in enumerate(observables):
        observables_elements_basis_found.append(np.zeros((len(observable)), dtype=np.bool_))
        for basis in measure_paulis:
            basis_paulis = []
            basis_coeffs = []
            # find the elements that commutes with this basis
            for element_index, (observable_element, observable_coeff) in enumerate(
                zip(observable.paulis, observable.coeffs)
            ):
                # use only the first commuting basis found for each observable element
                # TODO: enable multiple bases for each element, lowering variance in the expectation value calculation
                if observables_elements_basis_found[observable_index][element_index]:
                    continue
                commutes = (
                    np.dot(observable_element.z, basis.x) + np.dot(observable_element.x, basis.z)
                ) % 2 == 0
                if commutes:
                    basis_paulis.append(observable_element)
                    basis_coeffs.append(observable_coeff)
                    observables_elements_basis_found[observable_index][element_index] = True
            measurement_dict[basis][observable_index] = (
                SparsePauliOp(basis_paulis, basis_coeffs) if basis_paulis else None
            )
    if any(
        False in observable_elements_list
        for observable_elements_list in observables_elements_basis_found
    ):
        # print the problematic observable elements
        for observable_index, observable in enumerate(observables):
            for element_index, observable_element in enumerate(observable.paulis):
                if not observables_elements_basis_found[observable_index][element_index]:
                    print(
                        f"Observable element {observable_element} of observable {observable} do not commute with any measurement basis."
                    )
        raise ValueError("Some observable elements do not commute with any measurement basis.")
    return measurement_dict


def _validate_and_format_basis_mapping(basis_mapping, bool_array, meas_basis_axis):
    """Validate input and format bases to observables mapping.

    Args:
        basis_mapping: The Pauli observables and associated bases which were measured. Can be a ``tuple``, ``(observables, msmt_bases)``, or a
            ``dict``, ``{basis: commuting_observables}``.
            In case of a dict, the bases are already mapped to the observables and only validation is needed.
        bool_array: The data from measured qubits.
        meas_basis_axis: Axis of bool_array that indexes measurement bases.

    Returns:
        The observables mapped to bases in a dict format ``{basis: commuting_observables}`` and the number of observables.
    """
    if isinstance(basis_mapping, dict):
        if len(basis_mapping) != bool_array.shape[meas_basis_axis]:
            raise ValueError(
                f"{len(basis_mapping) = } does not match {bool_array.shape[meas_basis_axis] = }."
            )
    elif isinstance(basis_mapping, tuple):
        if len(basis_mapping) != 2:
            raise ValueError(
                "if basis_mapping is a tuple, it must contain observables element and measurement_bases element."
            )
        if len(basis_mapping[1]) != bool_array.shape[meas_basis_axis]:
            raise ValueError(
                f"{len(basis_mapping[1]) = } does not match {bool_array.shape[meas_basis_axis] = }."
            )
        try:
            basis_mapping = _find_measure_basis_to_observable_mapping(
                basis_mapping[0], basis_mapping[1]
            )
        except ValueError as err:
            raise ValueError(
                "The observables and measurement bases in `basis_mapping` do not match. "
                "Please check the values of `basis_mapping` and try again."
            ) from err
    else:
        raise ValueError("basis_mapping must be either a dict or a tuple")

    # Validate number of observable match for each basis
    for i, v in enumerate(basis_mapping.values()):
        if i == 0:
            num_observables = len(v)
            continue
        if len(v) != num_observables:
            raise ValueError(
                f"Entry 0 in `basis_mapping` indicates {num_observables} observables, but entry {i} indicates {len(v)} observables."
            )

    ##### FORMAT OBSERVABLES:
    original_num_bits = bool_array.shape[-1]

    # Convert SparsePauliOps to SparseObservables
    basis_mapping_ = {}
    for basis, spo_list in basis_mapping.items():
        diag_obs_list = []
        for spo in spo_list:
            if isinstance(spo, SparseObservable):
                diag_obs_list.append(spo)
            elif spo is None:
                diag_obs_list.append(SparseObservable.zero(original_num_bits))
            else:
                diag_obs_list.append(SparseObservable(spo))
        basis_mapping_[basis] = diag_obs_list

    return basis_mapping_, num_observables


def _bitarray_expectation_value(
    outcomes: BitArray,
    observables: list[SparseObservable],
    shots: np.ndarray[tuple[int, ...], np.dtype[np.int64]] | None = None,
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
    nulled_by_projector = np.any((outcomes & mask_0).array, axis=-1)
    nulled_by_projector |= np.any((~outcomes & mask_1).array, axis=-1)
    coeffs_01 = ~nulled_by_projector

    # Compute expectation values
    shape = np.broadcast_shapes(outcomes.shape, mask_z.shape)
    expvals_each_term = np.zeros(shape, dtype=float)
    sq_expvals_each_term = np.zeros_like(expvals_each_term)

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
    shots = outcomes.num_shots if shots is None else shots[..., np.newaxis]

    # Edge case of counts dict containing outcomes but with total shots, eg {"0": 0}.
    with np.errstate(divide="ignore"):
        expvals_each_term /= shots
        sq_expvals_each_term /= shots
        variances_each_term = np.clip(sq_expvals_each_term - expvals_each_term**2, 0, None) / shots
    expvals_each_term[~np.isfinite(expvals_each_term)] = np.nan
    variances_each_term[~np.isfinite(variances_each_term)] = np.nan

    # len(all_coeffs) == number of bit terms == expvals_each_term.shape[-1], so broadcasts:
    # TODO: test case of empty observable
    expvals_each_term *= all_coeffs
    variances_each_term *= all_coeffs**2

    # Divide by rescale factors if supplied
    if rescale_each_observable is not None:
        # flatten rescale factors of all the terms of all observables into 1D array
        # remove factors of empty observables to keep the shape as the expvals
        rescale_each_term = np.reshape(np.array(rescale_each_observable)[np.nonzero(obs_lengths)] , -1)
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
