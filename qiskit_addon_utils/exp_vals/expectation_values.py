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

import numpy as np
from qiskit.primitives import BitArray
from qiskit.quantum_info import Pauli, SparseObservable, SparsePauliOp


def expectation_values(
    bool_array: np.ndarray[np._bool],
    basis_dict: dict[Pauli, list[SparsePauliOp | None]],
    meas_basis_axis: int,
    avg_axis: int | tuple[int] | None = None,
    meas_flips: np.ndarray[np._bool] | None = None,
    pec_signs: np.ndarray[np._bool] | None = None,
    postselect_mask: np.ndarray[np._bool] | None = None,
):
    """Computes expectation values from data.

    Uses data in `bool_array`, acquired with measurement bases as ordered in keys of `basis_dict`, to compute observables encoded in values of `basis_dict`.

    Optionally allows averaging over additional axes of `bool_array`, as when twirling.

    Optionally supports measurement twirling, PEC, and postselection.

    Args:
        bool_array: Boolean array, presumably representing data from measured qubits.
            The last two axes are the number of shots and number of classical bits, respectively.
            At least one more axis must be defined indicating the measurement bases, at position `meas_basis_axis` and of length `len(basis_dict)`.
        basis_dict: This dict encodes how the data in `bool_array` should be used to estimate the desired list of Pauli observables.
            The ith key is a measurement basis assumed to correspond to the ith slice of `bool_array` along the `meas_basis_axis` axis.
            Each dict value is a list of length equal to the number of desired observables.
            The jth element of this list is a `SparsePauliOp` assumed to be compatible (qubit-wise commuting) with the measurement-basis key.
            In place of a `SparsePauliOp`, `None` may be used to represent the zero operator, when a basis is not used to compute an observable.
            The jth observable is defined as the sum of the jth element of each dict value (contribution from each meas basis).
            - Note the order of dict entries is relied on here for indexing; the dict keys are never used.
            - Assumes each Pauli term (in dict values) is compatible with each measurement basis (in keys).
            - Assumes each term in each observable appears for exactly one basis (TODO: remove this assumption).
        meas_basis_axis: Axis of bool_array that indexes measurement bases. Ordering must match ordering in `basis_dict`.
        avg_axis: Optional axis or axes of bool_array to average over when computing expectation values. Usually this is the "twirling" axis.
            Must be nonnegative. (The shots axis, assumed to be at index -2 in the boolean array, is always averaged over).
        meas_flips: Optional boolean array used with measurement twirling. Indicates which bits were acquired with measurements preceded by bit-flip gates.
            Data processing will use the result of `xor`ing this array with `bool_array`. Must be same shape as `bool_array`.
        pec_signs: Optional boolean array used with probabilistic error cancellation (PEC). Indicates which errors were inserted in each
            circuit randomization. Final axis, assumed to index all error generators in circuit, is immediately collapsed as a sum mod 2.
            Remaining shape must be `pec_signs.shape[:-1] == bool_array.shape[:-2]`. Note this array does not have a shots axis.
        postselect_mask: Optional boolean array used for postselection. `True` (`False`) indicates a shot accepted (rejected) by postselection.
            Shape must be `bool_array.shape[:-1]`.

    Returns:
        A list of tuples, one for each desired observable. Each tuple is length 2
    """
    if avg_axis is None:
        avg_axis = tuple()
    elif isinstance(avg_axis, int):
        avg_axis = (avg_axis,)
    else:
        avg_axis = tuple(avg_axis)

    if any(a < 0 for a in avg_axis):
        raise ValueError("`avg_axis` must be nonnegative")

    if len(basis_dict) != bool_array.shape[meas_basis_axis]:
        raise ValueError(
            f"{len(basis_dict) =} does not match {bool_array.shape[meas_basis_axis] =}."
        )

    for i, v in enumerate(basis_dict.values()):
        if i == 0:
            num_observables = len(v)
            continue
        if len(v) != num_observables:
            raise ValueError(
                f"Entry 0 in `basis_dict` indicates {num_observables} observables, but entry {i} indicates {len(v)} observables."
            )

    if meas_flips is not None:
        bool_array = np.logical_xor(bool_array, meas_flips)

    original_num_bits = bool_array.shape[-1]

    # Convert SparsePauliOps to SparseObservables,
    # and make diagonal (replace X or Y with Z, assuming
    # correct rotation gates were included in circuit):
    basis_dict_ = {}
    for basis, spo_list in basis_dict.items():
        diag_obs_list = []
        for spo in spo_list:
            if spo is None:
                diag_obs_list.append(SparseObservable.zero(original_num_bits))
            else:
                paulis = spo.paulis.copy()
                paulis.z = paulis.z | paulis.x
                paulis.x = 0
                paulis.phase = spo.paulis.phase.copy()
                spo = SparsePauliOp(paulis, spo.coeffs)
                diag_obs_list.append(SparseObservable.from_sparse_pauli_op(spo))
        basis_dict_[basis] = diag_obs_list
    basis_dict = basis_dict_

    bool_array, basis_dict, num_shots_kept = apply_postselect_mask(
        bool_array, basis_dict, postselect_mask
    )
    # We will need to correct the shot counts later when computing expectation values.

    bool_array, basis_dict, net_signs = apply_pec_signs(bool_array, basis_dict, pec_signs)
    ## We will need to correct the twirl counts later when computing expectation values,
    ## which may be interpreted as treating the negated randomizations as negative counts.
    ## This is approximately equivalent to rescaling by gamma from the noisy circuit.

    barray = BitArray.from_bool_array(bool_array)

    mean_each_observable = np.zeros(num_observables, dtype=float)
    var_each_observable = np.zeros(num_observables, dtype=float)

    skip_axes = list([slice(None) for _ in range(meas_basis_axis)])

    for meas_basis_idx, (_, diagonal_observables) in enumerate(basis_dict.items()):
        # Take element `meas_basis_idx` along axis `meas_basis_axis` of BitArray:
        idx = tuple([*skip_axes, meas_basis_idx])
        barray_this_basis = barray[idx]
        num_kept = num_shots_kept[idx]

        ## AVERAGE OVER SHOTS:
        means = barray_this_basis.expectation_values(diagonal_observables)
        # BitArray.expectation_values normalized by num_shots, but
        # we should only count those kept by postselection:
        means *= barray_this_basis.num_shots / num_kept
        # For each circuit, shots are nominally samples of binomial distribution,
        # so can compute variance from expectation value:
        variances = (1 - means**2) / num_kept

        ## AVERAGE OVER SPECIFIED AXES ("TWIRLS"):
        # Will weight each twirl by its fraction of kept shots
        weights = num_kept / np.sum(num_kept, axis=avg_axis)
        num_minus = np.count_nonzero(net_signs, axis=avg_axis)
        num_plus = np.count_nonzero(~net_signs, axis=avg_axis)
        num_twirls = num_plus + num_minus
        gamma = num_twirls / (num_plus - num_minus)
        means = gamma * np.sum(means * weights, axis=avg_axis)
        # Propagate uncertainties:
        variances = gamma**2 * np.sum(variances * weights**2, axis=avg_axis)

        mean_each_observable += means
        var_each_observable += variances

    mean_and_var_each_observable = list(zip(mean_each_observable.tolist(), var_each_observable.tolist()))

    return mean_and_var_each_observable


def apply_postselect_mask(
    bool_array: np.ndarray[np._bool],
    basis_dict: dict[Pauli, list[SparseObservable]],
    postselect_mask: np.ndarray[np._bool],
):
    """Applies postselection mask in preparation for computing expectation values.

    Args:
        bool_array: Boolean array, presumably representing data from measured qubits.
            The last two axes are the number of shots and number of classical bits, respectively.
        basis_dict: This dict encodes how the data in `bool_array` should be used to estimate the desired list of Pauli observables.
            Similar to `basis_dict` arg of `expectation_values()`, but here Pauli components and coefficients must be represented as
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


def apply_pec_signs(
        bool_array, 
        basis_dict, 
        pec_signs
        ):
    """Applies PEC signs in preparation for computing expectation values.

    Args:
        bool_array: Boolean array, presumably representing data from measured qubits.
            The last two axes are the number of shots and number of classical bits, respectively.
        basis_dict: This dict encodes how the data in `bool_array` should be used to estimate the desired list of Pauli observables.
            Similar to `basis_dict` arg of `expectation_values()`, but here `None` may not be used as a placeholder for the zero operator.
        pec_signs: Optional boolean array used with probabilistic error cancellation (PEC). Indicates which errors were inserted in each
                circuit randomization. Final axis, assumed to index all error generators in circuit, is immediately collapsed as a sum mod 2.
                Remaining shape must be `pec_signs.shape[:-1] == bool_array.shape[:-2]`. Note this array does not have a shots axis.

    Returns:
        - A copy of `bool_array` with an extra classical bit appended indicating the PEC sign.
        - A copy of `basis_dict` where each observable term has had an extra `Z` appended.
        - An array indicating the net sign of each circuit randomization. When computing expectation values,
            this may be used to compute an approximation of the PEC rescaling factor gamma.
    """
    if pec_signs is not None:
        # signs axes are [..., error_generator]
        # Append sign bit to classical bits:
        net_signs = np.asarray(np.sum(pec_signs, axis=-1) % 2, dtype=bool)
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
