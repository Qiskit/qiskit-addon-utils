def apply_postselect_mask(
        basis_dict: dict[Pauli, list[SparseObservable]],
        bool_array: np.ndarray[np._bool],
        postselect_mask: np.ndarray[np._bool],
        ):
    """
    Args:
        basis_dict: This dict encodes how the data in `bool_array` should be used to estimate the desired list of Pauli observables. 
            Similar to `basis_dict` arg of `expectation_values()`, but here Pauli components and coefficients must be represented as
            `SparseObservable` instead of `SparsePauliOp`, and `None` may not be used as a placeholder for the zero operator.
        bool_array: Boolean array, presumably representing data from measured qubits. 
            The last two axes are the number of shots and number of classical bits, respectively.
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
        basis_dict = {basis:
                    [obs.expand('1') for obs in diag_obs_list]
                    for basis, diag_obs_list in basis_dict.items()}
        # Append ps bit to classical bits:
        bool_array = np.concatenate((bool_array, postselect_mask[..., np.newaxis]), axis=-1)
    else:
        postselect_mask = np.ones(bool_array.shape[:-1], dtype=bool)
        bool_array = bool_array.copy()
    
    num_shots_kept = np.sum(postselect_mask, axis=-1)

    return bool_array, basis_dict, num_shots_kept


def apply_pec_signs(bool_array, basis_dict, pec_signs):
    """
    Args:
        basis_dict: This dict encodes how the data in `bool_array` should be used to estimate the desired list of Pauli observables. 
            Similar to `basis_dict` arg of `expectation_values()`, but here `None` may not be used as a placeholder for the zero operator.
        bool_array: Boolean array, presumably representing data from measured qubits. 
            The last two axes are the number of shots and number of classical bits, respectively.
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
        basis_dict = {basis:
                    [obs.expand('Z') for obs in diag_obs_list]
                    for basis, diag_obs_list in basis_dict.items()
                    }
    else:
        net_signs = np.zeros(bool_array.shape[:-2], dtype=bool)
        bool_array = bool_array.copy()

    return bool_array, basis_dict, net_signs