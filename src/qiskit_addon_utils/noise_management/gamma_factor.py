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

"""PEC-like sampling overhead rescaling factor computation."""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import PauliLindbladMap


def gamma_from_noisy_boxes(
    noise_models: dict[str, PauliLindbladMap],
    box_id_to_noise_id: dict[str, str],
    noise_scales_each_box: dict[str, np.ndarray] | None = None,
) -> float:
    """Calculate the gamma factor for a circuit given the Pauli-Lindblad noise models for the boxes in that circuit.

    This function expects the noise models to represent the noise in the circuit, and thus to have positive Lindblad
    rates. The returned gamma is that associated with the inverse noise maps needed to cancel the noise in the circuit.

    Args:
        noise_models: Dict of noise-model IDs (strings) and learned noise models for each unique noisy box in the circuit.
        box_id_to_noise_id: Dict of box IDs and noise-model IDs.
        noise_scales_each_box: Dict of box IDs and factors by which to rescale the Lindblad error rates of each generator in the
            associated noise model.

    Returns:
        The gamma factor.

    Raises:
        ValueError if the length of an array in `noise_scales_each_box` does not equal the length of `rates` of the associated
            `PauliLindbladMap` in `noise_models`.
    """
    gamma = 1.0

    for box_id, noise_id in box_id_to_noise_id.items():
        plm = noise_models[noise_id]

        if noise_scales_each_box is not None:
            scales = noise_scales_each_box[box_id]
            if len(scales) != len(plm.rates):
                raise ValueError(
                    f"Cannot apply noise scales of length {len(scales)} to PauliLindbladMap with {len(plm.rates)} terms."
                )
            plm = PauliLindbladMap.from_components(
                plm.rates * scales, plm.get_qubit_sparse_pauli_list_copy()
            )

        gamma *= plm.inverse().gamma()

    return gamma
