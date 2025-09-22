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

# Reminder: update the RST file in docs/apidocs when adding new interfaces.
"""Utility methods for circuit slicing.

For more information, check out the `how-to guide <https://qiskit.github.io/qiskit-addon-utils/how_tos/create_circuit_slices.html>`__ which
discusses this submodule.
"""

from .combine_slices import combine_slices
from .slice_by_barriers import slice_by_barriers
from .slice_by_coloring import slice_by_coloring
from .slice_by_depth import slice_by_depth
from .slice_by_gate_types import slice_by_gate_types

__all__ = [
    "combine_slices",
    "slice_by_barriers",
    "slice_by_coloring",
    "slice_by_depth",
    "slice_by_gate_types",
]
