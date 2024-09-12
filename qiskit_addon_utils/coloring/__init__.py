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

"""Utility methods for coloring.

.. currentmodule:: qiskit_addon_utils.coloring

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   auto_color_edges
   is_valid_edge_coloring
"""

from .automatic_coloring import auto_color_edges
from .validation import is_valid_edge_coloring

__all__ = [
    "auto_color_edges",
    "is_valid_edge_coloring",
]
