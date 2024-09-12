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

"""A submodule with transpilation passes for slicing.

.. currentmodule:: qiskit_addon_utils.slicing.transpiler.passes

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   CollectOpColor
   CollectOpSize
   CollectOpType
"""

from .collect_op_color import CollectOpColor
from .collect_op_size import CollectOpSize
from .collect_op_type import CollectOpType

__all__ = [
    "CollectOpColor",
    "CollectOpSize",
    "CollectOpType",
]
