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

# Reminder: update the RST file in docs/apidocs when adding new interfaces.
"""X-pulse type used by the bit-flip check passes."""

from enum import Enum


class XPulseType(str, Enum):
    """The type of X-pulse to apply for a bit-flip check measurement."""

    XSLOW = "xslow"
    """An ``xslow`` gate."""

    RX = "rx"
    """Twenty ``rx`` gates with angles ``pi/20``."""
