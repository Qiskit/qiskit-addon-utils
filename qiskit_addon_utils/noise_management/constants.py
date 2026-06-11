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
"""Constant values."""

DEFAULT_POST_SELECTION_SUFFIX = "_ps"
"""
The default suffix to append to the names of the classical registers used for post selection measurements.
"""

DEFAULT_SPECTATOR_CREG_NAME = "spec"
"""
The default name of the classical register used for measuring spectator qubits.
"""

# Constants for the ``bit_flip_checks`` sub-package. Both pre- and post-circuit bit-flip
# checks feed a single post-selection routine; the "pre"/"post" labels below distinguish
# *where in the circuit* the bit-flip check is inserted (start vs. end), not two different
# selection techniques.

DEFAULT_POST_CHECK_SUFFIX = "_ps"
"""
The default suffix appended to classical registers holding post-circuit bit-flip check measurements.
"""

DEFAULT_PRE_CHECK_SUFFIX = "_pre"
"""
The default suffix appended to classical registers holding pre-circuit bit-flip check measurements.
"""

DEFAULT_SPECTATOR_PRE_CREG_NAME = "spec_pre"
"""
The default name of the classical register used for measuring spectator qubits in pre-circuit bit-flip checks.
"""
