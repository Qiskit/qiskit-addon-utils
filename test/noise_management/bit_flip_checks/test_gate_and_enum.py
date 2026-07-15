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
"""Tests for the public ``XSlowGate`` and ``XPulseType`` exports."""

from __future__ import annotations

import pytest
from qiskit_addon_utils.noise_management.bit_flip_checks import XSlowGate
from qiskit_addon_utils.noise_management.bit_flip_checks.passes import XPulseType


def test_xslow_gate_defaults():
    """A default ``XSlowGate`` is a single-qubit, parameterless gate named ``"xslow"``."""
    gate = XSlowGate()
    assert gate.name == "xslow"
    assert gate.label == "xslow"
    assert gate.num_qubits == 1
    assert gate.params == []


def test_xslow_gate_custom_name():
    """The gate name and label are configurable."""
    gate = XSlowGate(label="slow", xslow_gate_name="xslow_2")
    assert gate.name == "xslow_2"
    assert gate.label == "slow"


def test_xpulse_type_values():
    """``XPulseType`` is a string enum with the expected members."""
    assert XPulseType.XSLOW == "xslow"
    assert XPulseType.RX == "rx"
    assert {member.value for member in XPulseType} == {"xslow", "rx"}


def test_xpulse_type_from_string():
    """A raw string round-trips to the matching enum member; unknown values raise."""
    assert XPulseType("xslow") is XPulseType.XSLOW
    assert XPulseType("rx") is XPulseType.RX
    with pytest.raises(ValueError):
        XPulseType("rz")
