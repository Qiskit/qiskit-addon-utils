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
"""Tests for node-based post/pre-check passes (no spectators)."""

from __future__ import annotations

import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit_addon_utils.noise_management.bit_flip_checks.passes import (
    AddPostCircuitBitFlipChecks,
    AddPreCircuitBitFlipChecks,
)


@pytest.mark.parametrize("pass_cls", [AddPostCircuitBitFlipChecks, AddPreCircuitBitFlipChecks])
def test_empty_circuit_returns_unchanged(pass_cls):
    """Empty circuit is unchanged by the check pass."""
    qc = QuantumCircuit(1)
    pm = PassManager([pass_cls()])
    assert pm.run(qc) == qc


@pytest.mark.parametrize("pass_cls", [AddPostCircuitBitFlipChecks, AddPreCircuitBitFlipChecks])
def test_invalid_x_pulse_type(pass_cls):
    """Unknown ``x_pulse_type`` is rejected by the check pass."""
    with pytest.raises(ValueError):
        pass_cls(x_pulse_type="rz")


def test_pre_check_no_measurements_returns_unchanged():
    """Active qubits but no measurements: pass exits early without modification."""
    qc = QuantumCircuit(1)
    qc.h(0)
    pm = PassManager([AddPreCircuitBitFlipChecks()])
    assert pm.run(qc) == qc


def test_pre_check_only_ignored_registers_returns_unchanged():
    """All measurements go to an ignored register: pass exits early."""
    from qiskit.circuit import ClassicalRegister, QuantumRegister

    qc = QuantumCircuit(QuantumRegister(1, "q"), ClassicalRegister(1, "spec"))
    qc.h(0)
    qc.measure(0, 0)
    # Default ``ignore_creg_names=["spec"]`` filters out the only register.
    pm = PassManager([AddPreCircuitBitFlipChecks()])
    assert pm.run(qc) == qc


def test_unsupported_op_raises():
    """Operations outside the supported set are rejected with a clear error."""
    from qiskit.circuit.library import Initialize
    from qiskit.transpiler.exceptions import TranspilerError

    qc = QuantumCircuit(1, 1)
    qc.append(Initialize("0"), [0])
    qc.measure(0, 0)
    pm = PassManager([AddPostCircuitBitFlipChecks()])
    with pytest.raises(TranspilerError, match="not supported"):
        pm.run(qc)


def test_delay_is_supported():
    """``delay`` instructions are treated as regular (identity) ops and accepted."""
    from qiskit.circuit import Delay

    qc = QuantumCircuit(1, 1)
    qc.append(Delay(100), [0])
    qc.measure(0, 0)
    pm = PassManager([AddPostCircuitBitFlipChecks()])
    pm.run(qc)


def test_post_check_skips_user_reset():
    """A reset before a measurement must not break terminal-measurement detection."""
    from qiskit.circuit import ClassicalRegister, QuantumRegister

    qreg = QuantumRegister(1, "q")
    creg = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qreg, creg)
    qc.h(0)
    qc.reset(0)
    qc.measure(0, creg[0])

    result = PassManager([AddPostCircuitBitFlipChecks()]).run(qc)
    assert {creg.name for creg in result.cregs} == {"c", "c_ps"}
