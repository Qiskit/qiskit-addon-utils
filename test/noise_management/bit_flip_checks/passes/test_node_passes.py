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
"""Tests for node-based post/pre-check passes (no spectators).

Mirrors the scenarios in ``visualize_passes.py``. Instead of comparing
rendered circuit diagrams these tests assert structural invariants of the
resulting circuit:

* the expected classical registers are present with the right sizes,
* each data qubit's measurements land in the right registers in the right
  order (e.g. ``c_pre`` first, ``c`` in the middle, ``c_ps`` last).
"""

from __future__ import annotations

import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit_addon_utils.noise_management.bit_flip_checks.transpiler.passes import (
    AddPostCircuitBitFlipChecks,
    AddPreCircuitBitFlipChecks,
)


def _make_circuit() -> QuantumCircuit:
    """3-qubit chain with terminal measurements on every qubit."""
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc


def _creg_map(circuit: QuantumCircuit) -> dict[str, int]:
    return {creg.name: creg.size for creg in circuit.cregs}


def _meas_registers(circuit: QuantumCircuit, qubit_idx: int) -> list[str]:
    """Names of classical registers the qubit measures into, in order."""
    qubit = circuit.qubits[qubit_idx]
    out: list[str] = []
    for inst in circuit.data:
        if inst.operation.name == "measure" and qubit in inst.qubits:
            clbit = inst.clbits[0]
            for creg in circuit.cregs:
                if clbit in creg:
                    out.append(creg.name)
                    break
    return out


# ---------------------------------------------------------------------------
# Single-pass scenarios
# ---------------------------------------------------------------------------


def test_post_sel_default():
    """Post-check alone with default suffix: adds ``c_ps``."""
    pm = PassManager([AddPostCircuitBitFlipChecks(x_pulse_type="rx")])
    result = pm.run(_make_circuit())

    assert _creg_map(result) == {"c": 3, "c_ps": 3}
    for q in range(3):
        assert _meas_registers(result, q) == ["c", "c_ps"]


def test_post_sel_custom_suffix():
    """Post-check with custom suffix: adds ``c_check``, never ``c_ps``."""
    pm = PassManager([AddPostCircuitBitFlipChecks(x_pulse_type="rx", post_check_suffix="_check")])
    result = pm.run(_make_circuit())

    assert _creg_map(result) == {"c": 3, "c_check": 3}
    for q in range(3):
        assert _meas_registers(result, q) == ["c", "c_check"]


def test_pre_sel_default():
    """Pre-check alone with default suffix: adds ``c_pre``."""
    pm = PassManager([AddPreCircuitBitFlipChecks(x_pulse_type="rx")])
    result = pm.run(_make_circuit())

    assert _creg_map(result) == {"c": 3, "c_pre": 3}
    for q in range(3):
        assert _meas_registers(result, q) == ["c_pre", "c"]


def test_pre_sel_custom_suffix():
    """Pre-check with custom suffix: adds ``c_init``, never ``c_pre``."""
    pm = PassManager([AddPreCircuitBitFlipChecks(x_pulse_type="rx", pre_check_suffix="_init")])
    result = pm.run(_make_circuit())

    assert _creg_map(result) == {"c": 3, "c_init": 3}
    for q in range(3):
        assert _meas_registers(result, q) == ["c_init", "c"]


# ---------------------------------------------------------------------------
# Combined-pass scenarios
# ---------------------------------------------------------------------------


def test_pre_then_post_default():
    """Pre then post (default suffixes). No cross-contamination of registers."""
    pm = PassManager(
        [
            AddPreCircuitBitFlipChecks(x_pulse_type="rx"),
            AddPostCircuitBitFlipChecks(x_pulse_type="rx"),
        ]
    )
    result = pm.run(_make_circuit())

    cregs = _creg_map(result)
    assert set(cregs) == {"c", "c_pre", "c_ps"}
    assert "c_pre_ps" not in cregs  # pre-check register must not be post-selected
    for q in range(3):
        assert _meas_registers(result, q) == ["c_pre", "c", "c_ps"]


def test_post_then_pre_default():
    """Post then pre (default suffixes). Final structure matches ``pre_then_post``."""
    pm = PassManager(
        [
            AddPostCircuitBitFlipChecks(x_pulse_type="rx"),
            AddPreCircuitBitFlipChecks(x_pulse_type="rx"),
        ]
    )
    result = pm.run(_make_circuit())

    cregs = _creg_map(result)
    assert set(cregs) == {"c", "c_pre", "c_ps"}
    assert "c_ps_pre" not in cregs  # post-check register must not be pre-selected
    for q in range(3):
        assert _meas_registers(result, q) == ["c_pre", "c", "c_ps"]


def test_pre_then_post_custom():
    """Pre then post with custom suffixes; pre must be ignored by post."""
    pm = PassManager(
        [
            AddPreCircuitBitFlipChecks(x_pulse_type="rx", pre_check_suffix="_init"),
            AddPostCircuitBitFlipChecks(
                x_pulse_type="rx",
                post_check_suffix="_check",
                ignore_creg_suffixes=["_init"],
            ),
        ]
    )
    result = pm.run(_make_circuit())

    cregs = _creg_map(result)
    assert set(cregs) == {"c", "c_init", "c_check"}
    assert "c_init_check" not in cregs
    for q in range(3):
        assert _meas_registers(result, q) == ["c_init", "c", "c_check"]


def test_post_then_pre_custom():
    """Post then pre with custom suffixes; post must be ignored by pre."""
    pm = PassManager(
        [
            AddPostCircuitBitFlipChecks(x_pulse_type="rx", post_check_suffix="_check"),
            AddPreCircuitBitFlipChecks(
                x_pulse_type="rx",
                pre_check_suffix="_init",
                ignore_creg_suffixes=["_check"],
            ),
        ]
    )
    result = pm.run(_make_circuit())

    cregs = _creg_map(result)
    assert set(cregs) == {"c", "c_init", "c_check"}
    assert "c_check_init" not in cregs
    for q in range(3):
        assert _meas_registers(result, q) == ["c_init", "c", "c_check"]


# ---------------------------------------------------------------------------
# Trivial / degenerate inputs
# ---------------------------------------------------------------------------


def test_empty_circuit_post_check():
    """Empty circuit is unchanged by ``AddPostCircuitBitFlipChecks``."""
    qc = QuantumCircuit(1)
    pm = PassManager([AddPostCircuitBitFlipChecks()])
    assert pm.run(qc) == qc


def test_empty_circuit_pre_check():
    """Empty circuit is unchanged by ``AddPreCircuitBitFlipChecks``."""
    qc = QuantumCircuit(1)
    pm = PassManager([AddPreCircuitBitFlipChecks()])
    assert pm.run(qc) == qc


def test_invalid_x_pulse_type_post():
    """Unknown ``x_pulse_type`` is rejected by post-check pass."""
    with pytest.raises(ValueError):
        AddPostCircuitBitFlipChecks(x_pulse_type="rz")


def test_invalid_x_pulse_type_pre():
    """Unknown ``x_pulse_type`` is rejected by pre-check pass."""
    with pytest.raises(ValueError):
        AddPreCircuitBitFlipChecks(x_pulse_type="rz")


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
    # Should run without raising.
    pm.run(qc)


def test_post_check_skips_user_reset():
    """User-provided ``reset`` operations are skipped by terminal-measurement detection.

    A reset followed by a measurement should still leave the qubit terminated;
    the reset itself is not a measurement and shouldn't break the analysis.
    """
    from qiskit.circuit import ClassicalRegister, QuantumRegister

    qreg = QuantumRegister(1, "q")
    creg = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qreg, creg)
    qc.h(0)
    qc.reset(0)
    qc.measure(0, creg[0])

    result = PassManager([AddPostCircuitBitFlipChecks()]).run(qc)
    # ``c_ps`` register added; qubit is still recognised as terminated despite the reset.
    assert {creg.name for creg in result.cregs} == {"c", "c_ps"}
