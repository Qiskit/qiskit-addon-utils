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
"""Control-flow coverage for the post-selection passes.

The :class:`~qiskit.circuit.ControlFlowOp` branches in each pass are exercised
by feeding circuits with both single-block (``box``) and multi-block
(``if_else``) constructs. The boxing case is inspired by test 10 of
``visualize_edge_passes.py``.
"""

from __future__ import annotations

from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.transpiler import PassManager
from qiskit_addon_utils.noise_management.post_selection.transpiler.passes import (
    AddPostSelectionMeasures,
    AddPreSelectionMeasures,
    AddSpectatorMeasures,
    AddSpectatorMeasuresPreSelection,
)

COUPLING_MAP = [(0, 1), (1, 2), (2, 3), (0, 4)]
ACTIVE_QUBITS = [0, 1, 2]
SPECTATOR_QUBITS = [3, 4]


def _box_circuit() -> QuantumCircuit:
    """5q circuit where the entangling gates live inside a single ``box``."""
    qreg = QuantumRegister(5, "q")
    creg = ClassicalRegister(3, "c")
    qc = QuantumCircuit(qreg, creg)
    qc.h(0)
    with qc.box():
        qc.cx(0, 1)
        qc.cx(1, 2)
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc


def _if_else_circuit_consistent() -> QuantumCircuit:
    """5q circuit where both branches measure the same qubit into the same clbit.

    The post-selection pass should treat that qubit as terminated.
    """
    qreg = QuantumRegister(5, "q")
    creg = ClassicalRegister(3, "c")
    qc = QuantumCircuit(qreg, creg)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(0, creg[0])
    with qc.if_test((creg[0], 0)) as else_:
        qc.x(1)
        qc.measure(1, creg[1])
    with else_:
        qc.measure(1, creg[1])
    qc.cx(1, 2)
    qc.measure(2, creg[2])
    return qc


def _if_else_circuit_inconsistent() -> QuantumCircuit:
    """5q circuit where branches measure different qubits.

    Neither qubit is terminated under the multi-block intersection rule.
    """
    qreg = QuantumRegister(5, "q")
    creg = ClassicalRegister(3, "c")
    qc = QuantumCircuit(qreg, creg)
    qc.h(0)
    qc.measure(0, creg[0])
    with qc.if_test((creg[0], 0)) as else_:
        qc.measure(1, creg[1])
    with else_:
        qc.measure(2, creg[1])
    return qc


def _meas_registers(circuit: QuantumCircuit, qubit_idx: int) -> list[str]:
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


def _creg_names(circuit: QuantumCircuit) -> set[str]:
    return {creg.name for creg in circuit.cregs}


# ---------------------------------------------------------------------------
# AddPostSelectionMeasures with control flow
# ---------------------------------------------------------------------------


def test_post_selection_box():
    """Single-block ``box``: every measured qubit is still terminated."""
    pm = PassManager([AddPostSelectionMeasures(x_pulse_type="rx")])
    result = pm.run(_box_circuit())

    assert _creg_names(result) == {"c", "c_ps"}
    for q in ACTIVE_QUBITS:
        assert _meas_registers(result, q) == ["c", "c_ps"]


def test_post_selection_if_else_consistent():
    """Multi-block ``if_else``: only top-level-terminated qubits get post-sel.

    Qubit 1 is measured consistently inside the branches but the trailing
    ``cx(1, 2)`` un-terminates it before the post-selection pass scans the top
    level, so q1 receives no ``c_ps`` measurement. Qubits 0 and 2 are
    terminated at the top and do.
    """
    pm = PassManager([AddPostSelectionMeasures(x_pulse_type="rx")])
    result = pm.run(_if_else_circuit_consistent())

    assert _creg_names(result) == {"c", "c_ps"}
    assert _meas_registers(result, 0) == ["c", "c_ps"]
    assert _meas_registers(result, 1) == []  # cx after if_else clears termination
    assert _meas_registers(result, 2) == ["c", "c_ps"]


def test_post_selection_if_else_inconsistent():
    """Branches measure different qubits: neither is terminated, no post-sel added.

    The branch-only measurements are buried inside the ``if_else`` op, so they
    don't appear at the top level of ``circuit.data`` either.
    """
    pm = PassManager([AddPostSelectionMeasures(x_pulse_type="rx")])
    result = pm.run(_if_else_circuit_inconsistent())

    # Only qubit 0 has an unconditional terminal measurement.
    assert _creg_names(result) == {"c", "c_ps"}
    assert _meas_registers(result, 0) == ["c", "c_ps"]
    # qubits 1 and 2 are measured only inside one branch each ⇒ no top-level
    # measurement and no post-sel measurement.
    assert _meas_registers(result, 1) == []
    assert _meas_registers(result, 2) == []


# ---------------------------------------------------------------------------
# AddPreSelectionMeasures with control flow
# ---------------------------------------------------------------------------


def test_pre_selection_box():
    """``box`` interior gates count as activity for pre-selection."""
    pm = PassManager([AddPreSelectionMeasures(COUPLING_MAP, x_pulse_type="rx")])
    result = pm.run(_box_circuit())

    assert _creg_names(result) == {"c", "c_pre"}
    for q in ACTIVE_QUBITS:
        assert _meas_registers(result, q) == ["c_pre", "c"]


def test_pre_selection_if_else():
    """Pre-selection picks up measurements buried inside ``if_else`` blocks.

    Qubit 1 is only measured inside the if/else branches but the recursive
    ``_find_measurements`` still discovers it, so a ``c_pre`` measurement is
    prepended at the top level.
    """
    pm = PassManager([AddPreSelectionMeasures(COUPLING_MAP, x_pulse_type="rx")])
    result = pm.run(_if_else_circuit_consistent())

    assert _creg_names(result) == {"c", "c_pre"}
    for q in ACTIVE_QUBITS:
        regs = _meas_registers(result, q)
        assert regs[0] == "c_pre", f"q{q} top-level meas regs: {regs}"


# ---------------------------------------------------------------------------
# AddSpectatorMeasures with control flow
# ---------------------------------------------------------------------------


def test_spectators_box():
    """Spectators of qubits gated only inside a ``box`` are still detected."""
    pm = PassManager([AddSpectatorMeasures(COUPLING_MAP)])
    result = pm.run(_box_circuit())

    assert _creg_names(result) == {"c", "spec", "spec_ps"}
    for q in ACTIVE_QUBITS:
        assert _meas_registers(result, q) == ["c"]
    for q in SPECTATOR_QUBITS:
        assert _meas_registers(result, q) == ["spec", "spec_ps"]


def test_spectators_if_else():
    """``include_unmeasured`` makes branch-only-active qubits become spectators."""
    pm = PassManager([AddSpectatorMeasures(COUPLING_MAP)])
    result = pm.run(_if_else_circuit_inconsistent())

    assert "spec" in _creg_names(result)
    # qubit 0 is the only fully-terminated active qubit; it is not a spectator.
    assert _meas_registers(result, 0) == ["c"]


def test_spectator_with_buried_data_neighbour_measure_falls_through():
    """A spec qubit whose only data neighbour has its terminal measure buried
    inside a box can't be paired up. It still gets its parity check, just
    synced via the extended ``barrier1`` instead of a small bundle barrier.
    """
    qreg = QuantumRegister(5, "q")
    creg = ClassicalRegister(3, "c")
    qc = QuantumCircuit(qreg, creg)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure(0, creg[0])
    qc.measure(1, creg[1])
    # q2's terminal measure is inside the box — can't be deferred top-level,
    # so spec q3 (q2's only data neighbour) is unpaired and falls through.
    with qc.box():
        qc.measure(2, creg[2])

    # AddPostSelectionMeasures runs first so AddSpectatorMeasures takes the
    # integrate-with-postsel path (where the pairing logic lives).
    pm = PassManager(
        [
            AddPostSelectionMeasures(x_pulse_type="rx"),
            AddSpectatorMeasures(COUPLING_MAP, x_pulse_type="rx"),
        ]
    )
    result = pm.run(qc)

    # Spec q3 still gets its full parity check (just no leading bundle barrier
    # specific to q3 — it's synced to barrier1 instead).
    q3_meas = _meas_registers(result, 3)
    assert q3_meas == ["spec", "spec_ps"]


# ---------------------------------------------------------------------------
# AddSpectatorMeasuresPreSelection with control flow
# ---------------------------------------------------------------------------


def test_spec_pre_selection_box():
    """Pre-selection spectators wrap correctly around a ``box``-bearing circuit."""
    pm = PassManager(
        [
            AddPreSelectionMeasures(COUPLING_MAP, x_pulse_type="rx"),
            AddSpectatorMeasuresPreSelection(COUPLING_MAP, x_pulse_type="rx"),
        ]
    )
    result = pm.run(_box_circuit())

    assert _creg_names(result) == {"c", "c_pre", "spec_pre"}
    for q in SPECTATOR_QUBITS:
        ops = [inst.operation.name for inst in result.data if result.qubits[q] in inst.qubits]
        assert ops == ["rx"] * 20 + ["x", "barrier", "measure"]


def test_spec_pre_selection_if_else():
    """Pre-selection spectators handle ``if_else``-driven activity correctly."""
    pm = PassManager(
        [
            AddPreSelectionMeasures(COUPLING_MAP, x_pulse_type="rx"),
            AddSpectatorMeasuresPreSelection(COUPLING_MAP, x_pulse_type="rx"),
        ]
    )
    result = pm.run(_if_else_circuit_consistent())

    assert "spec_pre" in _creg_names(result)
    assert "c_pre" in _creg_names(result)
