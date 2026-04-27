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
"""Tests for edge-based post/pre-selection passes (with spectator measurements).

Mirrors the scenarios in ``visualize_edge_passes.py``. The circuit under
test acts on qubits 0, 1, 2 of a 5-qubit system; under the coupling map
``4-0-1-2-3`` the spectator qubits are 3 and 4.

Tests assert structural invariants:

* the expected classical registers exist with the right sizes,
* each data qubit's measurements land in the right registers in the right
  order (e.g. ``c_pre``, ``c``, ``c_ps``),
* spectator qubits get their pre/post-selection ops in the right order
  (``measure -> reset`` for pre-selection spectators; trailing ``measure``
  for post-selection spectators),
* the synchronisation barriers are placed where downstream pulse-level
  scheduling expects them.
"""

from __future__ import annotations

import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit_addon_utils.noise_management.post_selection.transpiler.passes import (
    AddPostSelectionMeasures,
    AddPreSelectionMeasures,
    AddSpectatorMeasures,
    AddSpectatorMeasuresPreSelection,
)

# Coupling: 4 -- 0 -- 1 -- 2 -- 3.
# Active qubits (from ``_make_circuit``): {0, 1, 2}.
# Spectator qubits (neighbours of active that are themselves inactive): {3, 4}.
COUPLING_MAP = [(0, 1), (1, 2), (2, 3), (0, 4)]
ACTIVE_QUBITS = [0, 1, 2]
SPECTATOR_QUBITS = [3, 4]


def _make_circuit() -> QuantumCircuit:
    """5-qubit circuit; only qubits 0, 1, 2 carry gates and measurements."""
    qc = QuantumCircuit(5, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc


def _creg_map(circuit: QuantumCircuit) -> dict[str, int]:
    return {creg.name: creg.size for creg in circuit.cregs}


def _ops_for_qubit(circuit: QuantumCircuit, qubit_idx: int) -> list[str]:
    qubit = circuit.qubits[qubit_idx]
    return [inst.operation.name for inst in circuit.data if qubit in inst.qubits]


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


def _measurement_positions(
    circuit: QuantumCircuit, qubit_idx: int, register_name: str
) -> list[int]:
    """Indices in ``circuit.data`` where ``qubit_idx`` measures into ``register_name``."""
    qubit = circuit.qubits[qubit_idx]
    indices: list[int] = []
    for i, inst in enumerate(circuit.data):
        if inst.operation.name != "measure" or qubit not in inst.qubits:
            continue
        clbit = inst.clbits[0]
        for creg in circuit.cregs:
            if clbit in creg and creg.name == register_name:
                indices.append(i)
                break
    return indices


def _barrier_positions(circuit: QuantumCircuit) -> list[int]:
    return [i for i, inst in enumerate(circuit.data) if inst.operation.name == "barrier"]


# ---------------------------------------------------------------------------
# Single-pass / no-spectator baseline
# ---------------------------------------------------------------------------


def test_post_sel_only():
    """Edge variant: post-selection alone leaves spectator wires untouched."""
    pm = PassManager([AddPostSelectionMeasures(x_pulse_type="rx")])
    result = pm.run(_make_circuit())

    assert _creg_map(result) == {"c": 3, "c_ps": 3}
    for q in ACTIVE_QUBITS:
        assert _meas_registers(result, q) == ["c", "c_ps"]
    for q in SPECTATOR_QUBITS:
        assert _ops_for_qubit(result, q) == []


def test_pre_sel_only():
    """Edge variant: pre-selection alone leaves spectator wires untouched."""
    pm = PassManager([AddPreSelectionMeasures(COUPLING_MAP, x_pulse_type="rx")])
    result = pm.run(_make_circuit())

    assert _creg_map(result) == {"c": 3, "c_pre": 3}
    for q in ACTIVE_QUBITS:
        assert _meas_registers(result, q) == ["c_pre", "c"]
    for q in SPECTATOR_QUBITS:
        assert _ops_for_qubit(result, q) == []


# ---------------------------------------------------------------------------
# Spectator passes
# ---------------------------------------------------------------------------


def test_post_sel_with_spectators():
    """Spectator measurements follow data-qubit post-selection."""
    pm = PassManager(
        [
            AddPostSelectionMeasures(x_pulse_type="rx"),
            AddSpectatorMeasures(COUPLING_MAP),
        ]
    )
    result = pm.run(_make_circuit())

    cregs = _creg_map(result)
    assert set(cregs) == {"c", "c_ps", "spec"}
    assert cregs["spec"] == len(SPECTATOR_QUBITS)

    for q in ACTIVE_QUBITS:
        assert _meas_registers(result, q) == ["c", "c_ps"]
    for q in SPECTATOR_QUBITS:
        assert _meas_registers(result, q) == ["spec"]

    # Spectator measurements must come after the last data-qubit post-sel.
    last_data_ps = max(_measurement_positions(result, q, "c_ps")[-1] for q in ACTIVE_QUBITS)
    first_spec = min(_measurement_positions(result, q, "spec")[0] for q in SPECTATOR_QUBITS)
    assert first_spec > last_data_ps


def test_pre_sel_with_spectators():
    """Pre-selection spectators get measure+reset before the main circuit."""
    pm = PassManager(
        [
            AddPreSelectionMeasures(COUPLING_MAP, x_pulse_type="rx"),
            AddSpectatorMeasuresPreSelection(COUPLING_MAP, x_pulse_type="rx"),
        ]
    )
    result = pm.run(_make_circuit())

    cregs = _creg_map(result)
    assert set(cregs) == {"c", "c_pre", "spec_pre"}
    assert cregs["spec_pre"] == len(SPECTATOR_QUBITS)

    for q in ACTIVE_QUBITS:
        assert _meas_registers(result, q) == ["c_pre", "c"]

    # Each spectator wire is exactly: barrier (extended pre-sel barrier),
    # measure (into ``spec_pre``), reset. Per-wire ordering is the load-bearing
    # invariant — global linearisation of independent wires after the barrier
    # is unspecified and not a bug.
    for q in SPECTATOR_QUBITS:
        assert _ops_for_qubit(result, q) == ["barrier", "measure", "reset"]
        assert _meas_registers(result, q) == ["spec_pre"]


def test_spectators_only():
    """Spectator pass on its own still synchronises with a barrier (default)."""
    pm = PassManager([AddSpectatorMeasures(COUPLING_MAP)])
    result = pm.run(_make_circuit())

    cregs = _creg_map(result)
    assert set(cregs) == {"c", "spec"}
    assert cregs["spec"] == len(SPECTATOR_QUBITS)

    for q in ACTIVE_QUBITS:
        assert _meas_registers(result, q) == ["c"]
    for q in SPECTATOR_QUBITS:
        assert _meas_registers(result, q) == ["spec"]

    # A barrier separates main-circuit measurements from the spectator ones.
    first_spec = min(_measurement_positions(result, q, "spec")[0] for q in SPECTATOR_QUBITS)
    barriers_before_spec = [b for b in _barrier_positions(result) if b < first_spec]
    assert barriers_before_spec, "expected a synchronisation barrier before spectator measurements"


def test_spectators_only_no_barrier():
    """``add_barrier=False`` skips the synchronisation barrier."""
    pm = PassManager([AddSpectatorMeasures(COUPLING_MAP, add_barrier=False)])
    result = pm.run(_make_circuit())

    assert set(_creg_map(result)) == {"c", "spec"}
    assert not _barrier_positions(result)
    for q in SPECTATOR_QUBITS:
        assert _meas_registers(result, q) == ["spec"]


# ---------------------------------------------------------------------------
# Full-stack scenarios
# ---------------------------------------------------------------------------


def _assert_full_stack_invariants(result: QuantumCircuit) -> None:
    """Shared invariants for any pass ordering that produces the full stack."""
    cregs = _creg_map(result)
    assert set(cregs) == {"c", "c_pre", "c_ps", "spec_pre", "spec"}
    assert cregs["c"] == 3
    assert cregs["c_pre"] == 3
    assert cregs["c_ps"] == 3
    assert cregs["spec_pre"] == len(SPECTATOR_QUBITS)
    assert cregs["spec"] == len(SPECTATOR_QUBITS)

    for q in ACTIVE_QUBITS:
        assert _meas_registers(result, q) == ["c_pre", "c", "c_ps"]

    # Per-wire spectator ordering is the load-bearing invariant. The earlier
    # bug (spec_pre measure+reset landing AFTER the post-sel barrier on the
    # spec wire) would surface here as ``[barrier, barrier, measure, reset,
    # measure]`` instead of the correct alternation below.
    for q in SPECTATOR_QUBITS:
        assert _ops_for_qubit(result, q) == [
            "barrier",
            "measure",
            "reset",
            "barrier",
            "measure",
        ], f"qubit {q}: {_ops_for_qubit(result, q)}"
        assert _meas_registers(result, q) == ["spec_pre", "spec"]


def test_full_stack_pre_first():
    """Pre+spec_pre then post+spec produces the full stack."""
    pm = PassManager(
        [
            AddPreSelectionMeasures(COUPLING_MAP, x_pulse_type="rx"),
            AddSpectatorMeasuresPreSelection(COUPLING_MAP, x_pulse_type="rx"),
            AddPostSelectionMeasures(x_pulse_type="rx"),
            AddSpectatorMeasures(COUPLING_MAP),
        ]
    )
    _assert_full_stack_invariants(pm.run(_make_circuit()))


def test_full_stack_post_first():
    """Post+spec then pre+spec_pre yields the same full-stack invariants."""
    pm = PassManager(
        [
            AddPostSelectionMeasures(x_pulse_type="rx"),
            AddSpectatorMeasures(COUPLING_MAP),
            AddPreSelectionMeasures(COUPLING_MAP, x_pulse_type="rx"),
            AddSpectatorMeasuresPreSelection(COUPLING_MAP, x_pulse_type="rx"),
        ]
    )
    _assert_full_stack_invariants(pm.run(_make_circuit()))


# ---------------------------------------------------------------------------
# Targeted edge cases (defaults, conflicts, isolated patterns)
# ---------------------------------------------------------------------------


def test_default_xslow_pulse_type():
    """Default ``x_pulse_type`` (xslow) wires up the XSlowGate pulse sequence."""
    pm = PassManager(
        [
            AddPreSelectionMeasures(COUPLING_MAP),
            AddSpectatorMeasuresPreSelection(COUPLING_MAP),
        ]
    )
    result = pm.run(_make_circuit())

    # Pre-selection pulse on the data qubits is now ``xslow`` (one gate per qubit
    # before the X), not 20 ``rx`` rotations.
    op_names = {inst.operation.name for inst in result.data}
    assert "xslow" in op_names


def test_spec_no_spectators_returns_unchanged():
    """Active qubits without spectators in the coupling map: pass returns input."""
    pm = PassManager([AddSpectatorMeasures(coupling_map=[])])
    qc = _make_circuit()
    # Empty coupling map ⇒ no spectator neighbours ⇒ no work to do ⇒ same circuit.
    assert pm.run(qc) == qc


def test_spec_pre_no_spectators_returns_unchanged():
    """Spectator-pre pass on a circuit with no spectators returns the input."""
    pm = PassManager([AddSpectatorMeasuresPreSelection(coupling_map=[])])
    qc = _make_circuit()
    assert pm.run(qc) == qc


def test_spec_pre_existing_register_size_mismatch_raises():
    """Pre-existing ``spec_pre`` register with the wrong size raises."""
    from qiskit.circuit import ClassicalRegister, QuantumRegister
    from qiskit.transpiler.exceptions import TranspilerError

    qreg = QuantumRegister(5, "q")
    creg = ClassicalRegister(3, "c")
    bad_spec_pre = ClassicalRegister(99, "spec_pre")  # wrong size on purpose
    qc = QuantumCircuit(qreg, creg, bad_spec_pre)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure([0, 1, 2], [0, 1, 2])

    pm = PassManager([AddSpectatorMeasuresPreSelection(COUPLING_MAP, x_pulse_type="rx")])
    with pytest.raises(TranspilerError, match="already exists with size"):
        pm.run(qc)


def test_spec_include_unmeasured_toggles_lone_unterminated_qubit():
    """``include_unmeasured`` controls whether a lone unterminated qubit becomes a spec.

    With an empty coupling map there are no spectators-by-adjacency, so the
    only way a qubit can end up in the spectator set is via
    ``include_unmeasured=True`` picking up unterminated active qubits.
    """
    qc = QuantumCircuit(1)
    qc.h(0)  # active, never measured

    on = PassManager([AddSpectatorMeasures(coupling_map=[], include_unmeasured=True)])
    off = PassManager([AddSpectatorMeasures(coupling_map=[], include_unmeasured=False)])

    assert "spec" in {c.name for c in on.run(qc).cregs}
    assert off.run(qc) == qc


def test_spec_pre_include_unmeasured_false():
    """``include_unmeasured=False`` skips the unmeasured-active-qubit broadening."""
    pm = PassManager(
        [
            AddPreSelectionMeasures(COUPLING_MAP, x_pulse_type="rx"),
            AddSpectatorMeasuresPreSelection(
                COUPLING_MAP, x_pulse_type="rx", include_unmeasured=False
            ),
        ]
    )
    result = pm.run(_make_circuit())

    # Adjacency-only spectators are still picked up; ``include_unmeasured=False``
    # only excludes unterminated active qubits, of which we have none here.
    assert _creg_map(result)["spec_pre"] == len(SPECTATOR_QUBITS)
    for q in SPECTATOR_QUBITS:
        assert _meas_registers(result, q) == ["spec_pre"]


def test_spec_pre_existing_register_correct_size():
    """Pre-existing ``spec_pre`` register with the correct size is reused."""
    from qiskit.circuit import ClassicalRegister, QuantumRegister

    qreg = QuantumRegister(5, "q")
    creg = ClassicalRegister(3, "c")
    spec_pre = ClassicalRegister(len(SPECTATOR_QUBITS), "spec_pre")
    qc = QuantumCircuit(qreg, creg, spec_pre)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure([0, 1, 2], [0, 1, 2])

    pm = PassManager(
        [
            AddPreSelectionMeasures(COUPLING_MAP, x_pulse_type="rx"),
            AddSpectatorMeasuresPreSelection(COUPLING_MAP, x_pulse_type="rx"),
        ]
    )
    result = pm.run(qc)
    # No duplicate ``spec_pre`` register should be added.
    spec_pre_names = [c.name for c in result.cregs if c.name == "spec_pre"]
    assert len(spec_pre_names) == 1
    for q in SPECTATOR_QUBITS:
        assert _meas_registers(result, q) == ["spec_pre"]


def test_spec_x_gate_followed_by_normal_measure():
    """X immediately followed by a measure into a *non*-ignored register stays active.

    Exercises the inner-loop completion path where the X look-ahead finds a
    measurement but it isn't into a ``_pre``-suffixed register.
    """
    qc = QuantumCircuit(5, 3)
    qc.x(0)
    qc.measure(0, 0)  # X→measure into normal register, not ``_pre``
    qc.h(1)
    qc.cx(1, 2)
    qc.measure(1, 1)
    qc.measure(2, 2)

    pm = PassManager([AddSpectatorMeasures(COUPLING_MAP)])
    result = pm.run(qc)
    # q0 is treated as fully active (logical X, not pre-sel); spectators are
    # therefore the qubits adjacent to {0, 1, 2}, namely {3, 4}.
    assert "spec" in {c.name for c in result.cregs}
    for q in SPECTATOR_QUBITS:
        assert _meas_registers(result, q) == ["spec"]


def test_spec_post_sel_register_present_without_matching_barrier():
    """Post-sel measurements pre-existing without a barrier: skip the rebuild.

    If the user feeds a circuit where ``c_ps`` measurements exist but no
    barrier acts exactly on those qubits, ``AddSpectatorMeasures`` cannot
    extend a barrier and just appends the spectator measurements.
    """
    from qiskit.circuit import ClassicalRegister, QuantumRegister

    qreg = QuantumRegister(5, "q")
    creg = ClassicalRegister(3, "c")
    creg_ps = ClassicalRegister(3, "c_ps")
    qc = QuantumCircuit(qreg, creg, creg_ps)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure([0, 1, 2], [creg[0], creg[1], creg[2]])
    # Manual ``c_ps`` measurements, no barrier between them and the data ones.
    qc.measure(0, creg_ps[0])
    qc.measure(1, creg_ps[1])
    qc.measure(2, creg_ps[2])

    pm = PassManager([AddSpectatorMeasures(COUPLING_MAP)])
    result = pm.run(qc)
    assert "spec" in {c.name for c in result.cregs}
    for q in SPECTATOR_QUBITS:
        assert _meas_registers(result, q) == ["spec"]


def test_spec_x_gate_pre_selection_pattern():
    """X-gate immediately followed by ``measure(_pre)`` is treated as pre-sel.

    Exercises the look-ahead in ``_find_active_and_terminated_qubits`` that
    distinguishes a pre-selection X+measure(_pre) sequence from a logical X.
    """

    from qiskit.circuit import ClassicalRegister, QuantumRegister

    qreg = QuantumRegister(3, "q")
    creg = ClassicalRegister(2, "c")
    creg_pre = ClassicalRegister(1, "c_pre")
    qc = QuantumCircuit(qreg, creg, creg_pre)
    # Synthetic pre-sel sequence on q0 (no barrier between X and measure)
    # the pass must recognise this pattern and not treat q0 as logically active.
    qc.x(0)
    qc.measure(0, creg_pre[0])
    # Real circuit on q1, q2.
    qc.h(1)
    qc.cx(1, 2)
    qc.measure(1, creg[0])
    qc.measure(2, creg[1])

    pm = PassManager([AddSpectatorMeasures([(0, 1), (1, 2)])])
    result = pm.run(qc)
    # The pass should run without error and add the ``spec`` register.
    # q0, recognised as a pre-selection pattern, becomes a spectator-eligible
    # neighbour of q1 rather than an active qubit.
    assert "spec" in {c.name for c in result.cregs}


def test_full_stack_custom_suffixes():
    """Full stack with custom suffixes and a custom spec_pre register name."""
    pm = PassManager(
        [
            AddPreSelectionMeasures(COUPLING_MAP, x_pulse_type="rx", pre_selection_suffix="_init"),
            AddSpectatorMeasuresPreSelection(
                COUPLING_MAP,
                x_pulse_type="rx",
                spectator_creg_name="spec_init",
                pre_selection_suffix="_init",
            ),
            AddPostSelectionMeasures(
                x_pulse_type="rx",
                post_selection_suffix="_check",
                ignore_creg_suffixes=["_init"],
            ),
            AddSpectatorMeasures(
                COUPLING_MAP,
                ignore_creg_suffixes=["_init"],
                post_selection_suffix="_check",
            ),
        ]
    )
    result = pm.run(_make_circuit())

    cregs = _creg_map(result)
    assert set(cregs) == {"c", "c_init", "spec_init", "c_check", "spec"}

    for q in ACTIVE_QUBITS:
        assert _meas_registers(result, q) == ["c_init", "c", "c_check"]
    for q in SPECTATOR_QUBITS:
        assert _meas_registers(result, q) == ["spec_init", "spec"]
