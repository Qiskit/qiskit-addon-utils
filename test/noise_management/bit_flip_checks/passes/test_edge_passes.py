# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Tests for edge-based post/pre-check passes (with spectator measurements).

Circuit under test acts on qubits 0, 1, 2 of a 5-qubit system; under coupling
map ``4-0-1-2-3`` the spectator qubits are 3 and 4.
"""

from __future__ import annotations

import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit_addon_utils.noise_management.bit_flip_checks.passes import (
    AddPostCircuitBitFlipChecks,
    AddPreCircuitBitFlipChecks,
    AddSpectatorPostCircuitBitFlipChecks,
    AddSpectatorPreCircuitBitFlipChecks,
)
from qiskit_addon_utils.noise_management.constants import RX_PULSE_COUNT

# Coupling 4-0-1-2-3: active {0,1,2}, spectators (inactive neighbours) {3,4}.
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


@pytest.mark.parametrize(
    "pass_cls, expected_cregs, expected_order",
    [
        (AddPostCircuitBitFlipChecks, {"c": 3, "c_ps": 3}, ["c", "c_ps"]),
        (AddPreCircuitBitFlipChecks, {"c": 3, "c_pre": 3}, ["c_pre", "c"]),
    ],
    ids=["post_sel_only", "pre_sel_only"],
)
def test_single_pass_only_leaves_spectators_untouched(pass_cls, expected_cregs, expected_order):
    """Edge variant: a single check pass alone leaves spectator wires untouched."""
    pm = PassManager([pass_cls(x_pulse_type="rx")])
    result = pm.run(_make_circuit())

    assert _creg_map(result) == expected_cregs
    for q in ACTIVE_QUBITS:
        assert _meas_registers(result, q) == expected_order
    for q in SPECTATOR_QUBITS:
        assert _ops_for_qubit(result, q) == []


def test_post_sel_with_spectators():
    """Spectator parity check is spliced into the data-qubit post-sel sandwich."""
    pm = PassManager(
        [
            AddPostCircuitBitFlipChecks(x_pulse_type="rx"),
            AddSpectatorPostCircuitBitFlipChecks(COUPLING_MAP, x_pulse_type="rx"),
        ]
    )
    result = pm.run(_make_circuit())

    cregs = _creg_map(result)
    assert set(cregs) == {"c", "c_ps", "spec", "spec_ps"}
    assert cregs["spec"] == len(SPECTATOR_QUBITS)
    assert cregs["spec_ps"] == len(SPECTATOR_QUBITS)

    # Pairing under coupling 4-0-1-2-3: spec q3 -> data q2, spec q4 -> data q0;
    # q1 has no spec neighbour so its terminal measure is not deferred.
    paired_data = {0, 2}
    unpaired_data = {1}

    post_pulse_tail = ["barrier"] + ["rx"] * RX_PULSE_COUNT + ["barrier", "measure"]

    for q in ACTIVE_QUBITS:
        assert _meas_registers(result, q) == ["c", "c_ps"]
        ops = _ops_for_qubit(result, q)
        assert ops[-len(post_pulse_tail) :] == post_pulse_tail
        assert ops[-(len(post_pulse_tail) + 1)] == "measure"
        if q in paired_data:
            # Paired data qubits get a bundled barrier before their terminal measure.
            assert ops[-(len(post_pulse_tail) + 2)] == "barrier"
        elif q in unpaired_data:
            # Unpaired data qubits keep the terminal measure in its natural position.
            assert ops[-(len(post_pulse_tail) + 2)] != "barrier"

    for q in SPECTATOR_QUBITS:
        assert _meas_registers(result, q) == ["spec", "spec_ps"]
        ops = _ops_for_qubit(result, q)
        assert ops == ["barrier", "measure", "barrier"] + ["rx"] * RX_PULSE_COUNT + [
            "barrier",
            "measure",
        ]


def test_pre_sel_with_spectators():
    """Pre-check spectators run the same pulses+X+measure check as data qubits."""
    pm = PassManager(
        [
            AddPreCircuitBitFlipChecks(x_pulse_type="rx"),
            AddSpectatorPreCircuitBitFlipChecks(COUPLING_MAP, x_pulse_type="rx"),
        ]
    )
    result = pm.run(_make_circuit())

    cregs = _creg_map(result)
    assert set(cregs) == {"c", "c_pre", "spec_pre"}
    assert cregs["spec_pre"] == len(SPECTATOR_QUBITS)

    for q in ACTIVE_QUBITS:
        assert _meas_registers(result, q) == ["c_pre", "c"]

    for q in SPECTATOR_QUBITS:
        assert _ops_for_qubit(result, q) == ["rx"] * RX_PULSE_COUNT + [
            "x",
            "barrier",
            "measure",
        ]
        assert _meas_registers(result, q) == ["spec_pre"]


def test_spectators_only():
    """Spectator pass on its own builds its own parity-check sandwich.

    Three barriers wrap the measure/pulse/measure so scheduling can't reorder it.
    """
    pm = PassManager([AddSpectatorPostCircuitBitFlipChecks(COUPLING_MAP, x_pulse_type="rx")])
    result = pm.run(_make_circuit())

    cregs = _creg_map(result)
    assert set(cregs) == {"c", "spec", "spec_ps"}
    assert cregs["spec"] == len(SPECTATOR_QUBITS)
    assert cregs["spec_ps"] == len(SPECTATOR_QUBITS)

    for q in ACTIVE_QUBITS:
        assert _meas_registers(result, q) == ["c"]
    for q in SPECTATOR_QUBITS:
        assert _meas_registers(result, q) == ["spec", "spec_ps"]
        ops = _ops_for_qubit(result, q)
        assert ops[0] == "barrier"
        assert ops[1] == "measure"
        assert ops[2] == "barrier"
        assert ops[-1] == "measure"
        assert ops[-2] == "barrier"
        assert all(op == "rx" for op in ops[3:-2])


def _assert_full_stack_invariants(result: QuantumCircuit) -> None:
    """Shared invariants for any pass ordering that produces the full stack."""
    cregs = _creg_map(result)
    assert set(cregs) == {"c", "c_pre", "c_ps", "spec_pre", "spec", "spec_ps"}
    assert cregs["c"] == 3
    assert cregs["c_pre"] == 3
    assert cregs["c_ps"] == 3
    assert cregs["spec_pre"] == len(SPECTATOR_QUBITS)
    assert cregs["spec"] == len(SPECTATOR_QUBITS)
    assert cregs["spec_ps"] == len(SPECTATOR_QUBITS)

    for q in ACTIVE_QUBITS:
        assert _meas_registers(result, q) == ["c_pre", "c", "c_ps"]

    # Load-bearing invariant: the spec wire must run pre-sel BEFORE the post-sel
    # parity check.
    for q in SPECTATOR_QUBITS:
        assert _meas_registers(result, q) == ["spec_pre", "spec", "spec_ps"]
        ops = _ops_for_qubit(result, q)
        expected = (
            ["rx"] * RX_PULSE_COUNT
            + [
                "x",
                "barrier",
                "measure",  # spec_pre
                "barrier",  # pre-initial full-width barrier
                "measure",  # spec
                "barrier",
            ]
            + ["rx"] * RX_PULSE_COUNT
            + [
                "barrier",
                "measure",  # spec_ps
            ]
        )
        assert ops == expected, f"qubit {q}: {ops}"


@pytest.mark.parametrize(
    "pass_order",
    [
        [
            AddPreCircuitBitFlipChecks(x_pulse_type="rx"),
            AddSpectatorPreCircuitBitFlipChecks(COUPLING_MAP, x_pulse_type="rx"),
            AddPostCircuitBitFlipChecks(x_pulse_type="rx"),
            AddSpectatorPostCircuitBitFlipChecks(COUPLING_MAP, x_pulse_type="rx"),
        ],
        [
            AddPostCircuitBitFlipChecks(x_pulse_type="rx"),
            AddSpectatorPostCircuitBitFlipChecks(COUPLING_MAP, x_pulse_type="rx"),
            AddPreCircuitBitFlipChecks(x_pulse_type="rx"),
            AddSpectatorPreCircuitBitFlipChecks(COUPLING_MAP, x_pulse_type="rx"),
        ],
    ],
    ids=["pre_first", "post_first"],
)
def test_full_stack(pass_order):
    """Both pass orderings produce the same full-stack invariants.

    ``post_first`` uniquely covers the post-then-pre dispatch lines.
    """
    _assert_full_stack_invariants(PassManager(pass_order).run(_make_circuit()))


def test_default_xslow_pulse_type():
    """Default ``x_pulse_type`` (xslow) wires up the XSlowGate pulse sequence."""
    pm = PassManager(
        [
            AddPreCircuitBitFlipChecks(),
            AddSpectatorPreCircuitBitFlipChecks(COUPLING_MAP),
        ]
    )
    result = pm.run(_make_circuit())

    op_names = {inst.operation.name for inst in result.data}
    assert "xslow" in op_names


@pytest.mark.parametrize(
    "pass_cls",
    [AddSpectatorPostCircuitBitFlipChecks, AddSpectatorPreCircuitBitFlipChecks],
    ids=["post", "pre"],
)
def test_spec_no_spectators_returns_unchanged(pass_cls):
    """Spectator pass on a circuit with no spectators returns the input."""
    pm = PassManager([pass_cls(coupling_map=[])])
    qc = _make_circuit()
    assert pm.run(qc) == qc


def test_spec_pre_standalone_builds_own_structure():
    """Spectator-pre pass on its own prepends a self-contained pre-check."""
    pm = PassManager([AddSpectatorPreCircuitBitFlipChecks(COUPLING_MAP, x_pulse_type="rx")])
    result = pm.run(_make_circuit())

    cregs = _creg_map(result)
    assert set(cregs) == {"c", "spec_pre"}
    assert cregs["spec_pre"] == len(SPECTATOR_QUBITS)

    for q in ACTIVE_QUBITS:
        assert _meas_registers(result, q) == ["c"]
    for q in SPECTATOR_QUBITS:
        assert _meas_registers(result, q) == ["spec_pre"]
        ops = _ops_for_qubit(result, q)
        assert ops[-1] == "measure"
        assert ops[-2] == "barrier"
        assert ops[-3] == "x"
        assert all(op == "rx" for op in ops[:-3])


def test_spec_pre_existing_register_size_mismatch_raises():
    """Pre-existing ``spec_pre`` register with the wrong size raises."""
    from qiskit.circuit import ClassicalRegister, QuantumRegister

    qreg = QuantumRegister(5, "q")
    creg = ClassicalRegister(3, "c")
    bad_spec_pre = ClassicalRegister(99, "spec_pre")  # wrong size
    qc = QuantumCircuit(qreg, creg, bad_spec_pre)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure([0, 1, 2], [0, 1, 2])

    pm = PassManager([AddSpectatorPreCircuitBitFlipChecks(COUPLING_MAP, x_pulse_type="rx")])
    with pytest.raises(TranspilerError, match="already exists with size"):
        pm.run(qc)


def test_spec_include_unmeasured_toggles_lone_unterminated_qubit():
    """``include_unmeasured`` controls whether a lone unterminated qubit becomes a spec.

    Empty coupling map ⇒ no adjacency spectators, so the only route into the
    spectator set is ``include_unmeasured=True`` picking up unterminated qubits.
    """
    qc = QuantumCircuit(1)
    qc.h(0)  # active, never measured

    on = PassManager(
        [AddSpectatorPostCircuitBitFlipChecks(coupling_map=[], include_unmeasured=True)]
    )
    off = PassManager(
        [AddSpectatorPostCircuitBitFlipChecks(coupling_map=[], include_unmeasured=False)]
    )

    assert "spec" in {c.name for c in on.run(qc).cregs}
    assert off.run(qc) == qc


def test_spec_pre_include_unmeasured_false():
    """``include_unmeasured=False`` skips the unmeasured-active-qubit broadening."""
    pm = PassManager(
        [
            AddPreCircuitBitFlipChecks(x_pulse_type="rx"),
            AddSpectatorPreCircuitBitFlipChecks(
                COUPLING_MAP, x_pulse_type="rx", include_unmeasured=False
            ),
        ]
    )
    result = pm.run(_make_circuit())

    # Adjacency spectators are still picked up; the flag only excludes
    # unterminated active qubits, of which we have none here.
    assert _creg_map(result)["spec_pre"] == len(SPECTATOR_QUBITS)
    for q in SPECTATOR_QUBITS:
        assert _meas_registers(result, q) == ["spec_pre"]


def test_spec_pre_include_unmeasured_true_picks_up_lone_unterminated_qubit():
    """``include_unmeasured=True`` broadens the spectator set to unterminated active qubits.

    Empty coupling map ⇒ no adjacency spectators, so the only route into the
    spectator set is the ``include_unmeasured=True`` broadening.
    """
    qc = QuantumCircuit(1)
    qc.h(0)  # active, never measured

    on = PassManager(
        [AddSpectatorPreCircuitBitFlipChecks(coupling_map=[], include_unmeasured=True)]
    )
    off = PassManager(
        [AddSpectatorPreCircuitBitFlipChecks(coupling_map=[], include_unmeasured=False)]
    )

    assert "spec_pre" in {c.name for c in on.run(qc).cregs}
    assert off.run(qc) == qc


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
            AddPreCircuitBitFlipChecks(x_pulse_type="rx"),
            AddSpectatorPreCircuitBitFlipChecks(COUPLING_MAP, x_pulse_type="rx"),
        ]
    )
    result = pm.run(qc)
    spec_pre_names = [c.name for c in result.cregs if c.name == "spec_pre"]
    assert len(spec_pre_names) == 1
    for q in SPECTATOR_QUBITS:
        assert _meas_registers(result, q) == ["spec_pre"]


def test_spec_x_gate_followed_by_normal_measure():
    """X immediately followed by a measure into a *non*-ignored register stays active.

    Exercises the X look-ahead finding a measurement that isn't into a ``_pre`` register.
    """
    qc = QuantumCircuit(5, 3)
    qc.x(0)
    qc.measure(0, 0)  # X→measure into normal register, not ``_pre``
    qc.h(1)
    qc.cx(1, 2)
    qc.measure(1, 1)
    qc.measure(2, 2)

    pm = PassManager([AddSpectatorPostCircuitBitFlipChecks(COUPLING_MAP)])
    result = pm.run(qc)
    creg_names = {c.name for c in result.cregs}
    assert "spec" in creg_names
    assert "spec_ps" in creg_names
    for q in SPECTATOR_QUBITS:
        assert _meas_registers(result, q) == ["spec", "spec_ps"]


def test_spec_post_sel_register_present_without_matching_barrier():
    """``c_ps`` measurements with no matching barrier: pass falls back to standalone."""
    from qiskit.circuit import ClassicalRegister, QuantumRegister

    qreg = QuantumRegister(5, "q")
    creg = ClassicalRegister(3, "c")
    creg_ps = ClassicalRegister(3, "c_ps")
    qc = QuantumCircuit(qreg, creg, creg_ps)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure([0, 1, 2], [creg[0], creg[1], creg[2]])
    # Manual ``c_ps`` measurements, no barrier before them.
    qc.measure(0, creg_ps[0])
    qc.measure(1, creg_ps[1])
    qc.measure(2, creg_ps[2])

    pm = PassManager([AddSpectatorPostCircuitBitFlipChecks(COUPLING_MAP, x_pulse_type="rx")])
    result = pm.run(qc)
    creg_names = {c.name for c in result.cregs}
    assert "spec" in creg_names
    assert "spec_ps" in creg_names
    for q in SPECTATOR_QUBITS:
        assert _meas_registers(result, q) == ["spec", "spec_ps"]


def test_pre_sel_spectators_do_not_cascade_post_sel_spectators():
    """Pre-sel spectator qubits must not pull their neighbours into post-sel.

    Regression guard: a pre-sel-only qubit's trailing X gate must not make the
    post-sel pass treat it as data and cascade its neighbours into the spec set.
    """
    from qiskit.circuit import ClassicalRegister, QuantumRegister

    # data {0,1,2}, pre-sel spectators {3,4}, outer ring {5,6} adjacent to specs only.
    qreg = QuantumRegister(7, "q")
    creg = ClassicalRegister(3, "c")
    qc = QuantumCircuit(qreg, creg)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure([0, 1, 2], [0, 1, 2])

    # 0-1-2-3 + 0-4 + 3-5 + 4-6
    coupling = [(0, 1), (1, 2), (2, 3), (0, 4), (3, 5), (4, 6)]

    pm = PassManager(
        [
            AddPreCircuitBitFlipChecks(x_pulse_type="rx"),
            AddSpectatorPreCircuitBitFlipChecks(coupling, x_pulse_type="rx"),
            AddPostCircuitBitFlipChecks(x_pulse_type="rx"),
            AddSpectatorPostCircuitBitFlipChecks(coupling, x_pulse_type="rx"),
        ]
    )
    result = pm.run(qc)

    # Two bits (q3, q4) not four — q5, q6 must not cascade in as post-sel specs.
    cregs = _creg_map(result)
    assert cregs["spec_pre"] == 2
    assert cregs["spec"] == 2
    assert cregs["spec_ps"] == 2

    for q in (5, 6):
        assert _ops_for_qubit(result, q) == []
        assert _meas_registers(result, q) == []


def test_full_stack_custom_suffixes():
    """Full stack with custom suffixes and a custom spec_pre register name."""
    pm = PassManager(
        [
            AddPreCircuitBitFlipChecks(x_pulse_type="rx", pre_check_suffix="_init"),
            AddSpectatorPreCircuitBitFlipChecks(
                COUPLING_MAP,
                x_pulse_type="rx",
                spectator_creg_name="spec_init",
                pre_check_suffix="_init",
            ),
            AddPostCircuitBitFlipChecks(
                x_pulse_type="rx",
                post_check_suffix="_check",
                ignore_creg_suffixes=["_init"],
            ),
            AddSpectatorPostCircuitBitFlipChecks(
                COUPLING_MAP,
                x_pulse_type="rx",
                ignore_creg_suffixes=["_init"],
                post_check_suffix="_check",
            ),
        ]
    )
    result = pm.run(_make_circuit())

    cregs = _creg_map(result)
    assert set(cregs) == {"c", "c_init", "spec_init", "c_check", "spec", "spec_check"}

    for q in ACTIVE_QUBITS:
        assert _meas_registers(result, q) == ["c_init", "c", "c_check"]
    for q in SPECTATOR_QUBITS:
        assert _meas_registers(result, q) == ["spec_init", "spec", "spec_check"]


@pytest.mark.parametrize(
    "pass_cls", [AddSpectatorPostCircuitBitFlipChecks, AddSpectatorPreCircuitBitFlipChecks]
)
def test_spectator_pass_rejects_circuit_smaller_than_coupling_map(pass_cls):
    """Coupling map larger than the circuit (e.g. virtual circuit + device map) is an error."""
    qc = QuantumCircuit(3, 3)
    qc.measure(range(3), range(3))
    pm = PassManager([pass_cls([(0, 1), (1, 2), (2, 3), (3, 4)])])  # spans 5 qubits, circuit has 3
    with pytest.raises(TranspilerError, match="coupling map spans"):
        pm.run(qc)
