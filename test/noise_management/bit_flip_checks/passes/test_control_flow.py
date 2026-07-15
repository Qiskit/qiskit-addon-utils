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
"""Control-flow coverage: exercise each pass's ControlFlowOp branches with
single-block (``box``) and multi-block (``if_else``) constructs.
"""

from __future__ import annotations

from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.transpiler import PassManager
from qiskit_addon_utils.noise_management.bit_flip_checks.passes import (
    AddPostCircuitBitFlipChecks,
    AddPreCircuitBitFlipChecks,
    AddSpectatorPostCircuitBitFlipChecks,
    AddSpectatorPreCircuitBitFlipChecks,
)

COUPLING_MAP = [(0, 1), (1, 2), (2, 3), (0, 4)]


def _if_else_circuit_consistent() -> QuantumCircuit:
    """Both branches measure the same qubit into the same clbit ⇒ qubit terminated."""
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
    """Branches measure different qubits ⇒ neither is terminated (intersection rule)."""
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


def test_post_check_if_else_consistent():
    """Multi-block ``if_else``: only top-level-terminated qubits get post-sel.

    The trailing ``cx(1, 2)`` un-terminates q1, so it gets no ``c_ps``; q0 and q2 do.
    """
    pm = PassManager([AddPostCircuitBitFlipChecks(x_pulse_type="rx")])
    result = pm.run(_if_else_circuit_consistent())

    assert _creg_names(result) == {"c", "c_ps"}
    assert _meas_registers(result, 0) == ["c", "c_ps"]
    assert _meas_registers(result, 1) == []  # cx after if_else clears termination
    assert _meas_registers(result, 2) == ["c", "c_ps"]


def test_post_check_if_else_inconsistent():
    """Branches measure different qubits: neither is terminated, no post-sel added."""
    pm = PassManager([AddPostCircuitBitFlipChecks(x_pulse_type="rx")])
    result = pm.run(_if_else_circuit_inconsistent())

    # Only q0 has an unconditional terminal measurement; q1/q2 measure inside one
    # branch each, so no top-level and no post-sel measurement.
    assert _creg_names(result) == {"c", "c_ps"}
    assert _meas_registers(result, 0) == ["c", "c_ps"]
    assert _meas_registers(result, 1) == []
    assert _meas_registers(result, 2) == []


def test_spectator_with_buried_data_neighbour_measure_falls_through():
    """A spec qubit whose only data neighbour's terminal measure is buried in a box
    can't be paired up, but still gets its parity check (synced via ``barrier1``).
    """
    qreg = QuantumRegister(5, "q")
    creg = ClassicalRegister(3, "c")
    qc = QuantumCircuit(qreg, creg)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure(0, creg[0])
    qc.measure(1, creg[1])
    # q2's terminal measure inside the box can't be deferred, so spec q3 (q2's only
    # data neighbour) is unpaired.
    with qc.box():
        qc.measure(2, creg[2])

    # Post pass first so the spectator pass takes the integrate-with-postsel path.
    pm = PassManager(
        [
            AddPostCircuitBitFlipChecks(x_pulse_type="rx"),
            AddSpectatorPostCircuitBitFlipChecks(COUPLING_MAP, x_pulse_type="rx"),
        ]
    )
    result = pm.run(qc)

    q3_meas = _meas_registers(result, 3)
    assert q3_meas == ["spec", "spec_ps"]


def test_spec_pre_check_if_else():
    """Pre-check spectators handle ``if_else``-driven activity correctly."""
    pm = PassManager(
        [
            AddPreCircuitBitFlipChecks(x_pulse_type="rx"),
            AddSpectatorPreCircuitBitFlipChecks(COUPLING_MAP, x_pulse_type="rx"),
        ]
    )
    result = pm.run(_if_else_circuit_consistent())

    assert "spec_pre" in _creg_names(result)
    assert "c_pre" in _creg_names(result)
