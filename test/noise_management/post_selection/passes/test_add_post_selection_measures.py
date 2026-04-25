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
"""Test the `AddPostSelectionMeasures` pass."""

import numpy as np
import pytest
from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RXGate
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passmanager import PassManager
from qiskit_addon_utils.noise_management.post_selection.transpiler.passes import (
    AddPostSelectionMeasures,
    XSlowGate,
)


def test_empty_circuit():
    """Test the pass on an empty circuit."""
    circuit = QuantumCircuit(1)
    assert circuit == PassManager([AddPostSelectionMeasures()]).run(circuit)


def test_circuit_with_final_layer_of_measurements():
    """Test the pass on a circuit with a final layer of measurements."""
    qreg = QuantumRegister(4, "q")
    creg = ClassicalRegister(3, "c")
    creg_ps = ClassicalRegister(3, "c_ps")

    circuit = QuantumCircuit(qreg, creg)
    circuit.h(0)
    circuit.cz(0, 1)
    circuit.cz(1, 2)
    circuit.cz(2, 3)
    circuit.measure(1, creg[0])
    circuit.measure(2, creg[1])
    circuit.measure(3, creg[2])

    expected_circuit = QuantumCircuit(qreg, creg, creg_ps)
    expected_circuit.h(0)
    expected_circuit.cz(0, 1)
    expected_circuit.cz(1, 2)
    expected_circuit.cz(2, 3)
    expected_circuit.measure(1, creg[0])
    expected_circuit.measure(2, creg[1])
    expected_circuit.measure(3, creg[2])
    expected_circuit.barrier([1, 2, 3])
    expected_circuit.append(XSlowGate(), [1])
    expected_circuit.append(XSlowGate(), [2])
    expected_circuit.append(XSlowGate(), [3])
    expected_circuit.measure(1, creg_ps[0])
    expected_circuit.measure(2, creg_ps[1])
    expected_circuit.measure(3, creg_ps[2])

    assert expected_circuit == PassManager([AddPostSelectionMeasures()]).run(circuit)


def test_circuit_with_measurements_in_a_box():
    """Test the pass on a circuit with measurements inside a box."""
    qreg = QuantumRegister(4, "q")
    creg = ClassicalRegister(3, "c")
    creg_ps = ClassicalRegister(3, "c_ps")

    circuit = QuantumCircuit(qreg, creg)
    circuit.h(0)
    circuit.cz(0, 1)
    circuit.cz(1, 2)
    circuit.cz(2, 3)
    circuit.measure(1, creg[0])
    with circuit.box():
        circuit.measure(2, creg[1])
    circuit.measure(3, creg[2])

    expected_circuit = QuantumCircuit(qreg, creg, creg_ps)
    expected_circuit.h(0)
    expected_circuit.cz(0, 1)
    expected_circuit.cz(1, 2)
    expected_circuit.cz(2, 3)
    expected_circuit.measure(1, creg[0])
    with expected_circuit.box():
        expected_circuit.measure(2, creg[1])
    expected_circuit.measure(3, creg[2])
    expected_circuit.barrier([1, 2, 3])
    expected_circuit.append(XSlowGate(), [1])
    expected_circuit.append(XSlowGate(), [2])
    expected_circuit.append(XSlowGate(), [3])
    expected_circuit.measure(1, creg_ps[0])
    expected_circuit.measure(2, creg_ps[1])
    expected_circuit.measure(3, creg_ps[2])

    assert expected_circuit == PassManager([AddPostSelectionMeasures()]).run(circuit)


def test_if_else():
    """Test the pass for circuits with if/else statements."""
    qreg = QuantumRegister(5, "q")
    creg = ClassicalRegister(2, "c")
    creg_ps = ClassicalRegister(2, "c_ps")

    circuit = QuantumCircuit(qreg, creg)
    circuit.barrier(0)
    circuit.measure(0, creg[0])
    with circuit.if_test((creg[0], 0)) as else_:
        circuit.measure(1, creg[1])
    with else_:
        circuit.measure(2, creg[1])
    with circuit.if_test((creg[0], 0)) as else_:
        circuit.measure(3, creg[1])
    with else_:
        circuit.x(1)
        circuit.measure(3, creg[1])

    expected_circuit = QuantumCircuit(qreg, creg, creg_ps)
    expected_circuit.barrier(0)
    expected_circuit.measure(0, creg[0])
    with expected_circuit.if_test((creg[0], 0)) as else_:
        expected_circuit.measure(1, creg[1])
    with else_:
        expected_circuit.measure(2, creg[1])
    with expected_circuit.if_test((creg[0], 0)) as else_:
        expected_circuit.measure(3, creg[1])
    with else_:
        expected_circuit.x(1)
        expected_circuit.measure(3, creg[1])
    expected_circuit.barrier([0, 3])
    expected_circuit.append(XSlowGate(), [0])
    expected_circuit.append(XSlowGate(), [3])
    expected_circuit.measure(0, creg_ps[0])
    expected_circuit.measure(3, creg_ps[1])

    assert expected_circuit == PassManager([AddPostSelectionMeasures()]).run(circuit)


def test_circuit_with_mid_circuit_measurements():
    """Test the pass on a circuit with mid-circuit measurements."""
    qreg = QuantumRegister(3, "q")
    creg = ClassicalRegister(2, "c")
    creg_ps = ClassicalRegister(2, "c_ps")

    circuit = QuantumCircuit(qreg, creg)
    circuit.measure(1, 0)
    circuit.measure(2, 1)
    with circuit.box():
        circuit.x(1)

    expected_circuit = QuantumCircuit(qreg, creg, creg_ps)
    expected_circuit.measure(1, creg[0])
    expected_circuit.measure(2, creg[1])
    with expected_circuit.box():
        expected_circuit.x(1)
    expected_circuit.barrier([2])
    expected_circuit.append(XSlowGate(), [2])
    expected_circuit.measure(2, creg_ps[1])

    assert expected_circuit == PassManager([AddPostSelectionMeasures()]).run(circuit)


def test_circuit_with_multiple_cregs():
    """Test for a circuit with multiple cregs."""
    qreg = QuantumRegister(4, "q")
    creg1 = ClassicalRegister(1, "c1")
    creg2 = ClassicalRegister(2, "c2")
    creg1_ps = ClassicalRegister(1, "c1_ps")
    creg2_ps = ClassicalRegister(2, "c2_ps")

    circuit = QuantumCircuit(qreg, creg1, creg2)
    circuit.h(0)
    circuit.cz(0, 1)
    circuit.cz(1, 2)
    circuit.cz(2, 3)
    circuit.measure(1, creg1[0])
    circuit.measure(2, creg2[0])
    circuit.measure(3, creg2[1])

    expected_circuit = QuantumCircuit(qreg, creg1, creg2, creg1_ps, creg2_ps)
    expected_circuit.h(0)
    expected_circuit.cz(0, 1)
    expected_circuit.cz(1, 2)
    expected_circuit.cz(2, 3)
    expected_circuit.measure(1, creg1[0])
    expected_circuit.measure(2, creg2[0])
    expected_circuit.measure(3, creg2[1])
    expected_circuit.barrier([1, 2, 3])
    expected_circuit.append(XSlowGate(), [1])
    expected_circuit.append(XSlowGate(), [2])
    expected_circuit.append(XSlowGate(), [3])
    expected_circuit.measure(1, creg1_ps[0])
    expected_circuit.measure(2, creg2_ps[0])
    expected_circuit.measure(3, creg2_ps[1])

    assert expected_circuit == PassManager([AddPostSelectionMeasures()]).run(circuit)


def test_custom_post_selection_suffix():
    """Test the pass for a custom register suffix."""
    qreg = QuantumRegister(1, "q")
    creg = ClassicalRegister(1, "c")
    creg_ps = ClassicalRegister(1, "c_ciao")

    circuit = QuantumCircuit(qreg, creg)
    circuit.measure(0, 0)

    expected_circuit = QuantumCircuit(qreg, creg, creg_ps)
    expected_circuit.measure(0, [creg[0]])
    expected_circuit.barrier([0])
    expected_circuit.append(XSlowGate(), [0])
    expected_circuit.measure(0, creg_ps[0])

    pm = PassManager([AddPostSelectionMeasures(post_selection_suffix="_ciao")])
    assert expected_circuit == pm.run(circuit)


def test_x_pulse_type():
    """Test the pass for non-default X-pulse types."""
    qreg = QuantumRegister(1, "q")
    creg = ClassicalRegister(1, "c")
    creg_ps = ClassicalRegister(1, "c_ps")

    circuit = QuantumCircuit(qreg, creg)
    circuit.measure(0, 0)

    expected_circuit_rx = QuantumCircuit(qreg, creg, creg_ps)
    expected_circuit_rx.measure(0, [creg[0]])
    expected_circuit_rx.barrier([0])
    for _ in range(20):
        expected_circuit_rx.append(RXGate(np.pi / 20), [0])
    expected_circuit_rx.measure(0, creg_ps[0])

    pm = PassManager([AddPostSelectionMeasures(x_pulse_type="rx")])
    assert expected_circuit_rx == pm.run(circuit)


def test_raises():
    """Test that the pass raises."""
    with pytest.raises(ValueError):
        AddPostSelectionMeasures(x_pulse_type="rz")

    pm = PassManager([AddPostSelectionMeasures(x_pulse_type="rx")])
    circuit = QuantumCircuit(1)
    circuit.reset(0)
    with pytest.raises(TranspilerError, match="``'reset'`` is not supported"):
        pm.run(circuit)


def test_ignore_creg_suffixes_with_pre_selection():
    """Test that post-selection ignores pre-selection registers."""
    from qiskit_addon_utils.noise_management.post_selection.transpiler.passes import (
        AddPreSelectionMeasures,
    )

    qreg = QuantumRegister(2, "q")
    creg = ClassicalRegister(2, "c")

    # Create circuit with measurements
    circuit = QuantumCircuit(qreg, creg)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])

    # Apply pre-selection first, then post-selection
    coupling_map = [(0, 1)]
    pm = PassManager([AddPreSelectionMeasures(coupling_map), AddPostSelectionMeasures()])
    result = pm.run(circuit)

    # Should have c, c_pre, and c_ps registers (but NOT c_pre_ps)
    creg_names = [creg.name for creg in result.cregs]
    assert "c" in creg_names
    assert "c_pre" in creg_names
    assert "c_ps" in creg_names
    assert "c_pre_ps" not in creg_names

    # Verify the c_pre register was ignored (not copied to c_pre_ps)
    assert len(result.cregs) == 3


def test_ignore_creg_suffixes_custom():
    """Test custom ignore_creg_suffixes parameter."""
    qreg = QuantumRegister(2, "q")
    creg = ClassicalRegister(2, "c")
    creg_custom = ClassicalRegister(2, "c_custom")

    circuit = QuantumCircuit(qreg, creg, creg_custom)
    circuit.h(0)
    circuit.measure(0, creg[0])
    circuit.measure(1, creg_custom[0])

    # Ignore registers ending with "_custom"
    pm = PassManager([AddPostSelectionMeasures(ignore_creg_suffixes=["_custom"])])
    result = pm.run(circuit)

    # Should have c_ps but NOT c_custom_ps
    creg_names = [creg.name for creg in result.cregs]
    assert "c_ps" in creg_names
    assert "c_custom_ps" not in creg_names


def test_ignore_creg_suffixes_empty_list():
    """Test with empty ignore list - all registers should be copied."""
    qreg = QuantumRegister(2, "q")
    creg = ClassicalRegister(2, "c")
    creg_pre = ClassicalRegister(2, "c_pre")

    circuit = QuantumCircuit(qreg, creg, creg_pre)
    circuit.measure(0, creg[0])
    circuit.measure(1, creg_pre[0])

    # Don't ignore any registers
    pm = PassManager([AddPostSelectionMeasures(ignore_creg_suffixes=[])])
    result = pm.run(circuit)

    # Should have both c_ps and c_pre_ps
    creg_names = [creg.name for creg in result.cregs]
    assert "c_ps" in creg_names
    assert "c_pre_ps" in creg_names


def test_pre_then_post_selection():
    """Test applying pre-selection then post-selection passes."""
    from qiskit_addon_utils.noise_management.post_selection.transpiler.passes import (
        AddPreSelectionMeasures,
    )

    qreg = QuantumRegister(2, "q")
    creg = ClassicalRegister(2, "c")

    circuit = QuantumCircuit(qreg, creg)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])

    coupling_map = [(0, 1)]
    pm = PassManager([AddPreSelectionMeasures(coupling_map), AddPostSelectionMeasures()])
    result = pm.run(circuit)

    # Check structure: pre-selection measurements, barrier, main circuit, barrier, post-selection
    ops = [inst.operation.name for inst in result.data]

    # Should have xslow and x gates at the start (pre-selection)
    assert "xslow" in ops[:5]
    assert "x" in ops[:5]

    # Should have xslow gates at the end (post-selection, no x gate)
    assert "xslow" in ops[-5:]
    assert ops.count("barrier") >= 2  # At least 2 barriers


def test_post_then_pre_selection():
    """Test applying post-selection then pre-selection passes (different order)."""
    from qiskit_addon_utils.noise_management.post_selection.transpiler.passes import (
        AddPreSelectionMeasures,
    )

    qreg = QuantumRegister(2, "q")
    creg = ClassicalRegister(2, "c")

    circuit = QuantumCircuit(qreg, creg)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])

    # Apply post-selection first
    pm_post = PassManager([AddPostSelectionMeasures()])
    circuit_with_post = pm_post.run(circuit)

    # Then apply pre-selection
    coupling_map = [(0, 1)]
    pm_pre = PassManager([AddPreSelectionMeasures(coupling_map)])
    result = pm_pre.run(circuit_with_post)

    # Should have c, c_ps, and c_pre registers
    creg_names = [creg.name for creg in result.cregs]
    assert "c" in creg_names
    assert "c_ps" in creg_names
    assert "c_pre" in creg_names


def test_custom_suffixes_both_passes():
    """Test using custom suffixes for both pre and post selection."""
    from qiskit_addon_utils.noise_management.post_selection.transpiler.passes import (
        AddPreSelectionMeasures,
    )

    qreg = QuantumRegister(2, "q")
    creg = ClassicalRegister(2, "c")

    circuit = QuantumCircuit(qreg, creg)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])

    coupling_map = [(0, 1)]
    pm = PassManager(
        [
            AddPreSelectionMeasures(coupling_map, pre_selection_suffix="_init"),
            AddPostSelectionMeasures(
                post_selection_suffix="_check", ignore_creg_suffixes=["_init"]
            ),
        ]
    )
    result = pm.run(circuit)

    # Should have c, c_init, and c_check registers
    creg_names = [creg.name for creg in result.cregs]
    assert "c" in creg_names
    assert "c_init" in creg_names
    assert "c_check" in creg_names
    assert "c_init_check" not in creg_names
