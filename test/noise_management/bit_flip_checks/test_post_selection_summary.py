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
"""Tests for ``PostSelectionSummary``."""

from copy import deepcopy

import pytest
from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_addon_utils.noise_management.bit_flip_checks import PostSelectionSummary

_UNSET = object()


def _build_summary_circuit(reg_specs, num_qubits, *, pre=False, post=False):
    """Build a circuit from ``reg_specs`` (``(base_name, size)`` primaries over consecutive qubits);
    ``pre`` adds ``{name}_pre`` checks before a barrier, ``post`` adds ``{name}_ps`` checks after."""
    qreg = QuantumRegister(num_qubits, "q")
    primary = [ClassicalRegister(size, name) for name, size in reg_specs]
    regs = list(primary)
    pre_regs = post_regs = None
    if pre:
        pre_regs = [ClassicalRegister(size, f"{name}_pre") for name, size in reg_specs]
        regs += pre_regs
    if post:
        post_regs = [ClassicalRegister(size, f"{name}_ps") for name, size in reg_specs]
        regs += post_regs
    circuit = QuantumCircuit(qreg, *regs)

    def _measure(target_regs):
        q = 0
        for reg in target_regs:
            for bit in range(reg.size):
                circuit.measure(qreg[q], reg[bit])
                q += 1

    if pre:
        _measure(pre_regs)
        circuit.barrier()
    _measure(primary)
    if post:
        _measure(post_regs)
    return circuit


def _build_spectator_circuit(data_name="c", spec_name=None):
    """2-bit data primary with matching ``_ps``; ``spec_name`` adds a 1-bit spectator primary on a third qubit."""
    n_data = 2
    num_qubits = n_data + (1 if spec_name else 0)
    qreg = QuantumRegister(num_qubits, "q")
    data = ClassicalRegister(n_data, data_name)
    data_ps = ClassicalRegister(n_data, f"{data_name}_ps")
    regs = [data, data_ps]
    if spec_name:
        spec = ClassicalRegister(1, spec_name)
        spec_ps = ClassicalRegister(1, f"{spec_name}_ps")
        regs += [spec, spec_ps]
    circuit = QuantumCircuit(qreg, *regs)

    for i in range(n_data):
        circuit.measure(qreg[i], data[i])
    if spec_name:
        circuit.measure(qreg[n_data], spec[0])
    for i in range(n_data):
        circuit.measure(qreg[i], data_ps[i])
    if spec_name:
        circuit.measure(qreg[n_data], spec_ps[0])
    return circuit


def _base_summary(**overrides):
    """Return a fully-populated ``PostSelectionSummary`` with optional field overrides."""
    kwargs = dict(
        primary_cregs={"alpha", "spec"},
        measure_map={0: ("alpha", 0), 1: ("alpha", 1), 2: ("spec", 0)},
        edges={frozenset({0, 1}), frozenset({1, 2})},
        measure_map_ps={0: ("alpha_ps", 0), 1: ("alpha_ps", 1)},
        measure_map_pre={0: ("alpha_pre", 0), 1: ("alpha_pre", 1)},
        post_check_suffix="_ps",
        pre_check_suffix="_pre",
        spectator_cregs={"spec"},
    )
    kwargs.update(overrides)
    return PostSelectionSummary(**kwargs)


def test_init():
    """Test the constructor forwards every field to its getter."""
    summary = PostSelectionSummary(
        primary_cregs := {"alpha"},
        measure_map := {0: ("alpha", 0), 1: ("alpha", 1)},
        edges := {frozenset([0, 1])},
        measure_map_ps={0: ("alpha_ps", 0), 1: ("alpha_ps", 1)},
        measure_map_pre={0: ("alpha_pre", 0), 1: ("alpha_pre", 1)},
        post_check_suffix="_ps",
        pre_check_suffix="_pre",
        spectator_cregs={"alpha"},
    )

    assert summary.primary_cregs == primary_cregs
    assert summary.measure_map == measure_map
    assert summary.measure_map_ps == {0: ("alpha_ps", 0), 1: ("alpha_ps", 1)}
    assert summary.measure_map_pre == {0: ("alpha_pre", 0), 1: ("alpha_pre", 1)}
    assert summary.edges == edges
    assert summary.post_check_suffix == "_ps"
    assert summary.pre_check_suffix == "_pre"
    assert summary.spectator_cregs == {"alpha"}


@pytest.mark.parametrize(
    "reg_specs, num_qubits, pre, post, coupling_map, kwargs, expected",
    [
        # Post-check only (5 qubits, two primaries).
        (
            [("alpha", 3), ("beta", 2)],
            5,
            False,
            True,
            [(0, 1), (1, 2), (2, 3), (3, 4)],
            {"post_check_suffix": "_ps"},
            {
                "primary_cregs": {"alpha", "beta"},
                "measure_map": {
                    0: ("alpha", 0),
                    1: ("alpha", 1),
                    2: ("alpha", 2),
                    3: ("beta", 0),
                    4: ("beta", 1),
                },
                "edges": {frozenset(p) for p in [(0, 1), (1, 2), (2, 3), (3, 4)]},
                "post_check_suffix": "_ps",
            },
        ),
        # Pre-check only (5 qubits, two primaries).
        (
            [("alpha", 3), ("beta", 2)],
            5,
            True,
            False,
            [(0, 1), (1, 2), (2, 3), (3, 4)],
            {"pre_check_suffix": "_pre"},
            {
                "primary_cregs": {"alpha", "beta"},
                "measure_map": {
                    0: ("alpha", 0),
                    1: ("alpha", 1),
                    2: ("alpha", 2),
                    3: ("beta", 0),
                    4: ("beta", 1),
                },
                "measure_map_pre": {
                    0: ("alpha_pre", 0),
                    1: ("alpha_pre", 1),
                    2: ("alpha_pre", 2),
                    3: ("beta_pre", 0),
                    4: ("beta_pre", 1),
                },
                "measure_map_ps": {},
                "edges": {frozenset(p) for p in [(0, 1), (1, 2), (2, 3), (3, 4)]},
                "pre_check_suffix": "_pre",
            },
        ),
        # Combined pre- and post-check (3 qubits, single primary).
        (
            [("alpha", 3)],
            3,
            True,
            True,
            [(0, 1), (1, 2)],
            {"post_check_suffix": "_ps", "pre_check_suffix": "_pre"},
            {
                "primary_cregs": {"alpha"},
                "measure_map": {0: ("alpha", 0), 1: ("alpha", 1), 2: ("alpha", 2)},
                "measure_map_ps": {0: ("alpha_ps", 0), 1: ("alpha_ps", 1), 2: ("alpha_ps", 2)},
                "measure_map_pre": {
                    0: ("alpha_pre", 0),
                    1: ("alpha_pre", 1),
                    2: ("alpha_pre", 2),
                },
                "edges": {frozenset(p) for p in [(0, 1), (1, 2)]},
                "post_check_suffix": "_ps",
                "pre_check_suffix": "_pre",
            },
        ),
        # Pre-check with only some qubits terminally measured (lenient mode).
        (
            [("alpha", 2)],
            3,
            True,
            False,
            [(0, 1), (1, 2)],
            {"pre_check_suffix": "_pre"},
            {
                "primary_cregs": {"alpha"},
                "measure_map": {0: ("alpha", 0), 1: ("alpha", 1)},
                "measure_map_pre": {0: ("alpha_pre", 0), 1: ("alpha_pre", 1)},
                "measure_map_ps": {},
            },
        ),
    ],
)
def test_from_circuit(reg_specs, num_qubits, pre, post, coupling_map, kwargs, expected):
    """Test ``from_circuit`` builds the expected maps across post/pre/combined/partial circuits."""
    circuit = _build_summary_circuit(reg_specs, num_qubits, pre=pre, post=post)
    summary = PostSelectionSummary.from_circuit(circuit, coupling_map, **kwargs)

    for attr, value in expected.items():
        assert getattr(summary, attr) == value


@pytest.mark.parametrize(
    "overrides",
    [
        {"primary_cregs": {"alpha"}},
        {"measure_map": {0: ("alpha", 0)}},
        {"edges": {frozenset({0, 1})}},
        {"measure_map_ps": {0: ("alpha_ps", 1)}},
        {"measure_map_pre": {}},
        {"post_check_suffix": "ciao"},
        {"pre_check_suffix": "_different"},
        {"spectator_cregs": set()},
    ],
)
def test_eq_field_mutations(overrides):
    """Mutating any single ``__eq__`` comparand makes the summaries unequal."""
    assert _base_summary() != _base_summary(**overrides)


def test_eq_identity_and_type():
    """A summary equals its deepcopy and differs from a non-summary object."""
    summary = _base_summary()
    assert summary == deepcopy(summary)
    assert summary != 3


def test_invalid_cregs_raises():
    """Test that the constructor from circuits raises when the cregs are invalid."""
    qreg = QuantumRegister(5, "q")
    creg0 = ClassicalRegister(3, "alpha")
    creg1 = ClassicalRegister(2, "beta")
    creg0_ps = ClassicalRegister(3, "alpha_ps")

    circuit = QuantumCircuit(qreg, creg0, creg0_ps, creg1)
    with pytest.raises(ValueError):
        PostSelectionSummary.from_circuit(circuit, [])

    creg1_ps_invalid = ClassicalRegister(3, "beta_ps")
    circuit = QuantumCircuit(qreg, creg1, creg1_ps_invalid)
    with pytest.raises(ValueError):
        PostSelectionSummary.from_circuit(circuit, [])


def test_invalid_measure_maps_raises():
    """Test that from_circuit raises on mismatched primary/post-check measurement maps."""
    qreg = QuantumRegister(3, "q")
    creg0 = ClassicalRegister(3, "alpha")
    creg0_ps = ClassicalRegister(3, "alpha_ps")

    circuit = QuantumCircuit(qreg, creg0, creg0_ps)
    circuit.measure(qreg[0], creg0[0])
    circuit.measure(qreg, creg0_ps)
    with pytest.raises(ValueError, match="1 measurements and 3 post check measurements"):
        PostSelectionSummary.from_circuit(circuit, [])

    circuit = QuantumCircuit(qreg, creg0, creg0_ps)
    circuit.measure(qreg[0], creg0[0])
    circuit.measure(qreg[1], creg0_ps[0])
    with pytest.raises(ValueError, match="Missing post check measurement on qubit 0"):
        PostSelectionSummary.from_circuit(circuit, [])

    circuit = QuantumCircuit(qreg, creg0, creg0_ps)
    circuit.measure(qreg[0], creg0[0])
    circuit.measure(qreg[0], creg0_ps[1])
    with pytest.raises(ValueError, match="Expected measurement on qubit 0 writing to bit 0"):
        PostSelectionSummary.from_circuit(circuit, [])


def test_pre_check_invalid_measure_map_raises():
    """Test that pre-check validation raises when measure maps don't match."""
    qreg = QuantumRegister(3, "q")
    creg = ClassicalRegister(3, "alpha")
    creg_pre = ClassicalRegister(3, "alpha_pre")

    # Case: pre-check measurement writes to the wrong bit.
    circuit = QuantumCircuit(qreg, creg, creg_pre)
    circuit.measure(qreg[0], creg_pre[1])
    circuit.barrier()
    circuit.measure(qreg[0], creg[0])

    with pytest.raises(
        ValueError, match="Pre check measurement on qubit 0 writes to bit 1 of creg alpha_pre"
    ):
        PostSelectionSummary.from_circuit(circuit, [], pre_check_suffix="_pre")

    # Case: pre-check measurement writes to the wrong register (mismatched base name).
    creg_beta = ClassicalRegister(3, "beta")
    creg_alpha_pre2 = ClassicalRegister(3, "alpha_pre")
    creg_beta_pre = ClassicalRegister(3, "beta_pre")
    circuit2 = QuantumCircuit(qreg, creg, creg_beta, creg_alpha_pre2, creg_beta_pre)
    circuit2.measure(qreg[0], creg_beta_pre[0])  # to beta_pre instead of alpha_pre
    circuit2.barrier()
    circuit2.measure(qreg[0], creg[0])
    circuit2.measure(qreg[1], creg_beta[0])

    with pytest.raises(
        ValueError, match="Pre check measurement on qubit 0 writes to bit 0 of creg beta_pre"
    ):
        PostSelectionSummary.from_circuit(circuit2, [], pre_check_suffix="_pre")


@pytest.mark.parametrize(
    "data_name, spec_name, coupling_map, spectator_arg, expected_spectator, expected_primary",
    [
        # Default flags a register named ``spec`` as a spectator.
        ("c", "spec", [(0, 1), (1, 2)], _UNSET, {"spec"}, {"c", "spec"}),
        # Custom names override the default; a non-existent name is silently dropped.
        (
            "alpha",
            "spec_init",
            [(0, 1), (1, 2)],
            ["spec_init", "ghost"],
            {"spec_init"},
            {"alpha", "spec_init"},
        ),
        # No ``spec`` primary -> empty spectator set.
        ("c", None, [(0, 1)], _UNSET, set(), {"c"}),
        # Empty list opts out of the default ``spec`` classification.
        ("c", "spec", [(0, 1), (1, 2)], [], set(), {"c", "spec"}),
    ],
)
def test_spectator_cregs(
    data_name, spec_name, coupling_map, spectator_arg, expected_spectator, expected_primary
):
    """Test spectator-register classification across defaults, custom names, and opt-out."""
    circuit = _build_spectator_circuit(data_name, spec_name)
    kwargs = {} if spectator_arg is _UNSET else {"spectator_cregs": spectator_arg}
    summary = PostSelectionSummary.from_circuit(circuit, coupling_map, **kwargs)

    assert summary.spectator_cregs == expected_spectator
    assert summary.primary_cregs == expected_primary
    # Data-only view: caller can subtract.
    assert summary.primary_cregs - summary.spectator_cregs == expected_primary - expected_spectator


def test_spectator_post_check_with_data_pre_check():
    """A spectator primary (``spec`` + ``spec_ps``, no ``spec_pre``) is exempt from the pre-check
    partner requirement, so it coexists with data pre-checks and the summary still builds."""
    qreg = QuantumRegister(4, "q")
    creg_data = ClassicalRegister(3, "c")
    creg_data_pre = ClassicalRegister(3, "c_pre")
    creg_data_ps = ClassicalRegister(3, "c_ps")
    creg_spec = ClassicalRegister(1, "spec")
    creg_spec_ps = ClassicalRegister(1, "spec_ps")
    circuit = QuantumCircuit(qreg, creg_data, creg_data_pre, creg_data_ps, creg_spec, creg_spec_ps)
    # Data qubits: pre-check, primary, post-check. Spectator (q3): primary + post only.
    for i in range(3):
        circuit.measure(qreg[i], creg_data_pre[i])
        circuit.measure(qreg[i], creg_data[i])
        circuit.measure(qreg[i], creg_data_ps[i])
    circuit.measure(qreg[3], creg_spec[0])
    circuit.measure(qreg[3], creg_spec_ps[0])

    summary = PostSelectionSummary.from_circuit(circuit, [(0, 1), (1, 2), (2, 3)])

    assert summary.spectator_cregs == {"spec"}
    # Pre-check covers only the data qubits; the spectator has no pre-check.
    assert set(summary.measure_map_pre) == {0, 1, 2}
    # The data->spectator edge participates in edge-based post-check.
    assert frozenset({2, 3}) in summary.edges


def test_spectator_without_pre_partner_still_raises_when_not_a_spectator():
    """A non-spectator primary lacking a ``_pre`` partner still fails pre-check validation."""
    qreg = QuantumRegister(4, "q")
    creg_data = ClassicalRegister(3, "c")
    creg_data_pre = ClassicalRegister(3, "c_pre")
    creg_data_ps = ClassicalRegister(3, "c_ps")
    creg_spec = ClassicalRegister(1, "spec")
    creg_spec_ps = ClassicalRegister(1, "spec_ps")
    circuit = QuantumCircuit(qreg, creg_data, creg_data_pre, creg_data_ps, creg_spec, creg_spec_ps)
    for i in range(3):
        circuit.measure(qreg[i], creg_data_pre[i])
        circuit.measure(qreg[i], creg_data[i])
        circuit.measure(qreg[i], creg_data_ps[i])
    circuit.measure(qreg[3], creg_spec[0])
    circuit.measure(qreg[3], creg_spec_ps[0])

    # Opting out makes ``spec`` a regular primary, which then requires a missing ``spec_pre`` partner.
    with pytest.raises(ValueError, match="missing matching pre check register"):
        PostSelectionSummary.from_circuit(circuit, [(0, 1), (1, 2), (2, 3)], spectator_cregs=[])


def test_from_circuit_does_not_recurse_into_control_flow_ops():
    """Document a current limitation: measurements inside a ``ControlFlowOp`` (e.g. ``BoxOp``) are
    invisible to ``_get_measure_maps`` (top-level DAG only), so a boxed primary with an outside-box
    ``_ps`` measurement trips the strict primary<->ps count validation."""
    from qiskit.transpiler import PassManager
    from qiskit_addon_utils.noise_management.bit_flip_checks.passes import (
        AddPostCircuitBitFlipChecks,
    )

    qc = QuantumCircuit(3, 3)
    with qc.box():
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure([0, 1, 2], [0, 1, 2])  # measurements buried inside the box

    out = PassManager([AddPostCircuitBitFlipChecks(x_pulse_type="rx")]).run(qc)
    # Summary sees the top-level ``c_ps`` but not the boxed primaries, so the count check raises.
    with pytest.raises(ValueError, match="measurements"):
        PostSelectionSummary.from_circuit(out, [(0, 1), (1, 2)])
