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
"""End-to-end tests: build circuits via the passes, then check ``PostSelector`` masks.

Per-rule mask semantics live in ``test_post_selector.py``; here we cover only the
full-stack pipelines, node-vs-edge divergence, legacy-ordering compatibility, and
the passes' rerun no-op guards.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit_addon_utils.noise_management.bit_flip_checks import (
    PostSelector,
)
from qiskit_addon_utils.noise_management.bit_flip_checks.passes import (
    AddPostCircuitBitFlipChecks,
    AddPreCircuitBitFlipChecks,
    AddSpectatorPostCircuitBitFlipChecks,
    AddSpectatorPreCircuitBitFlipChecks,
)

# Coupling 4-0-1-2-3, data qubits 0,1,2 active ⇒ spec qubits {3, 4}.
COUPLING = [(0, 1), (1, 2), (2, 3), (0, 4)]
DATA_QUBITS = (0, 1, 2)
SPEC_QUBITS = (3, 4)


def _data_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(5, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure([0, 1, 2], [0, 1, 2])
    return qc


def _passes_post_with_spec():
    return [
        AddPostCircuitBitFlipChecks(x_pulse_type="rx"),
        AddSpectatorPostCircuitBitFlipChecks(COUPLING, x_pulse_type="rx"),
    ]


def _passes_pre_with_spec():
    return [
        AddPreCircuitBitFlipChecks(x_pulse_type="rx"),
        AddSpectatorPreCircuitBitFlipChecks(COUPLING, x_pulse_type="rx"),
    ]


def _passes_full_stack(*, pre_first: bool, custom: bool = False):
    if custom:
        pre_suffix, post_suffix, spec_pre_name = "_init", "_check", "spec_init"
        # Post-first ordering: the pre passes must ignore the post-sel suffix so
        # they don't treat ``_check`` registers as active/pre targets.
        pre_args = {
            "x_pulse_type": "rx",
            "pre_check_suffix": pre_suffix,
            "ignore_creg_suffixes": [post_suffix],
        }
        spec_pre_args = {
            "x_pulse_type": "rx",
            "spectator_creg_name": spec_pre_name,
            "pre_check_suffix": pre_suffix,
            "ignore_creg_suffixes": [post_suffix],
        }
        post_args = {
            "x_pulse_type": "rx",
            "post_check_suffix": post_suffix,
            "ignore_creg_suffixes": [pre_suffix],
        }
        spec_args = {
            "x_pulse_type": "rx",
            "ignore_creg_suffixes": [pre_suffix],
            "post_check_suffix": post_suffix,
        }
    else:
        pre_args = {"x_pulse_type": "rx"}
        spec_pre_args = {"x_pulse_type": "rx"}
        post_args = {"x_pulse_type": "rx"}
        spec_args = {"x_pulse_type": "rx"}

    pre_block = [
        AddPreCircuitBitFlipChecks(**pre_args),
        AddSpectatorPreCircuitBitFlipChecks(COUPLING, **spec_pre_args),
    ]
    post_block = [
        AddPostCircuitBitFlipChecks(**post_args),
        AddSpectatorPostCircuitBitFlipChecks(COUPLING, **spec_args),
    ]
    return pre_block + post_block if pre_first else post_block + pre_block


def _suffix_pair(custom: bool) -> tuple[str, str]:
    """Return (post_suffix, pre_suffix) for the given configuration."""
    return ("_check", "_init") if custom else ("_ps", "_pre")


def _perfect_result(
    *,
    n_shots: int = 4,
    has_post: bool = True,
    has_pre: bool = True,
    has_spec: bool = True,
    custom: bool = False,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Synthesise a result dict in which every check passes."""
    rng = np.random.default_rng(seed)
    post_suffix, _ = _suffix_pair(custom)
    pre_suffix = _suffix_pair(custom)[1]
    spec_pre_name = "spec_init" if custom else "spec_pre"

    c = rng.integers(0, 2, size=(n_shots, len(DATA_QUBITS)), dtype=bool)
    out: dict[str, np.ndarray] = {"c": c}
    if has_post:
        out[f"c{post_suffix}"] = ~c
    if has_pre:
        out[f"c{pre_suffix}"] = np.zeros((n_shots, len(DATA_QUBITS)), dtype=bool)

    if has_spec:
        spec = rng.integers(0, 2, size=(n_shots, len(SPEC_QUBITS)), dtype=bool)
        out["spec"] = spec
        if has_post:
            out[f"spec{post_suffix}"] = ~spec
        if has_pre:
            out[spec_pre_name] = np.zeros((n_shots, len(SPEC_QUBITS)), dtype=bool)

    return out


@pytest.mark.parametrize(
    "passes_factory,mode",
    [
        (_passes_post_with_spec, "post"),
        # Sole cover of the partial branch in post_selection_summary.py for a
        # spec-pre qubit lacking a primary measurement.
        (_passes_pre_with_spec, "pre"),
    ],
)
@pytest.mark.parametrize("strategy", ["node", "edge"])
def test_independent_passes_perfect_result_keeps_all(passes_factory, mode, strategy):
    """Independent pass sets: a perfect result keeps every shot."""
    pm = PassManager(passes_factory())
    circuit = pm.run(_data_circuit())
    selector = PostSelector.from_circuit(circuit, COUPLING)

    has_post = mode == "post"
    has_pre = mode == "pre"
    result = _perfect_result(has_post=has_post, has_pre=has_pre, has_spec=True)

    mask = selector.compute_mask(result, strategy, mode=mode)
    assert mask.all(), f"mode={mode} strategy={strategy}: expected all True"


@pytest.mark.parametrize("custom", [False, True], ids=["default_names", "custom_names"])
def test_full_stack_perfect_result_keeps_all(custom):
    """Full-stack ``mode="both"`` end-to-end, for default and custom suffixes."""
    pm = PassManager(_passes_full_stack(pre_first=True, custom=custom))
    circuit = pm.run(_data_circuit())
    post_suffix, pre_suffix = _suffix_pair(custom)
    spectator_cregs = ["spec"]  # spec primary keeps the default name even with custom suffixes
    selector = PostSelector.from_circuit(
        circuit,
        COUPLING,
        post_check_suffix=post_suffix,
        pre_check_suffix=pre_suffix,
        spectator_cregs=spectator_cregs,
    )

    result = _perfect_result(custom=custom)
    mask = selector.compute_mask(result, "edge", mode="both")
    assert mask.all()


def test_full_stack_spec_only_failure_node_vs_edge():
    """A spec-only failure is dropped by node-mode but kept by edge-mode.

    The canonical motivation for edge-mode: don't discard a shot just because a
    spectator misbehaved when its data neighbour was fine.
    """
    pm = PassManager(_passes_full_stack(pre_first=True))
    selector = PostSelector.from_circuit(pm.run(_data_circuit()), COUPLING)
    mm = selector.summary.measure_map
    post_suffix = selector.summary.post_check_suffix

    result = _perfect_result()
    # Break only the spec parity check on q3; edge (2,3) has data q2 still flipping.
    spec_name, spec_clbit = mm[3]
    result[f"{spec_name}{post_suffix}"][0, spec_clbit] = result[spec_name][0, spec_clbit]

    node_mask = selector.compute_mask(result, "node", mode="post")
    edge_mask = selector.compute_mask(result, "edge", mode="post")
    assert not node_mask[0]  # node-mode discards because q3 failed
    assert edge_mask[0]  # edge-mode keeps it because q2 still flipped
    assert node_mask[1:].all()
    assert edge_mask[1:].all()


# Legacy ordering (spec pass before post-sel pass) must stay compatible: the
# post-sel pass is defensively guarded against adding a duplicate ``_ps`` register,
# so it should still yield a correct PostSelector matching the recommended order.


def _passes_legacy_spec_then_post(*, spectator_creg_name: str = "spec", x_pulse_type: str = "rx"):
    """Old client pattern: spec measurements first, then the post-sel pass."""
    return [
        AddSpectatorPostCircuitBitFlipChecks(
            COUPLING, x_pulse_type=x_pulse_type, spectator_creg_name=spectator_creg_name
        ),
        AddPostCircuitBitFlipChecks(x_pulse_type=x_pulse_type),
    ]


def test_legacy_order_does_not_crash():
    """Spec-then-post ordering must complete without a duplicate-register crash."""
    pm = PassManager(_passes_legacy_spec_then_post())
    out = pm.run(_data_circuit())
    assert {cr.name for cr in out.cregs} == {"c", "spec", "spec_ps", "c_ps"}


@pytest.mark.parametrize("strategy", ["node", "edge"])
def test_legacy_and_recommended_orderings_produce_identical_masks(strategy):
    """Either ordering ⇒ identical masks.

    Shot 1 breaks q0 and q1 (a connected pair) so both strategies must discard it.
    """
    legacy_pm = PassManager(_passes_legacy_spec_then_post())
    new_pm = PassManager(_passes_post_with_spec())

    sel_legacy = PostSelector.from_circuit(legacy_pm.run(_data_circuit()), COUPLING)
    sel_new = PostSelector.from_circuit(new_pm.run(_data_circuit()), COUPLING)

    result = _perfect_result(has_pre=False)
    result["c_ps"][1, 0] = result["c"][1, 0]
    result["c_ps"][1, 1] = result["c"][1, 1]

    mask_legacy = sel_legacy.compute_mask(result, strategy, mode="post")
    mask_new = sel_new.compute_mask(result, strategy, mode="post")
    assert np.array_equal(mask_legacy, mask_new)
    assert not mask_legacy.all(), "expected at least one shot discarded"
    assert not mask_legacy[1], "shot 1 should be discarded under both strategies"


def test_post_sel_pass_rerun_is_noop():
    """Re-running ``AddPostCircuitBitFlipChecks`` on its own output is a no-op."""
    once = PassManager([AddPostCircuitBitFlipChecks(x_pulse_type="rx")]).run(_data_circuit())
    twice = PassManager([AddPostCircuitBitFlipChecks(x_pulse_type="rx")]).run(once)
    assert {cr.name for cr in once.cregs} == {cr.name for cr in twice.cregs}
    pulse_count_once = sum(1 for instr in once.data if instr.operation.name in ("rx", "xslow"))
    pulse_count_twice = sum(1 for instr in twice.data if instr.operation.name in ("rx", "xslow"))
    assert pulse_count_once == pulse_count_twice


def test_pre_sel_pass_rerun_is_noop():
    """Re-running ``AddPreCircuitBitFlipChecks`` on its own output is a no-op."""
    once = PassManager([AddPreCircuitBitFlipChecks(x_pulse_type="rx")]).run(_data_circuit())
    twice = PassManager([AddPreCircuitBitFlipChecks(x_pulse_type="rx")]).run(once)
    assert {cr.name for cr in once.cregs} == {cr.name for cr in twice.cregs}
    pulse_count_once = sum(1 for instr in once.data if instr.operation.name in ("rx", "xslow"))
    pulse_count_twice = sum(1 for instr in twice.data if instr.operation.name in ("rx", "xslow"))
    assert pulse_count_once == pulse_count_twice
