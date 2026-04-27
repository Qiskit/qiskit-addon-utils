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
"""End-to-end tests: build circuits via the transpiler passes, then verify that
``PostSelector`` produces masks consistent with the documented semantics.

The four masking rules under test:

* **pre-sel-node**: discard a shot iff *any* pre-sel measurement reads ``1``.
* **pre-sel-edge**: discard a shot iff some coupling-map pair has *both*
  pre-sel measurements reading ``1``.
* **post-sel-node**: discard a shot iff *any* (primary, ps) pair fails to flip.
* **post-sel-edge**: discard a shot iff some coupling-map pair has *both*
  (primary, ps) pairs failing to flip.

Tests exercise every combination of:

* pass set → ``post`` only / ``pre`` only / ``post + spec`` / ``pre + spec_pre`` / full stack
* pre-first vs post-first pass ordering (where applicable)
* default vs custom register suffixes / spectator-register name
* :math:`{\\rm mode} \\times {\\rm strategy}` -- all 6 mask paths.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit_addon_utils.noise_management.post_selection import (
    PostSelectionStrategy,
    PostSelector,
)
from qiskit_addon_utils.noise_management.post_selection.transpiler.passes import (
    AddPostSelectionMeasures,
    AddPreSelectionMeasures,
    AddSpectatorMeasures,
    AddSpectatorMeasuresPreSelection,
)

# 5-qubit register; data qubits 0,1,2 are active.
# Coupling 4-0-1-2-3 ⇒ spec qubits {3, 4}, edges {(0,1), (1,2), (2,3), (0,4)}.
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


# ---------------------------------------------------------------------------
# Pass-manager fixtures for each scenario.
# ---------------------------------------------------------------------------


def _passes_post_only():
    return [AddPostSelectionMeasures(x_pulse_type="rx")]


def _passes_pre_only():
    return [AddPreSelectionMeasures(COUPLING, x_pulse_type="rx")]


def _passes_post_with_spec():
    return [
        AddPostSelectionMeasures(x_pulse_type="rx"),
        AddSpectatorMeasures(COUPLING, x_pulse_type="rx"),
    ]


def _passes_pre_with_spec():
    return [
        AddPreSelectionMeasures(COUPLING, x_pulse_type="rx"),
        AddSpectatorMeasuresPreSelection(COUPLING, x_pulse_type="rx"),
    ]


def _passes_full_stack(*, pre_first: bool, custom: bool = False):
    if custom:
        pre_suffix, post_suffix, spec_pre_name = "_init", "_check", "spec_init"
        pre_args = {
            "x_pulse_type": "rx",
            "pre_selection_suffix": pre_suffix,
            # When the post block runs first, the pre pass would otherwise
            # mistakenly attach an ``_init`` register to ``c_check`` /
            # ``spec_check``; tell it to ignore the post-sel suffix.
            "ignore_creg_suffixes": [post_suffix],
        }
        spec_pre_args = {
            "x_pulse_type": "rx",
            "spectator_creg_name": spec_pre_name,
            "pre_selection_suffix": pre_suffix,
            # Same reason: in post-first ordering the spec qubits' ``spec_check``
            # measurement would otherwise mark them as "active" and exclude them
            # from the spectator set. Default ignores ``_ps``; we replace with
            # the custom post-sel suffix.
            "ignore_creg_suffixes": [post_suffix],
        }
        post_args = {
            "x_pulse_type": "rx",
            "post_selection_suffix": post_suffix,
            "ignore_creg_suffixes": [pre_suffix],
        }
        spec_args = {
            "x_pulse_type": "rx",
            "ignore_creg_suffixes": [pre_suffix],
            "post_selection_suffix": post_suffix,
        }
    else:
        pre_args = {"x_pulse_type": "rx"}
        spec_pre_args = {"x_pulse_type": "rx"}
        post_args = {"x_pulse_type": "rx"}
        spec_args = {"x_pulse_type": "rx"}

    pre_block = [
        AddPreSelectionMeasures(COUPLING, **pre_args),
        AddSpectatorMeasuresPreSelection(COUPLING, **spec_pre_args),
    ]
    post_block = [
        AddPostSelectionMeasures(**post_args),
        AddSpectatorMeasures(COUPLING, **spec_args),
    ]
    return pre_block + post_block if pre_first else post_block + pre_block


# ---------------------------------------------------------------------------
# Result synthesis. A "perfect" result keeps every shot for every mode/strategy.
# ---------------------------------------------------------------------------


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
    """Synthesise a result dict in which every check passes.

    * ``c`` (data primary) is random; ``c<post_suffix>`` = ``~c`` so every
      data parity check flips.
    * ``c<pre_suffix>`` is zeros so every initialization check passes.
    * Same shape for the ``spec`` family if ``has_spec``.

    The spec *primary* register is always named ``"spec"`` (the default of
    ``AddSpectatorMeasures``); the spec *pre* register name follows
    ``AddSpectatorMeasuresPreSelection`` — ``"spec_pre"`` by default,
    ``"spec_init"`` in the custom-suffix configuration.
    """
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
        # Spec primary always lives in "spec".
        out["spec"] = spec
        if has_post:
            # Spec post-sel uses the (possibly custom) post_selection_suffix on "spec".
            out[f"spec{post_suffix}"] = ~spec
        if has_pre:
            out[spec_pre_name] = np.zeros((n_shots, len(SPEC_QUBITS)), dtype=bool)

    return out


# ---------------------------------------------------------------------------
# Mask-correctness tests on hand-crafted results (each rule in isolation).
# ---------------------------------------------------------------------------


def test_pre_node_rule_discards_when_any_init_reads_one():
    pm = PassManager(_passes_pre_with_spec())
    selector = PostSelector.from_circuit(pm.run(_data_circuit()), COUPLING)

    result = _perfect_result(has_post=False)
    # Break: shot 1 has data q0 init=1, shot 2 has spec q3 init=1. Both must be discarded.
    result["c_pre"][1, 0] = True
    result["spec_pre"][2, 0] = True

    mask = selector.compute_mask(result, "node", mode="pre")
    expected = np.array([True, False, False, True])
    assert np.array_equal(mask, expected)


def test_pre_edge_keeps_when_at_least_one_neighbour_init_ok():
    pm = PassManager(_passes_pre_with_spec())
    selector = PostSelector.from_circuit(pm.run(_data_circuit()), COUPLING)

    result = _perfect_result(has_post=False, has_spec=False)
    # In pre-only-with-spec-pre, spec qubits don't have a primary measurement,
    # so we look up the pre creg directly via ``measure_map_pre``.
    result["spec_pre"] = np.zeros((4, len(SPEC_QUBITS)), dtype=bool)
    mmpre = selector.summary.measure_map_pre  # {qubit_idx: (pre_creg_name, clbit_idx)}

    def set_init(qubit_idx: int, shot: int, value: bool):
        name, clbit = mmpre[qubit_idx]
        result[name][shot, clbit] = value

    # Shot 0: qubit 0 init=1 alone — paired with q1=0 across edge (0,1) and q4=0
    # across (0,4). Edge mode keeps it.
    set_init(0, 0, True)
    # Shot 1: qubits 0 AND 1 init=1 — edge (0,1) has both 1 ⇒ discard.
    set_init(0, 1, True)
    set_init(1, 1, True)
    # Shot 2: qubits 0 AND 4 init=1 — edge (0,4) has both 1 ⇒ discard.
    set_init(0, 2, True)
    set_init(4, 2, True)
    # Shot 3: untouched ⇒ keep.

    mask = selector.compute_mask(result, "edge", mode="pre")
    expected = np.array([True, False, False, True])
    assert np.array_equal(mask, expected)


def test_post_node_rule_discards_when_any_pair_fails_to_flip():
    pm = PassManager(_passes_post_with_spec())
    selector = PostSelector.from_circuit(pm.run(_data_circuit()), COUPLING)

    result = _perfect_result(has_pre=False)
    # Break parity check: make c_ps == c on (shot, clbit).
    result["c_ps"][1, 0] = result["c"][1, 0]  # shot 1 data q0 fails to flip
    result["spec_ps"][2, 1] = result["spec"][2, 1]  # shot 2 spec q4 fails to flip

    mask = selector.compute_mask(result, "node", mode="post")
    expected = np.array([True, False, False, True])
    assert np.array_equal(mask, expected)


def test_post_edge_keeps_when_at_least_one_neighbour_flips():
    pm = PassManager(_passes_post_with_spec())
    selector = PostSelector.from_circuit(pm.run(_data_circuit()), COUPLING)
    mm = selector.summary.measure_map
    post_suffix = selector.summary.post_selection_suffix

    def break_flip(qubit_idx: int, shot: int):
        """Make qubit's parity check fail (primary == ps) on the given shot."""
        name, clbit = mm[qubit_idx]
        ps_arr = result[f"{name}{post_suffix}"]
        primary_arr = result[name]
        ps_arr[shot, clbit] = primary_arr[shot, clbit]

    result = _perfect_result(has_pre=False)
    # Shot 0: only q4 fails. Its only edge is (0,4); q0 still flipped ⇒ keep.
    break_flip(4, 0)
    # Shot 1: q0 AND q1 fail — edge (0,1) has both failing ⇒ discard.
    break_flip(0, 1)
    break_flip(1, 1)
    # Shot 2: q2 AND q3 fail — edge (2,3) has both failing ⇒ discard.
    break_flip(2, 2)
    break_flip(3, 2)
    # Shot 3: only q3 fails — its only edge (2,3) still has q2 flipped ⇒ keep.
    break_flip(3, 3)

    mask = selector.compute_mask(result, "edge", mode="post")
    expected = np.array([True, False, False, True])
    assert np.array_equal(mask, expected)


# ---------------------------------------------------------------------------
# Permutation matrix: pass set, strategy, mode (where applicable).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "passes_factory,supported_modes",
    [
        (_passes_post_only, ("post",)),
        (_passes_post_with_spec, ("post",)),
        (_passes_pre_only, ("pre",)),
        (_passes_pre_with_spec, ("pre",)),
    ],
)
@pytest.mark.parametrize("strategy", ["node", "edge"])
def test_independent_passes_perfect_result_keeps_all(passes_factory, supported_modes, strategy):
    """Independent pass sets: a perfect result keeps every shot."""
    pm = PassManager(passes_factory())
    circuit = pm.run(_data_circuit())
    selector = PostSelector.from_circuit(circuit, COUPLING)

    has_post = "post" in supported_modes
    has_pre = "pre" in supported_modes
    has_spec = passes_factory in (_passes_post_with_spec, _passes_pre_with_spec)
    result = _perfect_result(has_post=has_post, has_pre=has_pre, has_spec=has_spec)

    for mode in supported_modes:
        mask = selector.compute_mask(result, strategy, mode=mode)
        assert mask.all(), f"mode={mode} strategy={strategy}: expected all True"


@pytest.mark.parametrize("pre_first", [True, False], ids=["pre_first", "post_first"])
@pytest.mark.parametrize("custom", [False, True], ids=["default_names", "custom_names"])
@pytest.mark.parametrize("strategy", ["node", "edge"])
@pytest.mark.parametrize("mode", ["post", "pre", "both"])
def test_full_stack_perfect_result_keeps_all(pre_first, custom, strategy, mode):
    """Full-stack pipelines: perfect input ⇒ every shot kept, every mode/strategy."""
    pm = PassManager(_passes_full_stack(pre_first=pre_first, custom=custom))
    circuit = pm.run(_data_circuit())
    post_suffix, pre_suffix = _suffix_pair(custom)
    spectator_cregs = ["spec"]  # spec primary uses the default name even with custom suffixes
    selector = PostSelector.from_circuit(
        circuit,
        COUPLING,
        post_selection_suffix=post_suffix,
        pre_selection_suffix=pre_suffix,
        spectator_cregs=spectator_cregs,
    )

    result = _perfect_result(custom=custom)
    mask = selector.compute_mask(result, strategy, mode=mode)
    assert mask.all()


def test_full_stack_pass_order_yields_equivalent_summary():
    """Pre-first and post-first orderings yield the same summary.

    The mask depends only on (primary_cregs, measure_map, edges, suffixes) and
    those should be insensitive to whether the pre or post block ran first.
    """
    pm_pre_first = PassManager(_passes_full_stack(pre_first=True))
    pm_post_first = PassManager(_passes_full_stack(pre_first=False))
    s_pre = PostSelector.from_circuit(pm_pre_first.run(_data_circuit()), COUPLING).summary
    s_post = PostSelector.from_circuit(pm_post_first.run(_data_circuit()), COUPLING).summary

    assert s_pre == s_post


def test_full_stack_custom_suffixes_propagate():
    """Custom suffixes appear in the summary unchanged once forwarded."""
    pm = PassManager(_passes_full_stack(pre_first=True, custom=True))
    selector = PostSelector.from_circuit(
        pm.run(_data_circuit()),
        COUPLING,
        post_selection_suffix="_check",
        pre_selection_suffix="_init",
    )
    summary = selector.summary

    assert summary.post_selection_suffix == "_check"
    assert summary.pre_selection_suffix == "_init"
    assert summary.primary_cregs == {"c", "spec"}


# ---------------------------------------------------------------------------
# Cross-cutting: spec-only failure on an edge-mode post-sel run.
# ---------------------------------------------------------------------------


def test_full_stack_spec_only_failure_node_vs_edge():
    """A failure on a spec qubit alone is dropped by node-mode but kept by edge-mode.

    This is the canonical motivation for edge-mode: don't discard a shot just
    because one spectator misbehaved when its data neighbour was fine.
    """
    pm = PassManager(_passes_full_stack(pre_first=True))
    selector = PostSelector.from_circuit(pm.run(_data_circuit()), COUPLING)
    mm = selector.summary.measure_map
    post_suffix = selector.summary.post_selection_suffix

    result = _perfect_result()
    # Break only the spec parity check on q3 (edge (2,3) has data q2 still flipping).
    spec_name, spec_clbit = mm[3]
    result[f"{spec_name}{post_suffix}"][0, spec_clbit] = result[spec_name][0, spec_clbit]

    node_mask = selector.compute_mask(result, "node", mode="post")
    edge_mask = selector.compute_mask(result, "edge", mode="post")
    assert not node_mask[0]  # node-mode discards because q3 failed
    assert edge_mask[0]  # edge-mode keeps it because q2 still flipped
    # Other shots untouched.
    assert node_mask[1:].all()
    assert edge_mask[1:].all()


# ---------------------------------------------------------------------------
# Mode/strategy errors.
# ---------------------------------------------------------------------------


def test_mode_pre_raises_when_no_pre_sel_in_circuit():
    pm = PassManager(_passes_post_with_spec())
    selector = PostSelector.from_circuit(pm.run(_data_circuit()), COUPLING)
    with pytest.raises(ValueError, match="No pre-selection measurements"):
        selector.compute_mask(_perfect_result(has_pre=False), "node", mode="pre")


def test_mode_post_raises_when_no_post_sel_in_circuit():
    pm = PassManager(_passes_pre_with_spec())
    selector = PostSelector.from_circuit(pm.run(_data_circuit()), COUPLING)
    with pytest.raises(ValueError, match="No post-selection measurements"):
        selector.compute_mask(_perfect_result(has_post=False), "node", mode="post")


def test_mode_both_raises_when_either_missing():
    pm = PassManager(_passes_post_with_spec())
    selector = PostSelector.from_circuit(pm.run(_data_circuit()), COUPLING)
    with pytest.raises(ValueError, match="No pre-selection measurements"):
        selector.compute_mask(_perfect_result(has_pre=False), "node", mode="both")


def test_invalid_strategy_string_raises():
    pm = PassManager(_passes_post_only())
    selector = PostSelector.from_circuit(pm.run(_data_circuit()), COUPLING)
    with pytest.raises(ValueError):
        selector.compute_mask(_perfect_result(has_pre=False, has_spec=False), "diagonal")


def test_strategy_enum_and_string_equivalent():
    """``strategy`` accepts either the string name or the enum member."""
    pm = PassManager(_passes_post_only())
    selector = PostSelector.from_circuit(pm.run(_data_circuit()), COUPLING)
    result = _perfect_result(has_pre=False, has_spec=False)
    m_str = selector.compute_mask(result, "node", mode="post")
    m_enum = selector.compute_mask(result, PostSelectionStrategy.NODE, mode="post")
    assert np.array_equal(m_str, m_enum)
