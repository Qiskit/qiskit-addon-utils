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

"""Tests for the mitigation utils module."""

import unittest

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Pauli, PauliList, SparsePauliOp
from qiskit.quantum_info.random import random_pauli_list
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_addon_utils.exp_vals.expectation_values import executor_expectation_values
from qiskit_addon_utils.exp_vals.measurement_bases import _convert_basis_to_uint_representation
from qiskit_addon_utils.exp_vals.observable_mappings import map_observable_virtual_to_canonical
from samplomatic.annotations import ChangeBasis
from samplomatic.builders import build
from samplomatic.transpiler import generate_boxing_pass_manager
from samplomatic.utils import get_annotation


def create_and_sample_circ(layout, paulis, add_x, gate_twirl, meas_twirl):
    """Helper function that creates a circuit and samples it using a samplex."""
    backend = GenericBackendV2(num_qubits=5)
    observable = SparsePauliOp("".join(paulis[::-1]), 1.0)

    num_qubits = len(paulis)
    qc = QuantumCircuit(num_qubits)
    for q, P in enumerate(paulis):
        if add_x[q]:
            qc.x(q)
        if P == "X":
            qc.h(q)
        elif P == "Y":
            qc.h(q)
            qc.s(q)
    qc.measure_all()

    isa_pm = generate_preset_pass_manager(
        backend=backend, initial_layout=layout, optimization_level=0
    )
    isa_circuit = isa_pm.run(qc)
    reverser = {observable.paulis[0]: [observable]}

    measure_annotations = "all" if meas_twirl else "change_basis"
    if gate_twirl:
        boxes_pm = generate_boxing_pass_manager(
            twirling_strategy="active",
            measure_annotations=measure_annotations,
        )
    else:
        boxes_pm = generate_boxing_pass_manager(
            enable_gates=False,
            measure_annotations=measure_annotations,
        )

    n_rand = 10 if gate_twirl or meas_twirl else 1

    boxed_circuit = boxes_pm.run(isa_circuit)
    # create measurement bases using the canonical qubits, assuming the mapping functions are working as expected
    measurement_box = boxed_circuit.data[-1]
    canonical_qubits = [
        qubit._index for qubit in boxed_circuit.qubits if qubit in measurement_box.qubits
    ]
    observable_executor_cannon = map_observable_virtual_to_canonical(
        observable, layout, canonical_qubits
    )
    measurement_bases = _convert_basis_to_uint_representation(
        PauliList(observable_executor_cannon.paulis)
    )

    # build the samplex and sample some parameters
    template, samplex = build(boxed_circuit)
    inputs = samplex.inputs()
    samplex_input_inputs = {
        "basis_changes": {
            get_annotation(boxed_circuit[-1].operation, ChangeBasis).ref: measurement_bases[0]
        }
    }
    samplex_arguments = inputs.bind(**samplex_input_inputs).make_broadcastable()
    outputs = samplex.sample(samplex_arguments, num_randomizations=n_rand)

    sam = StatevectorSampler()

    job = sam.run([(template, outputs["parameter_values"])], shots=10_000)
    res = job.result()

    bool_array = res[0].data.meas.to_bool_array("little")
    meas_flips = outputs["measurement_flips.meas"] if meas_twirl else None
    if n_rand == 1:
        bool_array = bool_array[0]
        avg_axis = None
    else:
        avg_axis = 0

    return executor_expectation_values(
        bool_array, reverser, meas_basis_axis=None, avg_axis=avg_axis, measurement_flips=meas_flips
    )


def _assert_expectation_values(exp_vals, expected_val, expected_var):
    """Helper to assert expectation value and variance.

    Args:
        exp_vals: List of (expectation_value, variance) tuples
        expected_val: Expected expectation value
        expected_var: Expected variance
    """
    assert len(exp_vals) == 1, "Expected exactly one observable result"
    assert np.isclose(exp_vals[0][0], expected_val), (
        f"Expectation value mismatch: expected {expected_val}, got {exp_vals[0][0]}"
    )
    assert np.isclose(exp_vals[0][1], expected_var), (
        f"Variance mismatch: expected {expected_var}, got {exp_vals[0][1]}"
    )


@pytest.mark.parametrize(
    "qubit_layout,paulis,add_x,gate_twirl,meas_twirl,expected_val,expected_var",
    [
        # No twirls test cases
        pytest.param([1, 2], ["X", "I"], [False, False], False, False, 1, 0, id="no_twirls"),
        pytest.param(
            [4, 1],
            ["X", "I"],
            [False, False],
            False,
            False,
            1,
            0,
            id="no_twirls_reversed_layout",
        ),
        pytest.param(
            [1, 2], ["X", "I"], [True, False], False, False, -1, 0, id="no_twirls_with_x_gate"
        ),
        pytest.param(
            [4, 1],
            ["X", "I"],
            [True, False],
            False,
            False,
            -1,
            0,
            id="no_twirls_with_x_gate_reversed_layout",
        ),
        pytest.param(
            [4, 2, 0],
            ["X", "Z", "Y"],
            [True, False, False],
            False,
            False,
            -1,
            0,
            id="no_twirls_with_x_gate_reversed_layout_extended_basis",
        ),
        # Gate twirls test cases
        pytest.param([1, 2], ["X", "I"], [False, False], True, False, 1, 0, id="gate_twirls"),
        pytest.param(
            [4, 1],
            ["X", "I"],
            [False, False],
            True,
            False,
            1,
            0,
            id="gate_twirls_reversed_layout",
        ),
        pytest.param(
            [1, 2], ["X", "I"], [True, False], True, False, -1, 0, id="gate_twirls_with_x_gate"
        ),
        pytest.param(
            [4, 1],
            ["X", "I"],
            [True, False],
            True,
            False,
            -1,
            0,
            id="gate_twirls_with_x_gate_reversed_layout",
        ),
        pytest.param(
            [4, 2, 0],
            ["X", "Z", "Y"],
            [True, False, False],
            True,
            False,
            -1,
            0,
            id="gate_twirls_with_x_gate_reversed_layout_extended_basis",
        ),
        # Measurement twirls test cases
        pytest.param([1, 2], ["X", "I"], [False, False], False, True, 1, 0, id="meas_twirls"),
        pytest.param(
            [4, 1],
            ["X", "I"],
            [False, False],
            False,
            True,
            1,
            0,
            id="meas_twirls_reversed_layout",
        ),
        pytest.param(
            [1, 2], ["X", "I"], [True, False], False, True, -1, 0, id="meas_twirls_with_x_gate"
        ),
        pytest.param(
            [4, 1],
            ["X", "I"],
            [True, False],
            False,
            True,
            -1,
            0,
            id="meas_twirls_with_x_gate_reversed_layout",
        ),
        pytest.param(
            [4, 2, 0],
            ["X", "Z", "Y"],
            [True, False, False],
            False,
            True,
            -1,
            0,
            id="meas_twirls_with_x_gate_reversed_layout_extended_basis",
        ),
        # Gate and measurement twirls test cases
        pytest.param(
            [1, 2], ["X", "I"], [False, False], True, True, 1, 0, id="gate_and_meas_twirls"
        ),
        pytest.param(
            [4, 1],
            ["X", "I"],
            [False, False],
            True,
            True,
            1,
            0,
            id="gate_and_meas_twirls_reversed_layout",
        ),
        pytest.param(
            [1, 2],
            ["X", "I"],
            [True, False],
            True,
            True,
            -1,
            0,
            id="gate_and_meas_twirls_with_x_gate",
        ),
        pytest.param(
            [4, 1],
            ["X", "I"],
            [True, False],
            True,
            True,
            -1,
            0,
            id="gate_and_meas_twirls_with_x_gate_reversed_layout",
        ),
        pytest.param(
            [4, 2, 0],
            ["X", "Z", "Y"],
            [True, False, False],
            True,
            True,
            -1,
            0,
            id="gate_and_meas_twirls_with_x_gate_reversed_layout_extended_basis",
        ),
    ],
)
def test_executor_expectation_values(
    qubit_layout, paulis, add_x, gate_twirl, meas_twirl, expected_val, expected_var
):
    """Test expectation values with various twirling configurations."""
    exp_vals = create_and_sample_circ(
        layout=qubit_layout,
        paulis=paulis,
        add_x=add_x,
        gate_twirl=gate_twirl,
        meas_twirl=meas_twirl,
    )
    _assert_expectation_values(exp_vals, expected_val, expected_var)


class TestExecutorExpectationValuesInputValidation(unittest.TestCase):
    """Test input validation for executor_expectation_values function."""

    def _create_minimal_valid_inputs(self, num_bases=1, num_shots=100, num_qubits=2):
        """Helper to create minimal valid inputs for testing."""
        # Create bool_array with shape (num_bases, num_shots, num_qubits) if num_bases > 1
        # or (num_shots, num_qubits) if num_bases == 1
        if num_bases > 1:
            bool_array = np.random.randint(
                0, 2, size=(num_bases, num_shots, num_qubits), dtype=bool
            )
        else:
            bool_array = np.random.randint(0, 2, size=(num_shots, num_qubits), dtype=bool)

        # Create basis_dict with num_bases entries
        basis_dict = {}
        for i in range(num_bases):
            pauli_str = ["X", "Y", "Z"][i % 3] * num_qubits
            basis = Pauli(pauli_str)
            observable = SparsePauliOp(pauli_str, coeffs=[1.0])
            basis_dict[basis] = [observable]

        return bool_array, basis_dict

    def test_negative_avg_axis_single_value(self):
        """Test that negative avg_axis raises ValueError."""
        bool_array, basis_dict = self._create_minimal_valid_inputs()

        with self.assertRaises(ValueError) as context:
            executor_expectation_values(
                bool_array,
                basis_dict,
                meas_basis_axis=None,
                avg_axis=-1,
            )
        self.assertIn("nonnegative", str(context.exception))

    def test_negative_avg_axis_in_tuple(self):
        """Test that negative value in avg_axis tuple raises ValueError."""
        bool_array, basis_dict = self._create_minimal_valid_inputs(num_bases=2)

        with self.assertRaises(ValueError) as context:
            executor_expectation_values(
                bool_array,
                basis_dict,
                meas_basis_axis=0,
                avg_axis=(1, -1),
            )
        self.assertIn("nonnegative", str(context.exception))

    def test_meas_basis_axis_none_with_multiple_bases(self):
        """Test that meas_basis_axis=None with multiple bases raises ValueError."""
        bool_array, basis_dict = self._create_minimal_valid_inputs(num_bases=2)

        with self.assertRaises(ValueError) as context:
            executor_expectation_values(
                bool_array,
                basis_dict,
                meas_basis_axis=None,
            )
        self.assertIn(
            "`meas_basis_axis` cannot be `None` unless there is only one measurement basis",
            str(context.exception),
        )

    def test_basis_dict_length_mismatch(self):
        """Test that mismatched basis_dict length and bool_array shape raises ValueError."""
        bool_array, basis_dict = self._create_minimal_valid_inputs(num_bases=2)

        # Add a third basis to basis_dict but bool_array only has 2 along meas_basis_axis
        extra_basis = Pauli("ZZ")
        basis_dict[extra_basis] = [SparsePauliOp("ZZ", coeffs=[1.0])]

        with self.assertRaises(ValueError) as context:
            executor_expectation_values(
                bool_array,
                basis_dict,
                meas_basis_axis=0,
            )
        self.assertIn("len(basis_dict)", str(context.exception))
        self.assertIn("does not match", str(context.exception))

    def test_inconsistent_observable_counts(self):
        """Test that inconsistent observable counts in basis_dict raises ValueError."""
        bool_array, _ = self._create_minimal_valid_inputs(num_bases=2)

        # Create basis_dict with inconsistent observable counts
        basis1 = Pauli("ZZ")
        basis2 = Pauli("XX")
        obs1 = SparsePauliOp("ZZ", coeffs=[1.0])
        obs2 = SparsePauliOp("XX", coeffs=[1.0])
        obs3 = SparsePauliOp("ZI", coeffs=[1.0])

        basis_dict = {
            basis1: [obs1, obs2],  # 2 observables
            basis2: [obs3],  # 1 observable - inconsistent!
        }

        with self.assertRaises(ValueError) as context:
            executor_expectation_values(
                bool_array,
                basis_dict,
                meas_basis_axis=0,
            )
        self.assertIn("`basis_dict` indicates 2 observables, but entry", str(context.exception))

    def test_measurement_flips_shape_mismatch(self):
        """Test that measurement_flips with wrong shape causes issues."""
        bool_array, basis_dict = self._create_minimal_valid_inputs()

        # Create measurement_flips with wrong shape
        wrong_shape = (bool_array.shape[0] + 1, bool_array.shape[1])
        measurement_flips = np.random.randint(0, 2, size=wrong_shape, dtype=bool)

        # This should fail during execution (broadcasting error or similar)
        with self.assertRaises((ValueError, IndexError, RuntimeError)):
            executor_expectation_values(
                bool_array,
                basis_dict,
                meas_basis_axis=None,
                measurement_flips=measurement_flips,
            )

    def test_pauli_signs_shape_mismatch(self):
        """Test that pauli_signs with wrong shape causes issues."""
        bool_array, basis_dict = self._create_minimal_valid_inputs()

        # pauli_signs.shape[:-1] should equal bool_array.shape[:-2]
        wrong_shape = (bool_array.shape[0] + 1, 5)  # Wrong first dimension
        pauli_signs = np.random.randint(0, 2, size=wrong_shape, dtype=bool)

        # This should fail during execution
        with self.assertRaises((ValueError, IndexError, RuntimeError)):
            executor_expectation_values(
                bool_array,
                basis_dict,
                meas_basis_axis=None,
                pauli_signs=pauli_signs,
            )

    def test_postselect_mask_shape_mismatch(self):
        """Test that postselect_mask with wrong shape causes issues."""
        bool_array, basis_dict = self._create_minimal_valid_inputs()

        # postselect_mask.shape should equal bool_array.shape[:-1]
        wrong_shape = (bool_array.shape[0] + 1,)
        postselect_mask = np.ones(wrong_shape, dtype=bool)

        # This should fail during execution
        with self.assertRaises((ValueError, IndexError, RuntimeError)):
            executor_expectation_values(
                bool_array,
                basis_dict,
                meas_basis_axis=None,
                postselect_mask=postselect_mask,
            )

    def test_valid_single_qubit_single_shot(self):
        """Test valid edge case: single qubit, single shot."""
        bool_array = np.array([[True]], dtype=bool)  # 1 shot, 1 qubit
        basis_dict = {Pauli("Z"): [SparsePauliOp("Z", coeffs=[1.0])]}

        result = executor_expectation_values(
            bool_array,
            basis_dict,
            meas_basis_axis=None,
        )

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], tuple)
        self.assertEqual(len(result[0]), 2)  # (mean, variance)

    def test_valid_none_observable_in_basis_dict(self):
        """Test valid edge case: None as placeholder for zero operator."""
        bool_array, _ = self._create_minimal_valid_inputs(num_bases=2)

        basis1 = Pauli("ZZ")
        basis2 = Pauli("XX")
        obs1 = SparsePauliOp("ZZ", coeffs=[1.0])

        basis_dict = {
            basis1: [obs1],
            basis2: [None],  # None as placeholder
        }

        result = executor_expectation_values(
            bool_array,
            basis_dict,
            meas_basis_axis=0,
        )

        self.assertEqual(len(result), 1)

    def test_valid_empty_avg_axis_tuple(self):
        """Test valid edge case: avg_axis as empty tuple."""
        bool_array, basis_dict = self._create_minimal_valid_inputs()

        result = executor_expectation_values(
            bool_array,
            basis_dict,
            meas_basis_axis=None,
            avg_axis=tuple(),
        )

        self.assertEqual(len(result), 1)

    def test_valid_meas_basis_axis_and_avg_axis(self):
        """Test valid edge case: multiple axes in avg_axis."""
        # Create array with extra dimensions for averaging
        bool_array = np.random.randint(0, 2, size=(2, 3, 100, 2), dtype=bool)
        _, basis_dict = self._create_minimal_valid_inputs(num_bases=2)

        result = executor_expectation_values(
            bool_array,
            basis_dict,
            meas_basis_axis=0,
            avg_axis=(1,),  # Average over axis 1
        )

        self.assertEqual(len(result), 1)

    def test_valid_multiple_avg_axes(self):
        """Test valid edge case: multiple axes in avg_axis."""
        # Create array with extra dimensions for averaging
        bool_array = np.random.randint(0, 2, size=(1, 3, 4, 100, 2), dtype=bool)
        _, basis_dict = self._create_minimal_valid_inputs(num_bases=1)

        result = executor_expectation_values(
            bool_array,
            basis_dict,
            meas_basis_axis=0,
            avg_axis=(1, 2),  # Average over axes 1 and 2
        )

        self.assertEqual(len(result), 1)

    def test_avg_axis_as_list(self):
        """Test that avg_axis works with list (converted to tuple internally)."""
        bool_array = np.random.randint(0, 2, size=(3, 100, 2), dtype=bool)
        _, basis_dict = self._create_minimal_valid_inputs(num_bases=1)

        result = executor_expectation_values(
            bool_array,
            basis_dict,
            meas_basis_axis=None,
            avg_axis=[0],  # Pass as list instead of tuple
        )

        self.assertEqual(len(result), 1)

    def test_valid_with_all_optional_parameters(self):
        """Test valid case with all optional parameters provided correctly."""
        num_shots = 100
        num_qubits = 2
        bool_array = np.random.randint(0, 2, size=(num_shots, num_qubits), dtype=bool)
        basis_dict = {Pauli("ZZ"): [SparsePauliOp("ZZ", coeffs=[1.0])]}

        # Create correctly shaped optional parameters
        measurement_flips = np.random.randint(0, 2, size=(num_shots, num_qubits), dtype=bool)
        pauli_signs = np.random.randint(0, 2, size=(5,), dtype=bool)  # 5 error generators
        postselect_mask = np.ones((num_shots,), dtype=bool)

        result = executor_expectation_values(
            bool_array,
            basis_dict,
            meas_basis_axis=None,
            measurement_flips=measurement_flips,
            pauli_signs=pauli_signs,
            postselect_mask=postselect_mask,
            gamma_factor=1.5,
        )

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], tuple)
        self.assertEqual(len(result[0]), 2)


class TestExecutorExpectationValuesSimple(unittest.TestCase):
    """Test certain simple cases of executor_expectation_values function."""

    def _synthesize_data(self, job_shape, num_shots, num_bits, seed=None):
        rng = np.random.default_rng(seed=seed)
        bool_array = rng.integers(2, size=(*job_shape, num_shots, num_bits), dtype=bool)
        return bool_array

    def _synthesize_simple_basis_dict(self, bool_array, meas_axis=None, seed=None):
        num_bases = bool_array.shape[meas_axis] if meas_axis is not None else 1
        num_bits = bool_array.shape[-1]
        # will just evaluate a single Pauli with each basis:
        pauli_each_basis = random_pauli_list(num_bits, size=num_bases, seed=seed, phase=False)
        meas_bases = pauli_each_basis.copy()
        is_id = np.logical_not(meas_bases.z | meas_bases.x)
        meas_bases.z[is_id] = True  # valid meas basis has no `I`, so changing to `Z`
        meas_bases.phase = 0
        obs_terms_each_basis = []
        for i, p in enumerate(pauli_each_basis):
            row = [None] * len(pauli_each_basis)
            row[i] = SparsePauliOp(p)
            obs_terms_each_basis.append(row)
        basis_dict = dict(zip(meas_bases, obs_terms_each_basis))
        return basis_dict

    def _test_executor_expectation_values_noPEC_noPS_pauliObservables(
        self,
        job_shape,
        num_shots,
        num_bits,
        meas_basis_axis=None,
        avg_axis=None,
        measurement_flips=None,
        seed=None,
    ):
        """Test `executor_expectation_values` against a simplified, feature-light implementation"""

        bool_array = self._synthesize_data(job_shape, num_shots, num_bits, seed=seed)
        basis_dict = self._synthesize_simple_basis_dict(
            bool_array, meas_axis=meas_basis_axis, seed=seed
        )

        exp_vals = executor_expectation_values(
            bool_array,
            basis_dict,
            meas_basis_axis,
            avg_axis=avg_axis,
            measurement_flips=measurement_flips,
        )
        exp_vals = list([val for (val, var) in exp_vals])

        ## Independent computation to check answer:
        if measurement_flips is not None:
            bool_array ^= measurement_flips

        if avg_axis is not None:
            avg_axis = np.asarray(avg_axis)

        if meas_basis_axis is None:
            # Prepending new length-1 axis, for single meas basis:
            meas_basis_axis = 0
            bool_array = np.reshape(bool_array, (1, *bool_array.shape))
            if avg_axis is not None:
                avg_axis += 1

        # Will consume meas_basis_axis in for loop below,
        # which may shift avg_axis:
        if avg_axis is not None:
            avg_axis = np.asarray(avg_axis)
            avg_axis[avg_axis > meas_basis_axis] -= 1

        target_exp_vals = []
        for basis_idx, (_, spo_list) in enumerate(basis_dict.items()):
            spo = [x for x in spo_list if x is not None]
            assert len(spo) == 1  # so far, support evaluation of only one observable w each basis
            spo = spo[0]
            assert len(spo) == 1  # in this test, limit to one Pauli per observable
            pauli = spo.paulis[0]
            coeff = spo.coeffs[0].real
            support = pauli.z | pauli.x
            bool_subarray = np.asarray(np.take(bool_array, basis_idx, axis=meas_basis_axis))
            if bool_subarray.shape:
                bool_subarray = np.compress(support, bool_subarray, axis=-1)
            else:
                bool_subarray = bool_subarray * support
            result = (-1) ** (np.sum(bool_subarray, axis=-1))
            result = np.mean(result, axis=-1)  # average shots
            print(f"{result = }")
            if avg_axis is not None:
                avg_axis = tuple(int(a) for a in np.atleast_1d(avg_axis).ravel())
                result = np.mean(result, axis=tuple(int(a) for a in avg_axis))
            result *= coeff
            print(f"{result = }")
            target_exp_vals.append(result.tolist())

        return np.array(exp_vals), np.array(target_exp_vals)

    # def test_exp_val_job0D(self):
    #     evs, target_evs = self._test_executor_expectation_values_noPEC_noPS_pauliObservables(
    #         job_shape=tuple(),
    #         num_shots=10,
    #         num_bits=19,
    #         meas_basis_axis=None,
    #         avg_axis=None,
    #         measurement_flips=None,
    #         seed=None,
    #         )
    #     self.assertAlmostEqual(evs, target_evs)

    def test_exp_val_job1Da(self):
        evs, target_evs = self._test_executor_expectation_values_noPEC_noPS_pauliObservables(
            job_shape=(1,),
            num_shots=10,
            num_bits=19,
            meas_basis_axis=None,
            avg_axis=None,
            measurement_flips=None,
            seed=None,
        )
        self.assertTrue(np.allclose(evs, target_evs))

    def test_exp_val_job1Db(self):
        evs, target_evs = self._test_executor_expectation_values_noPEC_noPS_pauliObservables(
            job_shape=(5,),
            num_shots=10,
            num_bits=19,
            meas_basis_axis=None,
            avg_axis=None,
            measurement_flips=None,
            seed=None,
        )
        print(f"{evs = }")
        print(f"{target_evs = }")
        self.assertTrue(np.allclose(evs, target_evs))

    def test_exp_val_job2Da(self):
        evs, target_evs = self._test_executor_expectation_values_noPEC_noPS_pauliObservables(
            job_shape=(5, 1),
            num_shots=10,
            num_bits=19,
            meas_basis_axis=None,
            avg_axis=None,
            measurement_flips=None,
            seed=None,
        )
        self.assertTrue(np.allclose(evs, target_evs))

    def test_exp_val_job2Db(self):
        evs, target_evs = self._test_executor_expectation_values_noPEC_noPS_pauliObservables(
            job_shape=(5, 6),
            num_shots=10,
            num_bits=19,
            meas_basis_axis=None,
            avg_axis=None,
            measurement_flips=None,
            seed=None,
        )
        self.assertTrue(np.allclose(evs, target_evs))

    def test_exp_val_jobManyD(self):
        evs, target_evs = self._test_executor_expectation_values_noPEC_noPS_pauliObservables(
            job_shape=(1, 1, 1, 1, 1),
            num_shots=10,
            num_bits=19,
            meas_basis_axis=None,
            avg_axis=None,
            measurement_flips=None,
            seed=None,
        )
        self.assertTrue(np.allclose(evs, target_evs))

    def test_exp_val_jobGeneral(self):
        evs, target_evs = self._test_executor_expectation_values_noPEC_noPS_pauliObservables(
            job_shape=(2, 3, 4),
            num_shots=10,
            num_bits=19,
            meas_basis_axis=None,
            avg_axis=None,
            measurement_flips=None,
            seed=None,
        )
        self.assertTrue(np.allclose(evs, target_evs))

    def test_exp_val_measBasis(self):
        evs, target_evs = self._test_executor_expectation_values_noPEC_noPS_pauliObservables(
            job_shape=(1, 2, 3),
            num_shots=10,
            num_bits=19,
            meas_basis_axis=0,
            avg_axis=None,
            measurement_flips=None,
            seed=None,
        )
        self.assertTrue(np.allclose(evs, target_evs))

    def test_exp_val_measBases(self):
        evs, target_evs = self._test_executor_expectation_values_noPEC_noPS_pauliObservables(
            job_shape=(1, 2, 3),
            num_shots=10,
            num_bits=19,
            meas_basis_axis=1,
            avg_axis=None,
            measurement_flips=None,
            seed=None,
        )
        self.assertTrue(np.allclose(evs, target_evs))

    def test_exp_val_avgAxisLo(self):
        evs, target_evs = self._test_executor_expectation_values_noPEC_noPS_pauliObservables(
            job_shape=(1, 2, 3),
            num_shots=10,
            num_bits=19,
            meas_basis_axis=1,
            avg_axis=0,
            measurement_flips=None,
            seed=None,
        )
        self.assertTrue(np.allclose(evs, target_evs))

    def test_exp_val_avgAxisHi(self):
        evs, target_evs = self._test_executor_expectation_values_noPEC_noPS_pauliObservables(
            job_shape=(1, 2, 3),
            num_shots=10,
            num_bits=19,
            meas_basis_axis=1,
            avg_axis=2,
            measurement_flips=None,
            seed=None,
        )
        self.assertTrue(np.allclose(evs, target_evs))

    def test_exp_val_avgAxes(self):
        evs, target_evs = self._test_executor_expectation_values_noPEC_noPS_pauliObservables(
            job_shape=(1, 2, 3),
            num_shots=10,
            num_bits=19,
            meas_basis_axis=1,
            avg_axis=(0, 2),
            measurement_flips=None,
            seed=None,
        )
        self.assertTrue(np.allclose(evs, target_evs))

    def test_exp_val_measFlips(self):
        evs, target_evs = self._test_executor_expectation_values_noPEC_noPS_pauliObservables(
            job_shape=(1, 2, 3),
            num_shots=10,
            num_bits=19,
            meas_basis_axis=1,
            avg_axis=(0, 2),
            measurement_flips=np.ones((1, 2, 3, 10, 19), dtype=bool),
            seed=None,
        )
        self.assertTrue(np.allclose(evs, target_evs))
