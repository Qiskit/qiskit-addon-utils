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

"""Tests for the TREX error mitigation method."""

import types
import unittest

import numpy as np
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives.containers.estimator_pub import ObservablesArray
from qiskit.quantum_info import PauliLindbladMap, SparsePauliOp
from qiskit_addon_utils.noise_management.error_mitigation.executor_quantum_program_result import (
    ExecutorQuantumProgramResult,
)
from qiskit_addon_utils.noise_management.error_mitigation.trex import TREX
from samplomatic import ChangeBasis, Twirl


class TestTREXInit(unittest.TestCase):
    """Tests for TREX initialization."""

    def test_init_default_parameters(self):
        """Test TREX initialization with default parameters."""
        trex = TREX()
        self.assertIsNone(trex.inputs)
        self.assertIsNone(trex.noise)
        self.assertFalse(trex.options["twirl_gates"])
        self.assertIsNone(trex.options["twirling_strategy"])
        self.assertIsNone(trex.options["twirling_decomposition"])
        self.assertFalse(trex.options["twirl_mcm"])
        self.assertEqual(trex.options["shots_per_randomization"], 64)
        self.assertEqual(trex.options["num_randomizations"], 128)
        self.assertEqual(trex.options["cal_randomizations"], 128)

    def test_init_custom_parameters(self):
        """Test TREX initialization with custom parameters."""
        inputs = [
            (
                QuantumCircuit(2),
                [SparsePauliOp("ZZ")],
            )
        ]
        noise = PauliLindbladMap.from_sparse_list([], num_qubits=2)
        trex = TREX(
            inputs=inputs,
            noise=noise,
            twirl_gates=True,
            twirling_strategy="active",
            twirling_decomposition="on-the-fly",
            twirl_mcm=True,
            shots_per_randomization=128,
            num_randomizations=256,
            cal_randomizations=64,
        )
        self.assertEqual(trex.inputs, inputs)
        self.assertEqual(trex.noise, noise)
        self.assertTrue(trex.options["twirl_gates"])
        self.assertEqual(trex.options["twirling_strategy"], "active")
        self.assertEqual(trex.options["twirling_decomposition"], "on-the-fly")
        self.assertTrue(trex.options["twirl_mcm"])
        self.assertEqual(trex.options["shots_per_randomization"], 128)
        self.assertEqual(trex.options["num_randomizations"], 256)
        self.assertEqual(trex.options["cal_randomizations"], 64)

    def test_init_state_variables(self):
        """Test that state variables are properly initialized."""
        trex = TREX()
        self.assertEqual(trex.data_register_names, [])
        self.assertIsNone(trex.noise_learning_layer)
        self.assertIsNone(trex.annotated_circuits)
        self.assertEqual(trex.basis_dict_list, [])
        self.assertEqual(trex.measure_bases_list, [])
        self.assertEqual(trex.observables_list, [])


class TestTREXRemoveMidcircuitBoxAnnotations(unittest.TestCase):
    """Tests for _remove_midcircuit_box_annotations method."""

    def setUp(self):
        self.trex = TREX()

    def test_remove_midcircuit_box_annotations_remove_mcm(self):
        """Test that _remove_midcircuit_box_annotations removes midcircuit box annotations."""
        self.trex.options["twirl_mcm"] = False

        circuit = QuantumCircuit(3, 1)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        with circuit.box(
            [Twirl(group="pauli", dressing="left", decomposition="rzsx"), ChangeBasis()]
        ):
            circuit.measure(0, 0)
        circuit.rx(Parameter("th0"), 0)
        circuit.rx(Parameter("th1"), 1)
        circuit.rx(Parameter("th2"), 2)
        with circuit.box(
            [Twirl(group="pauli", dressing="left", decomposition="rzsx"), ChangeBasis()]
        ):
            circuit.measure_all()
        result_circ = self.trex._remove_midcircuit_box_annotations(circuit)

        self.assertEqual(result_circ.data[3].operation.annotations, [])
        self.assertIsNotNone(result_circ.data[-1].operation.annotations)
        self.assertEqual(len(result_circ.data[-1].operation.annotations), 2)
        self.assertEqual(
            result_circ.data[-1].operation.annotations[0],
            Twirl(group="pauli", dressing="left", decomposition="rzsx"),
        )
        self.assertIsInstance(result_circ.data[-1].operation.annotations[1], ChangeBasis)

    def test_remove_midcircuit_box_annotations_keep_twirl(self):
        """Test that _remove_midcircuit_box_annotations keeps Twirl annotations for midcircuit box if twirl_mcm is True."""
        self.trex.options["twirl_mcm"] = True

        circuit = QuantumCircuit(3, 1)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        with circuit.box(
            [Twirl(group="pauli", dressing="left", decomposition="rzsx"), ChangeBasis()]
        ):
            circuit.measure(0, 0)
        circuit.rx(Parameter("th0"), 0)
        circuit.rx(Parameter("th1"), 1)
        circuit.rx(Parameter("th2"), 2)
        with circuit.box(
            [Twirl(group="pauli", dressing="left", decomposition="rzsx"), ChangeBasis()]
        ):
            circuit.measure_all()
        result_circ = self.trex._remove_midcircuit_box_annotations(circuit)

        self.assertIsNotNone(result_circ.data[3].operation.annotations)
        self.assertEqual(len(result_circ.data[3].operation.annotations), 1)
        self.assertEqual(
            result_circ.data[3].operation.annotations[0],
            Twirl(group="pauli", dressing="left", decomposition="rzsx"),
        )
        self.assertIsNotNone(result_circ.data[-1].operation.annotations)
        self.assertEqual(len(result_circ.data[-1].operation.annotations), 2)
        self.assertEqual(
            result_circ.data[-1].operation.annotations[0],
            Twirl(group="pauli", dressing="left", decomposition="rzsx"),
        )
        self.assertIsInstance(result_circ.data[-1].operation.annotations[1], ChangeBasis)


class TestTREXAnnotateCircuitAndFindLayers(unittest.TestCase):
    """Tests for _annotate_circuit_and_find_layers method."""

    def setUp(self):
        self.trex = TREX()
        self.trex.options["twirl_gates"] = False
        self.trex.options["twirling_strategy"] = None
        self.trex.options["twirling_decomposition"] = None
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.rx(Parameter("th0"), 0)
        circuit.rx(Parameter("th1"), 1)
        circuit.rx(Parameter("th2"), 2)
        circuit.measure_all()
        self.test_circuit = circuit

    def test_annotate_circuit_contains_annotations(self):
        """Test that circuit annotation are added."""
        result_circ, result_layer = self.trex._annotate_circuit_and_find_layers(self.test_circuit)

        # check layer
        self.assertEqual(result_layer.operation.name, "box")
        self.assertEqual(len(result_layer.clbits), 3)
        # check circuit annotations
        self.assertEqual(len(result_circ.data[-1].operation.annotations), 2)
        self.assertIsInstance(result_circ.data[-1].operation.annotations[0], Twirl)
        self.assertIsInstance(result_circ.data[-1].operation.annotations[1], ChangeBasis)
        self.assertNotEqual(result_circ.data[0].operation.name, "box")

    def test_annotate_circuit_twirl_gates_annotations(self):
        """Test that Twirl annotations are added if twirl_gates is True."""
        self.trex.options["twirl_gates"] = True
        result_circ, result_layer = self.trex._annotate_circuit_and_find_layers(self.test_circuit)

        # check layer
        self.assertEqual(result_layer.operation.name, "box")
        self.assertEqual(len(result_layer.clbits), 3)
        # check circuit annotations
        self.assertEqual(len(result_circ.data[-1].operation.annotations), 2)
        self.assertIsInstance(result_circ.data[-1].operation.annotations[0], Twirl)
        self.assertIsInstance(result_circ.data[-1].operation.annotations[1], ChangeBasis)
        self.assertEqual(result_circ.data[0].operation.name, "box")
        self.assertEqual(len(result_circ.data[0].operation.annotations), 1)
        self.assertIsInstance(result_circ.data[0].operation.annotations[0], Twirl)

    def test_annotate_circuit_data_register_naming(self):
        """Test that data register names are tracked correctly."""
        self.assertEqual(len(self.trex.data_register_names), 0)

        self.trex._annotate_circuit_and_find_layers(self.test_circuit)

        self.assertEqual(len(self.trex.data_register_names), 1)
        self.assertEqual(self.trex.data_register_names[0][:9], "trex_data")


class TestTREXAnnotateCircuitsAndFindLayers(unittest.TestCase):
    """Tests for annotate_circuits_and_find_layers method."""

    def setUp(self):
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.rx(Parameter("th0"), 0)
        circuit.rx(Parameter("th1"), 1)
        circuit.rx(Parameter("th2"), 2)
        circuit.measure_all()

        mcm_circuit = QuantumCircuit(3, 1)
        mcm_circuit.h(0)
        mcm_circuit.cx(0, 1)
        mcm_circuit.cx(1, 2)
        mcm_circuit.measure(0, 0)
        mcm_circuit.rx(Parameter("th0"), 0)
        mcm_circuit.rx(Parameter("th1"), 1)
        mcm_circuit.rx(Parameter("th2"), 2)

        parameter_values = np.array([[0, 0, 0], [0, np.pi, np.pi], [np.pi, np.pi, np.pi]])

        observables = [
            [SparsePauliOp(op) for op in (["XXX"], ["IXI"], ["ZZZ"])],
            [SparsePauliOp("YYY")],
        ]
        inputs = [
            (circuit, observables[0], parameter_values),
            (mcm_circuit, observables[1], parameter_values),
        ]
        self.trex = TREX(inputs)

    def test_annotate_circuits_and_find_layers(self):
        """Test annotating circuits and finding layers."""
        noise_layer = self.trex.annotate_circuits_and_find_layers()

        self.assertIsInstance(noise_layer, QuantumCircuit)
        self.assertEqual(len(noise_layer.clbits), 3)
        self.assertEqual(len(noise_layer.data), 3)
        for inst in noise_layer.data:
            self.assertEqual(inst.name, "measure")


class TestTREXCreateTrexCalibrationCircuit(unittest.TestCase):
    """Tests for _create_trex_calibration_circuit method."""

    def setUp(self):
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.rx(Parameter("th0"), 0)
        circuit.rx(Parameter("th1"), 1)
        circuit.rx(Parameter("th2"), 2)
        circuit.measure_all()

        mcm_circuit = QuantumCircuit(3, 1)
        mcm_circuit.h(0)
        mcm_circuit.cx(0, 1)
        mcm_circuit.cx(1, 2)
        mcm_circuit.measure(0, 0)
        mcm_circuit.rx(Parameter("th0"), 0)
        mcm_circuit.rx(Parameter("th1"), 1)
        mcm_circuit.rx(Parameter("th2"), 2)

        parameter_values = np.array([[0, 0, 0], [0, np.pi, np.pi], [np.pi, np.pi, np.pi]])

        observables = [
            [SparsePauliOp(op) for op in (["XXX"], ["IXI"], ["ZZZ"])],
            [SparsePauliOp("YYY")],
        ]
        inputs = [
            (circuit, observables[0], parameter_values),
            (mcm_circuit, observables[1], parameter_values),
        ]
        self.trex = TREX(inputs)

    def test_create_calibration_circuit_no_layers_extracted(self):
        """Test None is returned if noise layers were not extracted first."""
        cal_circ = self.trex._create_trex_calibration_circuit()

        self.assertIsNone(cal_circ)

    def test_create_calibration_circuit(self):
        """Test creation of calibration circuit."""
        self.trex.annotate_circuits_and_find_layers()
        cal_circ = self.trex._create_trex_calibration_circuit()

        self.assertIsInstance(cal_circ, QuantumCircuit)
        self.assertEqual(len(cal_circ.clbits), 3)
        self.assertEqual(len(cal_circ.data), 1)
        self.assertEqual(cal_circ.data[0].name, "box")
        self.assertIsInstance(cal_circ.data[0].operation.annotations[0], Twirl)


class TestTREXPrepare(unittest.TestCase):
    """Tests for prepare method."""

    def test_prepare_calls_annotate_when_needed(self):
        """Test that prepare calls annotate_circuits_and_find_layers when needed."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        observable = SparsePauliOp("ZZ")

        trex = TREX(inputs=[(circuit, [observable])])
        # Initially, annotated_circuits and noise_learning_layer are None
        self.assertIsNone(trex.annotated_circuits)
        self.assertIsNone(trex.noise_learning_layer)

        trex.old_annotate_circuits_and_find_layers = trex.annotate_circuits_and_find_layers
        trex.called_annotate_circuits_and_find_layers = False

        def new_annotate_circuits_and_find_layers(self):
            self.called_annotate_circuits_and_find_layers = True
            return self.old_annotate_circuits_and_find_layers()

        trex.annotate_circuits_and_find_layers = types.MethodType(
            new_annotate_circuits_and_find_layers, trex
        )

        trex.prepare()
        self.assertEqual(trex.called_annotate_circuits_and_find_layers, True)

    def test_prepare_returns_executor_quantum_program(self):
        """Test prepare returned value."""

        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.rx(Parameter("th0"), 0)
        circuit.rx(Parameter("th1"), 1)
        circuit.rx(Parameter("th2"), 2)
        observables = [SparsePauliOp(op) for op in (["XIX"], ["ZIZ"])]

        parameter_values = np.array(
            [[0, 0, 0], [0, 0, np.pi], [0, np.pi, np.pi], [np.pi, np.pi, np.pi]]
        )

        trex = TREX(inputs=[(circuit, observables, parameter_values)])
        result_program = trex.prepare()
        self.assertEqual(len(result_program.items), 2)
        self.assertEqual(result_program.items[0].shape, (2, 128, 4))
        self.assertEqual(result_program.items[1].shape, (128,))

        parameter_values2 = np.array([[0, np.pi, np.pi], [np.pi, np.pi, np.pi]])

        trex = TREX(
            inputs=[
                (circuit, observables, parameter_values2),
                (circuit, observables, parameter_values),
            ]
        )
        result_program = trex.prepare()
        self.assertEqual(len(result_program.items), 3)
        self.assertEqual(result_program.items[0].shape, (2, 128, 2))
        self.assertEqual(result_program.items[1].shape, (2, 128, 4))
        self.assertEqual(result_program.items[2].shape, (128,))

    def test_prepare_condition_for_calibration(self):
        """Test prepare checks the condition for creating calibration circuit."""
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        observables = [SparsePauliOp(op) for op in (["XIX"], ["ZIZ"])]
        inputs = (circuit, observables)

        trex = TREX(inputs=[inputs])
        result_program = trex.prepare()
        self.assertEqual(len(result_program.items), 2)

        noise_list = [("X", [0], 0.001), ("X", [1], 0.002), ("X", [2], 0.003)]
        trex_with_noise = TREX(
            inputs=[inputs], noise=PauliLindbladMap.from_sparse_list(noise_list, num_qubits=3)
        )
        result_program_with_noise_prelearned = trex_with_noise.prepare()
        self.assertEqual(len(result_program_with_noise_prelearned.items), 1)


class TestTREXPostProcess(unittest.TestCase):
    """Tests for post_process method."""

    def test_post_process_no_noise_provided(self):
        """Test post-processing with no noise model provided."""
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.rx(Parameter("th0"), 0)
        circuit.rx(Parameter("th1"), 1)
        circuit.rx(Parameter("th2"), 2)
        observables = [SparsePauliOp(op) for op in (["XIX"], ["ZIZ"])]

        parameter_values = np.array(
            [[0, 0, 0], [0, 0, np.pi], [0, np.pi, np.pi], [np.pi, np.pi, np.pi]]
        )
        inputs = [(circuit, observables, parameter_values)]

        # Create a mock result object
        num_bases = len(observables)
        num_radomizations = 128
        num_parameters = len(parameter_values)
        num_shots = 64
        num_qubits = 3
        temp_bool_array = np.random.randint(
            0,
            2,
            size=(num_bases, num_radomizations, num_parameters, num_shots, num_qubits),
            dtype=bool,
        )
        temp_flips = np.random.randint(
            0, 2, size=(num_bases, num_radomizations, num_parameters, 1, num_qubits), dtype=bool
        )
        results_data = [{"trex_data": temp_bool_array, "measurement_flips.trex_data": temp_flips}]
        # add cal results
        temp_cal_bool_array = np.random.randint(
            0, 2, size=(num_radomizations, num_shots, num_qubits), dtype=bool
        )
        temp_cal_flips = np.random.randint(
            0, 2, size=(num_radomizations, 1, num_qubits), dtype=bool
        )
        results_data.append(
            {"trex_cal": temp_cal_bool_array, "measurement_flips.trex_cal": temp_cal_flips}
        )
        passthrough_data = {
            "_trex": {
                "observables": [ObservablesArray.coerce(observables)],
                "measure_bases": [["XIX", "ZIZ"]],
                "data_register_names": ["trex_data"],
            }
        }

        results = ExecutorQuantumProgramResult(data=results_data, passthrough_data=passthrough_data)

        trex = TREX(
            inputs=inputs, num_randomizations=num_radomizations, shots_per_randomization=num_shots
        )
        exp_vals_list, exp_vars_list = trex.post_process(results)
        self.assertEqual(
            np.array(exp_vals_list).shape, (len(inputs), len(observables), len(parameter_values))
        )
        self.assertEqual(
            np.array(exp_vars_list).shape, (len(inputs), len(observables), len(parameter_values))
        )
        self.assertIsNotNone(trex.noise)
        self.assertIsNotNone(trex.basis_dict_list)

    def test_post_process_with_noise_model(self):
        """Test post-processing with noise model provided."""
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.rx(Parameter("th0"), 0)
        circuit.rx(Parameter("th1"), 1)
        circuit.rx(Parameter("th2"), 2)
        observables = [SparsePauliOp(op) for op in (["XIX"], ["ZIZ"])]

        parameter_values = np.array(
            [[0, 0, 0], [0, 0, np.pi], [0, np.pi, np.pi], [np.pi, np.pi, np.pi]]
        )
        inputs = [(circuit, observables, parameter_values)]

        noise = PauliLindbladMap.from_sparse_list(
            [("X", [0], 0.01), ("X", [1], 0.02), ("X", [2], 0.03)], num_qubits=3
        )

        # Create a mock result object
        num_bases = len(observables)
        num_radomizations = 128
        num_parameters = len(parameter_values)
        num_shots = 64
        num_qubits = 3
        temp_bool_array = np.random.randint(
            0,
            2,
            size=(num_bases, num_radomizations, num_parameters, num_shots, num_qubits),
            dtype=bool,
        )
        temp_flips = np.random.randint(
            0, 2, size=(num_bases, num_radomizations, num_parameters, 1, num_qubits), dtype=bool
        )
        results_data = [{"trex_data": temp_bool_array, "measurement_flips.trex_data": temp_flips}]

        passthrough_data = {
            "_trex": {
                "observables": [ObservablesArray.coerce(observables)],
                "measure_bases": [["XIX", "ZIZ"]],
                "data_register_names": ["trex_data"],
            }
        }

        results = ExecutorQuantumProgramResult(data=results_data, passthrough_data=passthrough_data)

        trex = TREX(
            inputs=inputs,
            noise=noise,
            num_randomizations=num_radomizations,
            shots_per_randomization=num_shots,
        )
        exp_vals_list, exp_vars_list = trex.post_process(results)
        self.assertEqual(
            np.array(exp_vals_list).shape, (len(inputs), len(observables), len(parameter_values))
        )
        self.assertEqual(
            np.array(exp_vars_list).shape, (len(inputs), len(observables), len(parameter_values))
        )

    def test_post_process_multiple_inputs(self):
        """Test post-processing with multiple inputs."""
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.rx(Parameter("th0"), 0)
        circuit.rx(Parameter("th1"), 1)
        circuit.rx(Parameter("th2"), 2)
        observables = [SparsePauliOp(op) for op in (["XIX"], ["ZIZ"])]
        observables2 = [SparsePauliOp(op) for op in (["YYY"])]

        parameter_values = np.array(
            [[0, 0, 0], [0, 0, np.pi], [0, np.pi, np.pi], [np.pi, np.pi, np.pi]]
        )
        inputs = [
            (circuit, observables, parameter_values),
            (circuit, observables2, parameter_values),
        ]

        # Create a mock result object
        num_bases = len(observables)
        num_bases2 = len(observables2)
        num_radomizations = 128
        num_parameters = len(parameter_values)
        num_shots = 64
        num_qubits = 3
        temp_bool_array = np.random.randint(
            0,
            2,
            size=(num_bases, num_radomizations, num_parameters, num_shots, num_qubits),
            dtype=bool,
        )
        temp_flips = np.random.randint(
            0, 2, size=(num_bases, num_radomizations, num_parameters, 1, num_qubits), dtype=bool
        )
        results_data = [{"trex_data": temp_bool_array, "measurement_flips.trex_data": temp_flips}]
        temp_bool_array2 = np.random.randint(
            0,
            2,
            size=(num_bases2, num_radomizations, num_parameters, num_shots, num_qubits),
            dtype=bool,
        )
        temp_flips2 = np.random.randint(
            0, 2, size=(num_bases2, num_radomizations, num_parameters, 1, num_qubits), dtype=bool
        )
        results_data.append(
            {"trex_data1": temp_bool_array2, "measurement_flips.trex_data1": temp_flips2}
        )
        # add cal results
        temp_cal_bool_array = np.random.randint(
            0, 2, size=(num_radomizations, num_shots, num_qubits), dtype=bool
        )
        temp_cal_flips = np.random.randint(
            0, 2, size=(num_radomizations, 1, num_qubits), dtype=bool
        )
        results_data.append(
            {"trex_cal": temp_cal_bool_array, "measurement_flips.trex_cal": temp_cal_flips}
        )
        passthrough_data = {
            "_trex": {
                "observables": [
                    ObservablesArray.coerce(observables),
                    ObservablesArray.coerce(observables2),
                ],
                "measure_bases": [["XIX", "ZIZ"], ["YYY"]],
                "data_register_names": ["trex_data", "trex_data1"],
            }
        }

        results = ExecutorQuantumProgramResult(data=results_data, passthrough_data=passthrough_data)

        trex = TREX(
            inputs=inputs, num_randomizations=num_radomizations, shots_per_randomization=num_shots
        )
        exp_vals_list, exp_vars_list = trex.post_process(results)
        self.assertEqual(len(exp_vals_list), len(inputs))
        self.assertEqual(len(exp_vars_list), len(inputs))
        self.assertEqual(exp_vals_list[0].shape, (len(observables), len(parameter_values)))
        self.assertEqual(exp_vals_list[1].shape, (len(observables2), len(parameter_values)))
        self.assertEqual(exp_vars_list[0].shape, (len(observables), len(parameter_values)))
        self.assertEqual(exp_vars_list[1].shape, (len(observables2), len(parameter_values)))
        self.assertIsNotNone(trex.noise)
        self.assertIsNotNone(trex.basis_dict_list)


class TestTREXIntegration(unittest.TestCase):
    """Integration tests for TREX workflow."""

    def _create_res_vector(self, program, contains_cal):
        num_shots = program.shots
        results = []
        num_items = len(program.items)
        for i, item in enumerate(program.items):
            num_qubits = item.circuit.num_clbits
            bool_array_shape = (*item.shape, num_shots, num_qubits)
            temp_bool_array = np.random.randint(0, 2, size=bool_array_shape, dtype=bool)
            flips_array_shape = (*item.shape, 1, num_qubits)
            temp_flips = np.random.randint(0, 2, size=flips_array_shape, dtype=bool)
            if contains_cal and i == num_items - 1:
                # calibration result
                register_name = "trex_cal"
            else:
                register_name = program.passthrough_data["_trex"]["data_register_names"][i]
            results.append(
                {register_name: temp_bool_array, f"measurement_flips.{register_name}": temp_flips}
            )
        return ExecutorQuantumProgramResult(data=results, passthrough_data=program.passthrough_data)

    def test_trex_full_workflow(self):
        """Test a complete TREX workflow from initialization to post-processing."""
        num_qubits = 3
        circuit = QuantumCircuit(num_qubits)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        observables = [SparsePauliOp(op) for op in (["YYY"])]

        inputs = [(circuit, observables)]

        trex = TREX(
            inputs=inputs,
            twirl_gates=False,
            shots_per_randomization=64,
            num_randomizations=128,
        )
        trex_program = trex.prepare()
        results_data = self._create_res_vector(trex_program, contains_cal=True)

        self.assertIsNone(trex.noise)

        exp_vals_list, exp_vars_list = trex.post_process(results_data)
        self.assertEqual(len(exp_vals_list), len(inputs))
        self.assertEqual(len(exp_vars_list), len(inputs))
        self.assertEqual(exp_vals_list[0].shape, (1,))
        self.assertEqual(exp_vars_list[0].shape, (1,))
        self.assertIsNotNone(trex.noise)
        self.assertIsNotNone(trex.basis_dict_list)

    def test_trex_with_multiple_observables(self):
        """Test TREX with multiple observables per circuit and parameters."""
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.rx(Parameter("th0"), 0)
        circuit.rx(Parameter("th1"), 1)
        circuit.rx(Parameter("th2"), 2)
        observables = [SparsePauliOp(op) for op in (["XIX"], ["ZIZ"])]

        parameter_values = np.array(
            [[0, 0, 0], [0, 0, np.pi], [0, np.pi, np.pi], [np.pi, np.pi, np.pi]]
        )
        inputs = [(circuit, observables, parameter_values)]

        trex = TREX(
            inputs=inputs,
            twirl_gates=False,
            shots_per_randomization=64,
            num_randomizations=128,
        )
        trex_program = trex.prepare()
        results_data = self._create_res_vector(trex_program, contains_cal=True)

        self.assertIsNone(trex.noise)

        exp_vals_list, exp_vars_list = trex.post_process(results_data)
        self.assertEqual(len(exp_vals_list), len(inputs))
        self.assertEqual(len(exp_vars_list), len(inputs))
        self.assertEqual(exp_vals_list[0].shape, (len(observables), len(parameter_values)))
        self.assertEqual(exp_vars_list[0].shape, (len(observables), len(parameter_values)))
        self.assertIsNotNone(trex.noise)
        self.assertIsNotNone(trex.basis_dict_list)

    def test_trex_with_multiple_inputs(self):
        """Test TREX with multiple inputs."""
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.rx(Parameter("th0"), 0)
        circuit.rx(Parameter("th1"), 1)
        circuit.rx(Parameter("th2"), 2)
        observables = [SparsePauliOp(op) for op in (["XIX"], ["ZIZ"])]
        observables2 = [SparsePauliOp(op) for op in (["YYY"])]

        parameter_values = np.array(
            [[0, 0, 0], [0, 0, np.pi], [0, np.pi, np.pi], [np.pi, np.pi, np.pi]]
        )
        inputs = [
            (circuit, observables, parameter_values),
            (circuit, observables2, parameter_values),
        ]

        trex = TREX(
            inputs=inputs,
            twirl_gates=False,
            shots_per_randomization=64,
            num_randomizations=128,
        )
        trex_program = trex.prepare()
        results_data = self._create_res_vector(trex_program, contains_cal=True)

        self.assertIsNone(trex.noise)

        exp_vals_list, exp_vars_list = trex.post_process(results_data)
        self.assertEqual(len(exp_vals_list), len(inputs))
        self.assertEqual(len(exp_vars_list), len(inputs))
        self.assertEqual(exp_vals_list[0].shape, (len(observables), len(parameter_values)))
        self.assertEqual(exp_vars_list[0].shape, (len(observables), len(parameter_values)))
        self.assertEqual(exp_vals_list[1].shape, (len(observables2), len(parameter_values)))
        self.assertEqual(exp_vars_list[1].shape, (len(observables2), len(parameter_values)))
        self.assertIsNotNone(trex.noise)
        self.assertIsNotNone(trex.basis_dict_list)

    def test_trex_with_noise_pre_learned(self):
        """Test TREX with noise pre learned."""
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.rx(Parameter("th0"), 0)
        circuit.rx(Parameter("th1"), 1)
        circuit.rx(Parameter("th2"), 2)
        observables = [SparsePauliOp(op) for op in (["XIX"], ["ZIZ"])]

        parameter_values = np.array(
            [[0, 0, 0], [0, 0, np.pi], [0, np.pi, np.pi], [np.pi, np.pi, np.pi]]
        )
        inputs = [(circuit, observables, parameter_values)]

        noise_list = [("X", [0], 0.001), ("X", [1], 0.002), ("X", [2], 0.003)]

        trex = TREX(
            inputs=inputs,
            noise=PauliLindbladMap.from_sparse_list(noise_list, num_qubits=3),
            twirl_gates=False,
            shots_per_randomization=64,
            num_randomizations=128,
        )
        trex_program = trex.prepare()
        results_data = self._create_res_vector(trex_program, contains_cal=False)

        self.assertIsNotNone(trex.noise)

        exp_vals_list, exp_vars_list = trex.post_process(results_data)
        self.assertEqual(len(exp_vals_list), len(inputs))
        self.assertEqual(len(exp_vars_list), len(inputs))
        self.assertEqual(exp_vals_list[0].shape, (len(observables), len(parameter_values)))
        self.assertEqual(exp_vars_list[0].shape, (len(observables), len(parameter_values)))
        self.assertIsNotNone(trex.noise)
        self.assertIsNotNone(trex.basis_dict_list)


if __name__ == "__main__":
    unittest.main()
