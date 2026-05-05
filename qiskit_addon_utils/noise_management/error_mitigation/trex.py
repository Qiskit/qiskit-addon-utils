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

"""TREX quantum error mitigation method."""

import numpy as np
from qiskit.circuit import QuantumCircuit, CircuitInstruction
from qiskit import ClassicalRegister
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.circuit.controlflow.box import BoxOp
from qiskit.quantum_info import PauliLindbladMap, SparsePauliOp
from qiskit.primitives.containers.estimator_pub import ObservablesArray
from samplomatic.transpiler import generate_boxing_pass_manager
from samplomatic import build

from .executor_quantum_program import ExecutorQuantumProgram
from .executor_quantum_program_result import ExecutorQuantumProgramResult
from qiskit_addon_utils.exp_vals import get_measurement_bases
from qiskit_addon_utils.exp_vals.expectation_values import (
    executor_expectation_values,
    _find_measure_basis_to_observable_mapping,
)
from qiskit_addon_utils.noise_management.trex_factors import trex_factors


class TREX:
    def __init__(self, pubs, noise=None, shots_per_randomization=64, num_randomizations=128, cal_randomizations=128):
        """
        pubs - list of sets in the form of (circuit, list of observables, circuit parameters)
        """
        self.pubs = pubs
        self.shots_per_randomization = shots_per_randomization
        self.num_randomizations = num_randomizations
        self.cal_randomizations = cal_randomizations
        self.data_register_name = None
        self.noise_learning_layers = None
        self.annotated_circuits = None
        self.basis_dict_list = []
        self.measure_bases_list = []
        self.observables_list = []
        self.noise = noise
        self.mitigation_factors = []

    @staticmethod
    def remove_midcircuit_box_annotations(circuit: QuantumCircuit):
        """Return a new circuit with all annotations removed from every box, besides the final box.

            Args:
                circuit: The circuit whose box annotations to strip.

            Returns:
                A new circuit with empty annotation lists on every box, besides the final box.
        """
        new_circuit = circuit.copy_empty_like()
        for instr in circuit.data[:-1]:
            if instr.operation.name == "box":
                box = instr.operation
                new_box = BoxOp(body=box.body, label=box.label, annotations=[])
                new_circuit.data.append(CircuitInstruction(new_box, instr.qubits, instr.clbits))
            else:
                new_circuit.data.append(instr)
        new_circuit.data.append(circuit.data[-1])
        return new_circuit

    def _annotate_circuit_and_find_layers(self, circuit):
        """annotate the final measurement layer with ChangeBasis and Twirling annotations.

        Remove terminal measurements in the circuit if present, and replace with new terminal measurements on all the circuit's qubits,
        writing the results to a new dedicated classical register.
        """
        if circuit.layout:
            qubits_in_layout = circuit.layout.final_index_layout()
        else:
            qubits_in_layout = list(range(len(circuit.qubits)))
        edited_circuit = circuit.remove_final_measurements(inplace=False)
        qubits_register = ClassicalRegister._new_with_prefix(len(qubits_in_layout), "trex_data")
        self.data_register_name = qubits_register.name
        edited_circuit.add_register(qubits_register)
        edited_circuit.barrier(qubits_in_layout)
        edited_circuit.measure(qubits_in_layout, qubits_register)

        boxing_pm = generate_boxing_pass_manager(
            enable_gates=False,
            enable_measures=True,
            measure_annotations="all",
        )
        annotated_circuit = boxing_pm.run(edited_circuit)
        annotated_circuit = self.remove_midcircuit_box_annotations(annotated_circuit)
        return annotated_circuit, annotated_circuit.data[-1]

    def find_layers(self):
        noise_learning_layers = []
        annotated_circuits = []
        for pub in self.pubs:
            annotated_circuit, noise_learning_layer = self._annotate_circuit_and_find_layers(pub[0])
            noise_learning_layers.append(noise_learning_layer)
            annotated_circuits.append(annotated_circuit)
        self.noise_learning_layers = noise_learning_layers
        self.annotated_circuits = annotated_circuits

        return noise_learning_layers

    def _create_trex_calibration_circuit(self, measured_qubits):
        qubit_list = list(range(len(measured_qubits)))

        classical_cal_reg = ClassicalRegister(len(measured_qubits), name="trex_cal")
        trex_circuit = QuantumCircuit(len(measured_qubits))
        trex_circuit.add_register(classical_cal_reg)
        trex_circuit.measure(qubit_list, classical_cal_reg)
        trex_isa_pm = generate_preset_pass_manager(
            initial_layout=measured_qubits, optimization_level=0
        )
        trex_circuit = trex_isa_pm.run(trex_circuit)
        boxing_pm = generate_boxing_pass_manager(
            enable_gates=False,
            enable_measures=True,
            measure_annotations="twirl",
        )
        annotated_trex_circuit = boxing_pm.run(trex_circuit)
        return annotated_trex_circuit

    def prepare(self) -> ExecutorQuantumProgram:
        if not self.annotated_circuits or not self.noise_learning_layers:
            self.find_layers()

        # create ExecutorQuantumProgram
        program = ExecutorQuantumProgram(shots=self.shots_per_randomization)
        for index, pub in enumerate(self.pubs):
            annotated_circuit = self.annotated_circuits[index]
            measure_bases, basis_dict = get_measurement_bases(pub[1])
            self.measure_bases_list.append(measure_bases)
            self.basis_dict_list.append(basis_dict)
            self.observables_list.append(pub[1])

            for register in annotated_circuit.cregs:
                if register.name == self.data_register_name:
                    num_qubits = len(register)
            # broadcast measurement basis shape
            if len(pub) > 2:
                parameter_values = pub[2]
                # add dimension also for the twirling randomizations
                bases_shape = (
                    (len(measure_bases),)
                    + (1,)
                    + (1,) * len(parameter_values.shape[:-1])
                    + (num_qubits,)
                )
                measure_bases_broadcastable = np.array(measure_bases).reshape(bases_shape)
                samplex_shape = (
                    (len(measure_bases),)
                    + (self.num_randomizations,)
                    + (1,) * len(parameter_values.shape[:-1])
                )
            else:
                # add dimension also for the twirling randomizations
                bases_shape = (len(measure_bases),) + (1,) + (num_qubits,)
                measure_bases_broadcastable = np.array(measure_bases).reshape(bases_shape)
                samplex_shape = (len(measure_bases),) + (self.num_randomizations,)

            template_circuit, samplex = build(annotated_circuit)
            # Generate `samplex_arguments` for the executor
            samplex_arguments = samplex.inputs().make_broadcastable()
            basis_changes_name = samplex.inputs().get_specs("basis_changes")[0].name
            if len(pub) > 2:
                samplex_arguments.bind(
                    **{
                        "parameter_values": pub[2],
                        basis_changes_name: measure_bases_broadcastable,
                    }
                )
            else:
                samplex_arguments.bind(
                    **{
                        basis_changes_name: measure_bases_broadcastable,
                    }
                )
            program.append_samplex_item(
                template_circuit,
                samplex=samplex,
                samplex_arguments=samplex_arguments,
                shape=samplex_shape,
            )

        # in case the noise was not learned before executing, add the noise learning circuit to the execution
        if not self.noise:
            measured_qubits = set()
            for circ_index, annotated_circuit in enumerate(self.annotated_circuits):
                # assuming all the measurement instructions are boxed together in the last instruction of the circuit
                if isinstance(annotated_circuit.data[-1].operation, BoxOp):
                    circuit_measured_qubits = [
                        qubit._index for qubit in annotated_circuit.data[-1].qubits
                    ]
                    measured_qubits.update(circuit_measured_qubits)
            calibration_circuit = self._create_trex_calibration_circuit(measured_qubits)

            template_calibration_circuit, calibration_samplex = build(calibration_circuit)
            program.append_samplex_item(
                template_calibration_circuit,
                samplex=calibration_samplex,
                shape=(self.cal_randomizations),
            )
        # save data in the program for post processing
        observables_arr = [ObservablesArray.coerce(observables) for observables in self.observables_list]
        program.passthrough_data = {
            "_trex": {
                "observables": observables_arr,
                "measure_bases": self.measure_bases_list,
                "data_register_name": self.data_register_name,
            }
        }
        return program

    def post_process(self, results: ExecutorQuantumProgramResult):
        data_results = results
        if not self.noise:
            # assume a calibration circuit was added to the quantum program as the last item
            noise_learning_result = results[-1]
            measurement_flips = noise_learning_result["measurement_flips.trex_cal"]
            noise_calibration_data = noise_learning_result["trex_cal"]
            noise_calibration_data_flipped = np.logical_xor(
                noise_calibration_data, measurement_flips
            )
            noise_list = []
            num_qubits = noise_calibration_data.shape[-1]
            for qubit_index in range(num_qubits):
                # the shape of the calibration data is (randomizations, shots, measured_qubit)
                excited_state_count = np.sum(noise_calibration_data_flipped[:, :, qubit_index])
                total_shots = len(noise_calibration_data_flipped[:, :, qubit_index].flatten())
                flip_rate = excited_state_count / total_shots
                noise_list.append(("X", [qubit_index], flip_rate))
            readout_noise = PauliLindbladMap.from_sparse_list(noise_list, num_qubits=num_qubits)
            self.noise = readout_noise
            data_results = results[:-1]

        if not self.basis_dict_list:
            observables_list = []
            for observables in results.passthrough_data["_trex"]["observables"]:
                observables_list.append([SparsePauliOp(observable) for observable in observables])
            bases_list = results.passthrough_data["_trex"]["measure_bases"]
            # TODO: change trex_factors so it can get a tuple of (observables, bases) as input
            for bases, observables in zip(bases_list, observables_list):
                self.basis_dict_list.append(
                    _find_measure_basis_to_observable_mapping(observables, bases)
                )

        exp_vals_list = []
        exp_vars_list = []
        data_register_name = results.passthrough_data["_trex"]["data_register_name"]
        for result_index, result in enumerate(data_results):
            measurement_flips = result[f"measurement_flips.{data_register_name}"]
            meas = result[data_register_name]
            basis_mapping = self.basis_dict_list[result_index]
            trex_factors_per_basis = trex_factors(self.noise, basis_mapping)

            if len(basis_mapping) > 1:
                avg_axes = tuple(range(1, len(meas.shape[1:-2])))
                meas_basis_axis = 0
            else:
                avg_axes = tuple(range(len(meas.shape[:-2])))
                meas_basis_axis = None

            res = executor_expectation_values(
                meas,
                basis_mapping,
                meas_basis_axis=meas_basis_axis,
                avg_axis=avg_axes,
                measurement_flips=measurement_flips,
                rescale_factors=trex_factors_per_basis,
            )
            res = np.array(res)
            exp_vals, exp_vars = res[:, 0], res[:, 1]
            exp_vals_list.append(exp_vals)
            exp_vars_list.append(exp_vars)
        return exp_vals_list, exp_vars_list
