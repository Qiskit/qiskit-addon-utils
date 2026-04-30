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
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.circuit.controlflow.box import BoxOp
from qiskit.quantum_info import PauliLindbladMap
from samplomatic.transpiler import generate_boxing_pass_manager
from samplomatic import build
from qiskit_ibm_runtime import QuantumProgram

from qiskit_addon_utils.exp_vals import get_measurement_bases
from qiskit_addon_utils.exp_vals.expectation_values import (
    executor_expectation_values,
    _find_measure_basis_to_observable_mapping,
)
from qiskit_addon_utils.noise_management.trex_factors import trex_factors


class TREX:
    def __init__(self, pubs, noise=None, shots_per_randomization=64, num_randomizations=128):
        """
        pubs - list of sets in the form of (circuit, list of observables, circuit parameters)
        """
        self.pubs = pubs
        self.shots_per_randomization = shots_per_randomization
        self.num_randomizations = num_randomizations
        self.noise_learning_layers = None
        self.annotated_circuits = None
        self.basis_dict_list = []
        self.measure_bases_list = []
        self.noise = noise
        self.mitigation_factors = []

    def _annotate_circuit_and_find_layers(self, circuit):
        boxing_pm = generate_boxing_pass_manager(
            enable_gates=False,
            enable_measures=True,
            measure_annotations="all",
        )
        annotated_circuit = boxing_pm.run(circuit)
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

        trex_circuit = QuantumCircuit(len(measured_qubits), len(measured_qubits))
        trex_circuit.measure(qubit_list, qubit_list)
        trex_isa_pm = generate_preset_pass_manager(
            initial_layout=measured_qubits, optimization_level=0
        )
        trex_circuit = trex_isa_pm.run(trex_circuit)
        return trex_circuit

    def prepare(self):
        if not self.annotated_circuits or not self.noise_learning_layers:
            self.find_layers()

        # create QuantumProgram
        program = QuantumProgram(shots=self.shots_per_randomization)
        for index, pub in enumerate(self.pubs):
            annotated_circuit = self.annotated_circuits[index]
            measure_bases, basis_dict = get_measurement_bases(pub[1])
            self.measure_bases_list.append(measure_bases)
            self.basis_dict_list.append(basis_dict)
            # broadcast measurement basis shape
            if len(pub) > 2:
                parameter_values = pub[2]
                # add dimension also for the twirling randomizations
                bases_shape = (
                    (len(measure_bases),)
                    + (1,)
                    + (1,) * len(parameter_values.shape[:-1])
                    + (annotated_circuit.num_qubits,)
                )
                measure_bases_broadcastable = np.array(measure_bases).reshape(bases_shape)
                samplex_shape = (
                    (len(measure_bases),)
                    + (self.num_randomizations,)
                    + (1,) * len(parameter_values.shape[:-1])
                )
            else:
                # add dimension also for the twirling randomizations
                bases_shape = (len(measure_bases),) + (1,) + (annotated_circuit.num_qubits,)
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
                shape=(self.num_randomizations),
            )

        return program

    def post_process(self, results, bases_list=None, observables_list=None):
        if not self.noise:
            # assume a calibration circuit was added to the quantum program as the last item
            noise_learning_result = results[-1]
            measurement_flips = noise_learning_result["measurement_flips.meas"]
            noise_calibration_data = noise_learning_result["meas"]
            noise_calibration_data_flipped = np.logical_xor(
                noise_calibration_data, measurement_flips
            )
            noise_list = []
            num_qubits = len(noise_calibration_data.shape[-1])
            for qubit_index in range(num_qubits):
                # the shape of the calibration data is (randomizations, shots, measured_qubit)
                excited_state_count = np.sum(noise_calibration_data_flipped[:, :, 0])
                total_shots = len(noise_calibration_data_flipped[:, :, 0].flatten())
                flip_rate = excited_state_count / total_shots
                noise_list.append(("X", [qubit_index], flip_rate))
            readout_noise = PauliLindbladMap.from_sparse_list(noise_list, num_qubits=num_qubits)
            self.noise = readout_noise

        if not self.basis_dict_list:
            # TODO: change trex_factors so it can get a tuple of (observables, bases) as input
            for bases, observables in zip(bases_list, observables_list):
                self.basis_dict_list.append(
                    _find_measure_basis_to_observable_mapping(observables, bases)
                )

        exp_vals_list = []
        exp_vars_list = []
        for result_index, result in enumerate(results):
            measurement_flips = result["measurement_flips.meas"]
            meas = result["meas"]
            basis_mapping = self.basis_dict_list[result_index]
            trex_factors_per_basis = trex_factors(self.noise, basis_mapping)

            pub = self.pubs[result_index]
            avg_axes = 1
            if len(pub) > 2:
                parameter_values = pub[2]
                avg_axes = (1,) + (1,) * len(parameter_values.shape[:-1])

            res = executor_expectation_values(
                meas,
                basis_mapping,
                meas_basis_axis=0,
                avg_axis=avg_axes,
                measurement_flips=measurement_flips,
                rescale_factors=trex_factors_per_basis,
            )
            res = np.array(res)
            exp_vals, exp_vars = res[:, 0], res[:, 1]
            exp_vals_list.append(exp_vals)
            exp_vars_list.append(exp_vars)
        return exp_vals_list, exp_vars_list
