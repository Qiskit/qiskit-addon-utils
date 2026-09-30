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

from typing import Optional

import numpy as np
from qiskit import ClassicalRegister
from qiskit.circuit import QuantumCircuit
from qiskit.primitives.containers.estimator_pub import ObservablesArray
from qiskit.quantum_info import PauliLindbladMap
from qiskit.transpiler import generate_preset_pass_manager
from samplomatic import build
from samplomatic.transpiler import generate_boxing_pass_manager

from qiskit_addon_utils.exp_vals import get_measurement_bases
from qiskit_addon_utils.exp_vals.expectation_values import (
    _find_measure_basis_to_observable_mapping,
    executor_expectation_values,
)
from qiskit_addon_utils.noise_management.error_mitigation.base_mitigator import BaseMitigator
from qiskit_addon_utils.noise_management.trex_factors import trex_factors

from .execution_inputs import ExecutionInputs, InputsLike
from .executor_quantum_program import ExecutorQuantumProgram
from .executor_quantum_program_result import ExecutorQuantumProgramResult


class TREX(BaseMitigator):
    """A class implementing the Twirled Readout Error eXtinction (TREX) error mitigation method.

    TREX is a model-free, computationally efficient quantum error mitigation technique used to mitigate State Preparation and Measurement (SPAM) errors for expectation value calculations.
    The class utilizes Samplomatic package to add readout twirling to the given circuit, that is needed to mitigate the errors using the TREX method.
    The class also handles the basis changes required to compute the given set of observables.
    """

    def __init__(
        self,
        inputs: Optional[list[InputsLike]] = None,
        noise: PauliLindbladMap | None = None,
        boxing_options: dict | None = None,
        twirl_mcm: bool = False,
        shots_per_randomization: int = 64,
        num_randomizations: int = 128,
        cal_randomizations: int = 128,
        parameters_outer_product: bool = True,
    ):
        """Implementation of Twirled Readout Error eXtinction (TREX) method.

        Args:
            inputs: List of ``InputsLike`` objects. Each InputsLike is a tuple in the form of (circuit, list of observables, list of circuit parameters)
            noise: Readout noise learned, in the form of PauliLindbladMap, to be used for the TREX factor post-processing calculation.
            boxing_options: Options used for boxing the circuit using samplomatic ``generate_boxing_pass_manager`` transpiler pass.
            twirl_mcm: If ``True``, twirl also layers of mid-circuit measurements.
            shots_per_randomization: Number of shots for each randomization for the observables execution.
            num_randomizations: Number of randomizations for twirling in the observables execution.
            cal_randomizations: Number of randomizations for twirling in the noise learning execution.
            parameters_outer_product: If ``True``, parameters binding variable must be an array with no empty shape. The shape of the observables and parameters will be broadcasted to create an outer-product calculation.
        """
        super().__init__(
            inputs=inputs,
            noise=noise,
            boxing_options=boxing_options,
            twirl_mcm=twirl_mcm,
            shots_per_randomization=shots_per_randomization,
            num_randomizations=num_randomizations,
            parameters_outer_product=parameters_outer_product,
        )
        self.options.update(
            {
                "cal_randomizations": cal_randomizations,
            }
        )

    def _annotate_circuit_and_find_layers(self, circuit):
        """Annotate the final measurement layer with ChangeBasis and Twirling annotations.

        Remove terminal measurements in the circuit if present, and replace with new terminal measurements on all the circuit's qubits,
        writing the results to a new dedicated classical register named `_trex_data`.

        Args:
            circuit: The circuit whose layers to annotate.

        Returns:
            A tuple of a circuit with annotations removed from every box, besides the final measurement box that contains Twirling and ChangeBasis annotations,
            and the final measurement layer of the circuit as a list of CircuitInstructions.

        Raises:
            ValueError: If the circuit contains classical register named `_trex_data`.
        """
        edited_circuit, contain_mcm = self._create_final_measurement_layer(circuit, "_trex_data")
        # Required parameters that will overwrite user input
        boxing_params = {
            "enable_measures": True,
            "measure_annotations": "all",
        }
        updated_boxing_params = self.boxing_options.copy()
        updated_boxing_params.update(boxing_params)
        updated_boxing_params = {k: v for k, v in updated_boxing_params.items() if v is not None}

        boxing_pm = generate_boxing_pass_manager(**updated_boxing_params)

        annotated_circuit = boxing_pm.run(edited_circuit)
        if contain_mcm:
            annotated_circuit = self._remove_midcircuit_box_annotations(annotated_circuit)
        return annotated_circuit, annotated_circuit.data[-1]

    def annotate_circuits_and_find_layers(self):
        """Annotate the input circuits and find the combined measurement layer that is required for readout noise learning for all inputs.

        Split the circuits in ``self.inputs`` into layers, box and annotate terminal measurement layer with `ChangeBasis` and `Twirl` annotations,
        and annotate mid-circuit measurement layer with `Twirl` annotations if the option of ``twirl_mcm`` is True.
        Use the twirling options in ``self.options`` in the annotations.
        Extract the unique unified circuit layer relevant for noise learning based on the layers found for each circuit.

        Returns:
            Circuit layer containing the terminal measurements required for noise learning for all inputs combined.
        """
        noise_learning_layers = []
        annotated_circuits = []
        for execution_input in self.inputs:
            annotated_circuit, noise_learning_layer = self._annotate_circuit_and_find_layers(
                execution_input.circuit
            )
            noise_learning_layers.append(noise_learning_layer)
            annotated_circuits.append(annotated_circuit)
        self.annotated_circuits = annotated_circuits

        # create the combined noise learning layer of all given inputs
        measured_qubits = set()
        for learning_layer in noise_learning_layers:
            circuit_measured_qubits = [qubit._index for qubit in learning_layer.qubits]
            measured_qubits.update(circuit_measured_qubits)
        qubit_list = list(range(len(measured_qubits)))

        classical_cal_reg = ClassicalRegister(len(measured_qubits), name="_trex_cal")
        trex_circuit = QuantumCircuit(len(measured_qubits))
        trex_circuit.add_register(classical_cal_reg)
        trex_circuit.measure(qubit_list, classical_cal_reg)
        trex_isa_pm = generate_preset_pass_manager(
            initial_layout=measured_qubits, optimization_level=0
        )
        trex_circuit = trex_isa_pm.run(trex_circuit)
        self.noise_learning_layer = trex_circuit

        return trex_circuit

    def _create_trex_calibration_circuit(self):
        """Creates a TREX calibration circuit based on all circuit layers in ``self.noise_learning_layer``.

        Returns:
            Annotated calibration circuit for TREX factors calculation.
        """
        if not self.noise_learning_layer:
            print("No noise learning layer is set, returns None")
            return None
        boxing_pm = generate_boxing_pass_manager(
            enable_gates=False,
            enable_measures=True,
            measure_annotations="twirl",
        )
        annotated_trex_circuit = boxing_pm.run(self.noise_learning_layer)
        return annotated_trex_circuit

    def prepare(self) -> ExecutorQuantumProgram:
        """Create a QuantumProgram that contains a relevant samplex item for each input in ``self.inputs``.

        If ``annotated_circuits`` or ``noise_learning_layer`` is not set, find and annotate circuit layers.
        For each annotated circuit and relevant observables array, find a measurement basis set that enables calculating expectation values for all relevant observables.
        If a noise model is not set, add as the last item to the QuantumProgram also a calibration item for learning the readout noise for all the qubits that participate in at least one of the circuits.

        Returns:
            ExecutorQuantumProgram that contains a relevant samplex item for each input in ``self.inputs`` and an item for noise learning if a noise model is not set.

        Raises:
            * ValueError: If ``annotated_circuits`` and ``noise_learning_layer`` were set manually and one of the circuits does not contain a register named `_trex_data`.
            * ValueError: If one of the input circuits contains a register named `_trex_data`.
        """
        if self.inputs is None:
            return ExecutorQuantumProgram()
        if not self.annotated_circuits or not self.noise_learning_layer:
            self.annotate_circuits_and_find_layers()
        if self.annotated_circuits is None:
            return ExecutorQuantumProgram()

        # create ExecutorQuantumProgram
        if isinstance(self.options["shots_per_randomization"], int):
            program = ExecutorQuantumProgram(shots=self.options["shots_per_randomization"])
        else:
            raise ValueError("Shots_per_randomization must be an integer.")
        for index, execution_input in enumerate(self.inputs):
            annotated_circuit = self.annotated_circuits[index]
            sparse_observables = ExecutionInputs.observables_array_to_1d_sparse_obs(
                execution_input.observables
            )
            (measure_bases, measure_bases_str), basis_dict = get_measurement_bases(
                sparse_observables, bases_format="both"
            )
            measure_bases_str = [
                str(basis) for basis in measure_bases_str
            ]  # force the type of the returned bases
            self.measure_bases_list.append(measure_bases_str)
            self.basis_dict_list.append(basis_dict)
            self.observables_list.append(execution_input.observables)

            num_qubits = None
            for register in annotated_circuit.cregs:
                if register.name == "_trex_data":
                    num_qubits = len(register)
            if not num_qubits:
                raise ValueError(
                    f"The circuit in input number {index} does not contain a register named `_trex_data`."
                )
            # broadcast measurement basis shape
            if isinstance(self.options["num_randomizations"], int):
                num_randomizations = self.options["num_randomizations"]
            else:
                raise ValueError("num_randomizations must be an integer.")
            if execution_input.parameters is not None:
                parameter_values = execution_input.parameters
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
                    + (num_randomizations,)
                    + (1,) * len(parameter_values.shape[:-1])
                )
            else:
                # add dimension also for the twirling randomizations
                bases_shape = (len(measure_bases), 1, num_qubits)
                measure_bases_broadcastable = np.array(measure_bases).reshape(bases_shape)
                samplex_shape = (len(measure_bases), num_randomizations)

            template_circuit, samplex = build(annotated_circuit)
            # Generate `samplex_arguments` for the executor
            samplex_arguments = samplex.inputs().make_broadcastable()
            basis_changes_name = samplex.inputs().get_specs("basis_changes")[0].name
            if execution_input.parameters is not None:
                samplex_arguments.bind(
                    **{
                        "parameter_values": execution_input.parameters,
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
            calibration_circuit = self._create_trex_calibration_circuit()

            template_calibration_circuit, calibration_samplex = build(calibration_circuit)
            if isinstance(self.options["cal_randomizations"], int):
                cal_randomizations = self.options["cal_randomizations"]
            else:
                raise ValueError("cal_randomizations must be an integer.")
            program.append_samplex_item(
                template_calibration_circuit,
                samplex=calibration_samplex,
                shape=(cal_randomizations,),
            )
        # save data in the program for post processing
        program.passthrough_data = {
            "_trex": {
                "observables": self.observables_list,
                "measure_bases": self.measure_bases_list,
            }
        }
        return program

    def post_process(self, results: ExecutorQuantumProgramResult):
        """Calculates the TREX mitigated expectation values for all given observables, with all given parameters, in each input.

        Args:
            results: Result object from an Executor execution of a QuantumProgram created by a TREX class.
            If the results are loaded from the cloud, the needed internal variables are loaded from the result's ``passthrough_data`` parameter.

        Returns:
            A tuple of ndarray of all the expectation values and ndarray of all standard deviations for all given observables, for each parameters set, for each input.
        """
        data_results = results._data
        if not self.noise:
            # assume a calibration circuit was added to the quantum program as the last item
            noise_learning_result = results[-1]
            measurement_flips = noise_learning_result["measurement_flips._trex_cal"]
            noise_calibration_data = noise_learning_result["_trex_cal"]
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
            data_results = results._data[:-1]

        if not self.basis_dict_list:
            # map the observable to commuting bases, so the trex factors and the expectation values are calculated using the same mapping
            if results.passthrough_data:
                observables_list = results.passthrough_data.get("_trex", {}).get("observables")
                observables = [
                    ObservablesArray.coerce(observables) for observables in observables_list
                ]
                measure_bases = results.passthrough_data.get("_trex", {}).get("measure_bases")
                if observables is None or measure_bases is None:
                    raise ValueError(
                        "The result must contain observables and measure_bases in the _trex key of the passthrough_data."
                    )
            else:
                raise ValueError("The result must contain passthrough_data.")

            for bases, input_observables in zip(measure_bases, observables):
                sparse_observables = ExecutionInputs.observables_array_to_1d_sparse_obs(
                    input_observables
                )
                self.basis_dict_list.append(
                    _find_measure_basis_to_observable_mapping(sparse_observables, bases)
                )

        exp_vals_list = []
        exp_vars_list = []
        for result, basis_mapping in zip(data_results, self.basis_dict_list):
            if "_trex_data" in result and "measurement_flips._trex_data" in result:
                measurement_flips = result["measurement_flips._trex_data"]
                meas = result["_trex_data"]
            else:
                raise ValueError(
                    "The result must contain `_trex_data` and `measurement_flips._trex_data` keys."
                )
            trex_factors_per_basis = trex_factors(self.noise, basis_mapping)

            # The prepare function places meas_basis in axis 0, even for cases with only a single basis
            avg_axes = 1
            meas_basis_axis = 0

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
