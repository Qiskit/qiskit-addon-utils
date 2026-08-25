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

"""General quantum error mitigation method."""

from abc import ABC, abstractmethod
from typing import Optional

from qiskit import ClassicalRegister
from qiskit.circuit import CircuitError, CircuitInstruction, QuantumCircuit
from qiskit.circuit.controlflow.box import BoxOp
from qiskit.primitives.containers.estimator_pub import ObservablesArray
from qiskit.quantum_info import Pauli, PauliLindbladMap, SparsePauliOp
from samplomatic.annotations import Twirl

from .execution_inputs import ExecutionInputs, InputsLike
from .executor_quantum_program import ExecutorQuantumProgram
from .executor_quantum_program_result import ExecutorQuantumProgramResult


class BaseMitigator(ABC):
    """An abstract class of a general error mitigation method."""

    def __init__(
        self,
        inputs: Optional[list[InputsLike]] = None,
        noise: PauliLindbladMap | None = None,
        boxing_options: dict | None = None,
        twirl_mcm: bool = False,
        shots_per_randomization: int = 64,
        num_randomizations: int = 128,
        parameters_outer_product: bool = True,
    ):
        """Implementation of general error mitigation method.

        Args:
            inputs: List of ``InputsLike`` objects. Each InputsLike is a tuple in the form of (circuit, list of observables, list of circuit parameters)
            noise: Noise learned, in the form of PauliLindbladMap.
            boxing_options: Options used for boxing the circuit using samplomatic ``generate_boxing_pass_manager`` transpiler pass.
            twirl_mcm: If ``True``, twirl also layers of mid-circuit measurements.
            shots_per_randomization: Number of shots for each randomization for the observables execution.
            num_randomizations: Number of randomizations for twirling in the observables execution.
            parameters_outer_product: If ``True``, parameters binding variable must be an array with no empty shape. The shape of the observables and parameters will be broadcasted to create an outer-product calculation.
        """
        self.inputs: list[ExecutionInputs] | None = None
        if inputs is not None:
            self.inputs = [
                ExecutionInputs.coerce(execution_input, parameters_outer_product)
                for execution_input in inputs
            ]
        self.boxing_options: dict = boxing_options if boxing_options is not None else {}
        self.options = {
            "twirl_mcm": twirl_mcm,
            "shots_per_randomization": shots_per_randomization,
            "num_randomizations": num_randomizations,
        }

        self.noise_learning_layer: QuantumCircuit | None = None
        self.annotated_circuits: list[QuantumCircuit] | None = None
        self.basis_dict_list: list[dict[Pauli, list[SparsePauliOp | None]]] = []
        self.measure_bases_list: list[list[str]] = []
        self.observables_list: list[ObservablesArray] = []
        self.noise: PauliLindbladMap = noise
        self.parameters_outer_product: bool = parameters_outer_product

    def _remove_midcircuit_box_annotations(self, circuit: QuantumCircuit):
        """Return a new circuit with all annotations removed from every measurement box, besides the final box.

        If the option of twirl_mcm is set to ``True``, keep Twirl annotation also for layers with mid-circuit measurements.

        Args:
            circuit: The circuit whose box annotations to strip.

        Returns:
            A new circuit with empty annotation lists on every measurement box, besides the final measurement box.
        """
        new_circuit = circuit.copy_empty_like()
        for instr in circuit.data[:-1]:
            if instr.operation.name == "box":
                box = instr.operation
                if box.num_clbits > 0:  # box containing measurement
                    if self.options["twirl_mcm"]:
                        # keep only the twirling annotation
                        new_box = BoxOp(
                            body=box.body,
                            label=box.label,
                            annotations=[
                                annotation
                                for annotation in box.annotations
                                if isinstance(annotation, Twirl)
                            ],
                        )
                    else:
                        new_box = BoxOp(body=box.body, label=box.label, annotations=[])
                else:
                    new_box = box
                new_circuit.data.append(CircuitInstruction(new_box, instr.qubits, instr.clbits))
            else:
                new_circuit.data.append(instr)
        new_circuit.data.append(circuit.data[-1])
        return new_circuit

    @staticmethod
    def _create_final_measurement_layer(circuit, classical_reg_name):
        """Annotate the final measurement layer with ChangeBasis and Twirling annotations.

        Remove terminal measurements in the circuit if present, and replace with new terminal measurements on all the circuit's qubits,
        writing the results to a new dedicated classical register named as ``classical_reg_name`` variable.

        Args:
            circuit: The circuit whose layers to annotate.
            classical_reg_name: The classical register name of the final measurement layer.

        Returns:
            A tuple of a circuit with the final measurement replaced with a new measurement layer with dedicated classical register named ``classical_reg_name``,
            and a boolean indicating whether there are measurements in the circuit besides the final measurement layer that was added.

        Raises:
            ValueError: If the circuit contains classical register named as ``classical_reg_name``.
        """
        if circuit.layout:
            qubits_in_layout = circuit.layout.final_index_layout()
        else:
            qubits_in_layout = list(range(len(circuit.qubits)))
        edited_circuit = circuit.remove_final_measurements(inplace=False)
        contain_mcm = "measure" in edited_circuit.count_ops()
        data_register = ClassicalRegister(len(qubits_in_layout), classical_reg_name)
        try:
            edited_circuit.add_register(data_register)
        except CircuitError as err:
            raise ValueError(
                f"Register name `{classical_reg_name}` is reserved for a dedicated classical register used by this class."
            ) from err
        edited_circuit.barrier(qubits_in_layout)
        edited_circuit.measure(qubits_in_layout, data_register)
        return edited_circuit, contain_mcm

    @abstractmethod
    def annotate_circuits_and_find_layers(self):
        """Annotate the input circuits and find the combined measurement layer that is required for noise learning.

        Split the circuits in ``self.inputs`` into layers, box and annotate layers.
        Extract the unique circuit layers relevant for noise learning based on the layers found for each circuit.

        Returns:
            Circuit layers relevant for noise learning.
        """

    @abstractmethod
    def prepare(self) -> ExecutorQuantumProgram:
        """Create a QuantumProgram that contains a relevant samplex item for each input in ``self.inputs``.

        Returns:
        ExecutorQuantumProgram that contains a relevant samplex item for each input in ``self.inputs`` and an item for noise learning if a noise model is not set.
        """

    @abstractmethod
    def post_process(self, results: ExecutorQuantumProgramResult):
        """Calculates the mitigated expectation values for all given observables, with all given parameters, in each input.

        Args:
            results: Result object from an Executor execution of a QuantumProgram created by the ``prepare`` function of this class.
            If the results are loaded from the cloud, the needed internal variables are loaded from the result's ``passthrough_data`` parameter.

        Returns:
            A tuple of ndarray of all the expectation values and ndarray of all standard deviations for all given observables, for each parameters set, for each input.
        """
