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

from qiskit.circuit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import PauliList, SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_addon_utils.exp_vals.expectation_values import executor_expectation_values
from qiskit_addon_utils.exp_vals.measurement_bases import _convert_basis_to_uint_representation
from qiskit_addon_utils.exp_vals.observable_mappings import map_observable_virtual_to_canonical
from samplomatic.annotations import ChangeBasis
from samplomatic.builders import build
from samplomatic.transpiler import generate_boxing_pass_manager
from samplomatic.utils import get_annotation


def create_and_sample_circ(layout, paulis, add_x, gate_twirl, meas_twirl):
    backend = GenericBackendV2(num_qubits=5)
    observable = SparsePauliOp("".join(paulis[::-1]), 1.0)

    num_qubits = len(paulis)
    qc = QuantumCircuit(num_qubits)
    for q, P in enumerate(paulis):
        if add_x[q]:
            qc.x(q)
        if P == 'X':
            qc.h(q)
        elif P == 'Y':
            qc.h(q)
            qc.s(q)
    qc.measure_all()

    isa_pm = generate_preset_pass_manager(backend=backend, initial_layout=layout, optimization_level=0)
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
    canonical_qubits = [qubit._index for qubit in boxed_circuit.qubits if qubit in measurement_box.qubits]
    observable_executor_cannon = map_observable_virtual_to_canonical(observable, layout, canonical_qubits)
    measurement_bases = _convert_basis_to_uint_representation(PauliList(observable_executor_cannon.paulis))

    # build the samplex and sample some parameters
    template, samplex = build(boxed_circuit)
    inputs = samplex.inputs()
    samplex_input_inputs = {
        "basis_changes": {get_annotation(boxed_circuit[-1].operation, ChangeBasis).ref: measurement_bases[0]}}
    samplex_arguments = inputs.bind(**samplex_input_inputs).make_broadcastable()
    outputs = samplex.sample(samplex_arguments, num_randomizations=n_rand)

    sam = StatevectorSampler()

    job = sam.run([(template, outputs["parameter_values"])], shots=10_000)
    res = job.result()

    bool_array = res[0].data.meas.to_bool_array('little')
    meas_flips = outputs["measurement_flips.meas"] if meas_twirl else None
    if n_rand == 1:
        bool_array = bool_array[0]
        avg_axis = None
    else:
        avg_axis = 0

    return executor_expectation_values(
        bool_array, reverser, meas_basis_axis=None, avg_axis=avg_axis, measurement_flips=meas_flips
    )

class TestExpectationValues(unittest.TestCase):
    def test_executor_expectation_values_no_twirls(self):
        with self.subTest("Check exp val"):
            exp_vals = create_and_sample_circ(layout = [1, 2], paulis = ['X', 'I'], add_x=[False,False],
                                              gate_twirl=False, meas_twirl=False)
            self.assertEqual(len(exp_vals), 1)
            # exp val equals 1
            self.assertAlmostEqual(exp_vals[0][0], 1)
            # var equals 0
            self.assertAlmostEqual(exp_vals[0][1], 0)
        with self.subTest("Check exp val reversed layout"):
            exp_vals = create_and_sample_circ(layout = [4, 1], paulis = ['X', 'I'], add_x=[False,False],
                                              gate_twirl=False, meas_twirl=False)
            self.assertEqual(len(exp_vals), 1)
            # exp val equals 1
            self.assertAlmostEqual(exp_vals[0][0], 1)
            # var equals 0
            self.assertAlmostEqual(exp_vals[0][1], 0)
        with self.subTest("Check exp val with x gate"):
            exp_vals = create_and_sample_circ(layout = [1, 2], paulis = ['X', 'I'], add_x=[True,False],
                                              gate_twirl=False, meas_twirl=False)
            self.assertEqual(len(exp_vals), 1)
            # exp val equals -1
            self.assertAlmostEqual(exp_vals[0][0], -1)
            # var equals 0
            self.assertAlmostEqual(exp_vals[0][1], 0)
        with self.subTest("Check exp val with x gate reversed layout"):
            exp_vals = create_and_sample_circ(layout = [4, 1], paulis = ['X', 'I'], add_x=[True,False],
                                              gate_twirl=False, meas_twirl=False)
            self.assertEqual(len(exp_vals), 1)
            # exp val equals -1
            self.assertAlmostEqual(exp_vals[0][0], -1)
            # var equals 0
            self.assertAlmostEqual(exp_vals[0][1], 0)
        with self.subTest("Check exp val with x gate reversed layout extended basis"):
            exp_vals = create_and_sample_circ(layout = [4, 2, 0], paulis = ['X', 'Z', 'Y'], add_x=[True,False, False],
                                              gate_twirl=False, meas_twirl=False)
            self.assertEqual(len(exp_vals), 1)
            # exp val equals -1
            self.assertAlmostEqual(exp_vals[0][0], -1)
            # var equals 0
            self.assertAlmostEqual(exp_vals[0][1], 0)

    def test_executor_expectation_values_gate_twirls(self):
        with self.subTest("Check exp val with gate twirling"):
            exp_vals = create_and_sample_circ(layout=[1, 2], paulis=['X', 'I'], add_x=[False, False], gate_twirl=True,
                                   meas_twirl=False)
            self.assertEqual(len(exp_vals), 1)
            # exp val equals 1
            self.assertAlmostEqual(exp_vals[0][0], 1)
            # var equals 0
            self.assertAlmostEqual(exp_vals[0][1], 0)
        with self.subTest("Check exp val with gate twirling reversed layout"):
            exp_vals = create_and_sample_circ(layout=[4, 1], paulis=['X', 'I'], add_x=[False, False], gate_twirl=True,
                                   meas_twirl=False)
            self.assertEqual(len(exp_vals), 1)
            # exp val equals 1
            self.assertAlmostEqual(exp_vals[0][0], 1)
            # var equals 0
            self.assertAlmostEqual(exp_vals[0][1], 0)
        with self.subTest("Check exp val with gate twirling with x gate"):
            exp_vals = create_and_sample_circ(layout=[1, 2], paulis=['X', 'I'], add_x=[True, False], gate_twirl=True,
                                   meas_twirl=False)
            self.assertEqual(len(exp_vals), 1)
            # exp val equals -1
            self.assertAlmostEqual(exp_vals[0][0], -1)
            # var equals 0
            self.assertAlmostEqual(exp_vals[0][1], 0)
        with self.subTest("Check exp val with gate twirling and x gate reversed layout"):
            exp_vals = create_and_sample_circ(layout=[4, 1], paulis=['X', 'I'], add_x=[True, False], gate_twirl=True,
                                   meas_twirl=False)
            self.assertEqual(len(exp_vals), 1)
            # exp val equals -1
            self.assertAlmostEqual(exp_vals[0][0], -1)
            # var equals 0
            self.assertAlmostEqual(exp_vals[0][1], 0)
        with self.subTest("Check exp val with gate twirling and x gate reversed layout extended basis"):
            exp_vals = create_and_sample_circ(layout = [4, 2, 0], paulis = ['X', 'Z', 'Y'], add_x=[True,False, False],
                                              gate_twirl=True, meas_twirl=False)
            self.assertEqual(len(exp_vals), 1)
            # exp val equals -1
            self.assertAlmostEqual(exp_vals[0][0], -1)
            # var equals 0
            self.assertAlmostEqual(exp_vals[0][1], 0)

    def test_executor_expectation_values_meas_twirls(self):
        with self.subTest("Check exp val with measurement twirling"):
            exp_vals = create_and_sample_circ(layout=[1, 2], paulis=['X', 'I'], add_x=[False, False], gate_twirl=True,
                                   meas_twirl=True)
            self.assertEqual(len(exp_vals), 1)
            # exp val equals 1
            self.assertAlmostEqual(exp_vals[0][0], 1)
            # var equals 0
            self.assertAlmostEqual(exp_vals[0][1], 0)
        with self.subTest("Check exp val with measurement twirling reversed layout"):
            exp_vals = create_and_sample_circ(layout=[4, 1], paulis=['X', 'I'], add_x=[False, False], gate_twirl=False,
                                   meas_twirl=True)
            self.assertEqual(len(exp_vals), 1)
            # exp val equals 1
            self.assertAlmostEqual(exp_vals[0][0], 1)
            # var equals 0
            self.assertAlmostEqual(exp_vals[0][1], 0)
        with self.subTest("Check exp val with measurement twirling with x gate"):
            exp_vals = create_and_sample_circ(layout=[1, 2], paulis=['X', 'I'], add_x=[True, False], gate_twirl=False,
                                   meas_twirl=True)
            self.assertEqual(len(exp_vals), 1)
            # exp val equals -1
            self.assertAlmostEqual(exp_vals[0][0], -1)
            # var equals 0
            self.assertAlmostEqual(exp_vals[0][1], 0)
        with self.subTest("Check exp val with measurement twirling and x gate reversed layout"):
            exp_vals = create_and_sample_circ(layout=[4, 1], paulis=['X', 'I'], add_x=[True, False], gate_twirl=False,
                                   meas_twirl=True)
            self.assertEqual(len(exp_vals), 1)
            # exp val equals -1
            self.assertAlmostEqual(exp_vals[0][0], -1)
            # var equals 0
            self.assertAlmostEqual(exp_vals[0][1], 0)
        with self.subTest("Check exp val with measurement twirling and x gate reversed layout extended basis"):
            exp_vals = create_and_sample_circ(layout = [4, 2, 0], paulis = ['X', 'Z', 'Y'], add_x=[True,False, False],
                                              gate_twirl=False, meas_twirl=True)
            self.assertEqual(len(exp_vals), 1)
            # exp val equals -1
            self.assertAlmostEqual(exp_vals[0][0], -1)
            # var equals 0
            self.assertAlmostEqual(exp_vals[0][1], 0)

    def test_executor_expectation_values_gate_and_meas_twirls(self):
        with self.subTest("Check exp val with gate and measurement twirling"):
            exp_vals = create_and_sample_circ(layout=[1, 2], paulis=['X', 'I'], add_x=[False, False], gate_twirl=True,
                                   meas_twirl=True)
            self.assertEqual(len(exp_vals), 1)
            # exp val equals 1
            self.assertAlmostEqual(exp_vals[0][0], 1)
            # var equals 0
            self.assertAlmostEqual(exp_vals[0][1], 0)
        with self.subTest("Check exp val with gate and measurement twirling reversed layout"):
            exp_vals = create_and_sample_circ(layout=[4, 1], paulis=['X', 'I'], add_x=[False, False], gate_twirl=True,
                                   meas_twirl=True)
            self.assertEqual(len(exp_vals), 1)
            # exp val equals 1
            self.assertAlmostEqual(exp_vals[0][0], 1)
            # var equals 0
            self.assertAlmostEqual(exp_vals[0][1], 0)
        with self.subTest("Check exp val with gate and measurement twirling with x gate"):
            exp_vals = create_and_sample_circ(layout=[1, 2], paulis=['X', 'I'], add_x=[True, False], gate_twirl=True,
                                   meas_twirl=True)
            self.assertEqual(len(exp_vals), 1)
            # exp val equals -1
            self.assertAlmostEqual(exp_vals[0][0], -1)
            # var equals 0
            self.assertAlmostEqual(exp_vals[0][1], 0)
        with self.subTest("Check exp val with gate and measurement twirling and x gate reversed layout"):
            exp_vals = create_and_sample_circ(layout=[4, 1], paulis=['X', 'I'], add_x=[True, False], gate_twirl=True,
                                   meas_twirl=True)
            self.assertEqual(len(exp_vals), 1)
            # exp val equals -1
            self.assertAlmostEqual(exp_vals[0][0], -1)
            # var equals 0
            self.assertAlmostEqual(exp_vals[0][1], 0)
        with self.subTest("Check exp val with gate and measurement twirling and x gate reversed layout extended basis"):
            exp_vals = create_and_sample_circ(layout = [4, 2, 0], paulis = ['X', 'Z', 'Y'], add_x=[True,False, False],
                                              gate_twirl=True, meas_twirl=True)
            self.assertEqual(len(exp_vals), 1)
            # exp val equals -1
            self.assertAlmostEqual(exp_vals[0][0], -1)
            # var equals 0
            self.assertAlmostEqual(exp_vals[0][1], 0)
