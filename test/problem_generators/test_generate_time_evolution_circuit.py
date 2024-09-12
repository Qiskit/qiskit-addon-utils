# This code is a Qiskit project.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the time evolution circuit generator."""

import unittest

from qiskit.circuit.library import RXXGate, RYYGate, RZZGate
from qiskit.transpiler import CouplingMap
from qiskit_addon_utils.problem_generators import (
    generate_time_evolution_circuit,
    generate_xyz_hamiltonian,
)


class TestTimeEvolutionCircuit(unittest.TestCase):
    def test_generate_time_evolution_circuit(self):
        with self.subTest("Basic test"):
            lattice = CouplingMap.from_heavy_hex(3).reduce([7, 18, 8])
            ham = generate_xyz_hamiltonian(lattice)
            qc = generate_time_evolution_circuit(ham)
            self.assertEqual(6, len(qc.data))
            self.assertEqual(RXXGate(2.0), qc.data[0].operation)
            self.assertEqual(RZZGate(2.0), qc.data[-1].operation)
            self.assertEqual(RYYGate(2.0), qc.data[4].operation)
