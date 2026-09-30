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

"""Tests for coloring utilities."""

import unittest

from qiskit.quantum_info import Pauli, SparseObservable, SparsePauliOp
from qiskit_addon_utils.exp_vals import map_observable_isa_to_canonical


class TestObservableMapping(unittest.TestCase):
    def test_isa_to_canonical(self):
        with self.subTest("Pauli"):
            obs = Pauli("iIIIIZZIIII")
            canonical_qubits = [3, 4, 6, 5]
            mapped_obs = map_observable_isa_to_canonical(obs, canonical_qubits)
            target_obs = Pauli("iZIZI")
            self.assertEqual(target_obs, mapped_obs)
            self.assertEqual(target_obs.phase, mapped_obs.phase)
        with self.subTest("SparsePauliOp"):
            obs = SparsePauliOp(["iIZIZIZIZIZ", "iZIZIZIZIZI"], [2.0, 4.0])
            canonical_qubits = [1, 3, 5, 7, 9, 0, 2, 4, 6, 8]
            mapped_obs = map_observable_isa_to_canonical(obs, canonical_qubits)
            target_obs = SparsePauliOp(["iZZZZZIIIII", "iIIIIIZZZZZ"], [2.0, 4.0])
            self.assertEqual(target_obs, mapped_obs)
        with self.subTest("SparseObservable"):
            obs = SparseObservable.from_list([("IZIZIZIZIZ", 2.0j), ("ZIZIZIZIZI", 4.0j)])
            canonical_qubits = [1, 3, 5, 7, 9, 0, 2, 4, 6, 8]
            mapped_obs = map_observable_isa_to_canonical(obs, canonical_qubits)
            target_obs = SparseObservable.from_list([("ZZZZZIIIII", 2.0j), ("IIIIIZZZZZ", 4.0j)])
            self.assertEqual(target_obs, mapped_obs)

    def test_virtual_to_canonical(self):
        with self.subTest("Pauli"):
            obs = Pauli("iIIIIZZIIII")
            canonical_qubits = [3, 4, 6, 5]
            mapped_obs = map_observable_isa_to_canonical(obs, canonical_qubits)
            target_obs = Pauli("iZIZI")
            self.assertEqual(target_obs, mapped_obs)
            self.assertEqual(target_obs.phase, mapped_obs.phase)
        with self.subTest("SparsePauliOp"):
            obs = SparsePauliOp(["iIZIZIZIZIZ", "iZIZIZIZIZI"], [2.0, 4.0])
            canonical_qubits = [1, 3, 5, 7, 9, 0, 2, 4, 6, 8]
            mapped_obs = map_observable_isa_to_canonical(obs, canonical_qubits)
            target_obs = SparsePauliOp(["iZZZZZIIIII", "iIIIIIZZZZZ"], [2.0, 4.0])
            self.assertEqual(target_obs, mapped_obs)
        with self.subTest("SparseObservable"):
            obs = SparseObservable.from_list([("IZIZIZIZIZ", 2.0j), ("ZIZIZIZIZI", 4.0j)])
            canonical_qubits = [1, 3, 5, 7, 9, 0, 2, 4, 6, 8]
            mapped_obs = map_observable_isa_to_canonical(obs, canonical_qubits)
            target_obs = SparseObservable.from_list([("ZZZZZIIIII", 2.0j), ("IIIIIZZZZZ", 4.0j)])
            self.assertEqual(target_obs, mapped_obs)
