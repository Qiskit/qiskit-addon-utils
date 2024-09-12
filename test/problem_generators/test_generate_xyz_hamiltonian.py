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

"""Tests for the "XYZ model"-type Hamiltonian generator."""

import unittest

import pytest
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import CouplingMap
from qiskit_addon_utils.problem_generators import (
    PauliOrderStrategy,
    generate_xyz_hamiltonian,
)


class TestProblemGeneration(unittest.TestCase):
    def test_problem_generation(self):
        with self.subTest("Basic test"):
            target_obs = SparsePauliOp(
                ["IXX", "IYY", "IZZ", "XXI", "YYI", "ZZI"],
                coeffs=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            )
            lattice = CouplingMap.from_heavy_hex(3, bidirectional=False).reduce([7, 18, 8])
            ham = generate_xyz_hamiltonian(lattice)
            self.assertEqual(target_obs, ham)
        with self.subTest("Basic test w PyDiGraph and parallel edges"):
            target_obs = SparsePauliOp(
                ["IXX", "IYY", "IZZ", "XXI", "YYI", "ZZI"],
                coeffs=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            )
            lattice = CouplingMap.from_heavy_hex(3).reduce([7, 18, 8]).graph
            ham = generate_xyz_hamiltonian(lattice)
            self.assertEqual(target_obs, ham)
        with self.subTest("Basic test w PyGraph and parallel edges"):
            target_obs = SparsePauliOp(
                ["IXX", "IYY", "IZZ", "XXI", "YYI", "ZZI"],
                coeffs=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            )
            lattice = CouplingMap.from_heavy_hex(3).reduce([7, 18, 8]).graph.to_undirected()
            ham = generate_xyz_hamiltonian(lattice)
            self.assertEqual(target_obs, ham)
        with self.subTest("Basic test with PauliOrderStrategy.InteractionThenColor"):
            target_obs = SparsePauliOp(
                ["IXX", "XXI", "IYY", "YYI", "IZZ", "ZZI"],
                coeffs=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            )
            lattice = CouplingMap.from_heavy_hex(3, bidirectional=False).reduce([7, 18, 8])
            ham = generate_xyz_hamiltonian(
                lattice, pauli_order_strategy=PauliOrderStrategy.InteractionThenColor
            )
            self.assertEqual(target_obs, ham)
        with self.subTest("Basic test with PauliOrderStrategy.InteractionThenColorZigZag"):
            target_obs = SparsePauliOp(
                ["IXX", "XXI", "YYI", "IYY", "IZZ", "ZZI"],
                coeffs=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            )
            lattice = CouplingMap.from_heavy_hex(3, bidirectional=False).reduce([7, 18, 8])
            ham = generate_xyz_hamiltonian(
                lattice,
                pauli_order_strategy=PauliOrderStrategy.InteractionThenColorZigZag,
            )
            self.assertEqual(target_obs, ham)
        with self.subTest("Basic test with custom coloring"):
            target_obs = SparsePauliOp(
                ["XXI", "YYI", "ZZI", "IXX", "IYY", "IZZ"],
                coeffs=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            )
            lattice = CouplingMap.from_heavy_hex(3, bidirectional=False).reduce([7, 18, 8])
            ham = generate_xyz_hamiltonian(lattice, coloring={(0, 1): 1, (1, 2): 0})
            self.assertEqual(target_obs, ham)
        with self.subTest("Magnetic field terms"):
            target_obs = SparsePauliOp(
                ["IIX", "IIY", "IIZ", "IXI", "IYI", "IZI", "XII", "YII", "ZII"],
                coeffs=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            )
            lattice = CouplingMap.from_heavy_hex(3).reduce([7, 18, 8])
            ham = generate_xyz_hamiltonian(
                lattice, coupling_constants=(0.0, 0.0, 0.0), ext_magnetic_field=(1.0, 1.0, 1.0)
            )
            self.assertEqual(target_obs, ham)
        with self.subTest("Magnetic field terms and PauliOrderStrategy.InteractionThenColor"):
            target_obs = SparsePauliOp(
                ["IIX", "IXI", "XII", "IIY", "IYI", "YII", "IIZ", "IZI", "ZII"],
                coeffs=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            )
            lattice = CouplingMap.from_heavy_hex(3, bidirectional=False).reduce([7, 18, 8])
            ham = generate_xyz_hamiltonian(
                lattice,
                coupling_constants=(0.0, 0.0, 0.0),
                ext_magnetic_field=(1.0, 1.0, 1.0),
                pauli_order_strategy=PauliOrderStrategy.InteractionThenColor,
            )
            self.assertEqual(target_obs, ham)
        with self.subTest("Magnetic field terms and PauliOrderStrategy.InteractionThenColorZigZag"):
            target_obs = SparsePauliOp(
                ["IIX", "IXI", "XII", "IIY", "IYI", "YII", "IIZ", "IZI", "ZII"],
                coeffs=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            )
            lattice = CouplingMap.from_heavy_hex(3, bidirectional=False).reduce([7, 18, 8])
            ham = generate_xyz_hamiltonian(
                lattice,
                coupling_constants=(0.0, 0.0, 0.0),
                ext_magnetic_field=(1.0, 1.0, 1.0),
                pauli_order_strategy=PauliOrderStrategy.InteractionThenColorZigZag,
            )
            self.assertEqual(target_obs, ham)
        with self.subTest("Bad coupling constants"):
            lattice = CouplingMap.from_heavy_hex(3).reduce([7, 18, 8])
            with pytest.raises(ValueError) as e_info:
                generate_xyz_hamiltonian(lattice, coupling_constants=(1.0, 1.0))
            self.assertEqual(
                "Coupling constants must be specified by a length-3 sequence of floating point values.",
                e_info.value.args[0],
            )
        with self.subTest("Bad magnetic field"):
            lattice = CouplingMap.from_heavy_hex(3).reduce([7, 18, 8])
            with pytest.raises(ValueError) as e_info:
                generate_xyz_hamiltonian(lattice, ext_magnetic_field=(1.0, 1.0))
            self.assertEqual(
                "External magnetic field must be specified by a length-3 sequence of floating point values.",
                e_info.value.args[0],
            )
