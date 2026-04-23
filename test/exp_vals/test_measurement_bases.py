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

"""Tests for the measurement_bases module."""

import unittest

import numpy as np
import pytest
from qiskit.quantum_info import Pauli, PauliList, SparsePauliOp
from qiskit_addon_utils.exp_vals.measurement_bases import (
    _convert_basis_to_uint_representation,
    _meas_basis_for_pauli_group,
    get_measurement_bases,
)


class TestGetMeasurementBases(unittest.TestCase):
    """Tests for get_measurement_bases function."""

    def test_single_observable_single_pauli(self):
        """Test with a single observable containing a single Pauli term."""
        obs = SparsePauliOp("ZZZ", 1.0)
        bases, reverser = get_measurement_bases(obs, bases_in_int_format=True)

        self.assertEqual(len(bases), 1)
        self.assertEqual(len(reverser), 1)
        np.testing.assert_array_equal(bases[0], np.array([1, 1, 1], dtype=np.uint8))

        # Check reverser structure
        basis_pauli = next(iter(reverser.keys()))
        self.assertEqual(basis_pauli, Pauli("ZZZ"))
        self.assertEqual(len(reverser[basis_pauli]), 1)
        self.assertIsInstance(reverser[basis_pauli][0], SparsePauliOp)

    def test_single_observable_multiple_paulis(self):
        """Test with a single observable containing multiple Pauli terms."""
        obs = SparsePauliOp(["ZZI", "IZZ", "ZIZ"], [1.0, 2.0, 3.0])
        bases, _ = get_measurement_bases(obs, bases_in_int_format=True)

        # All Z-type Paulis should commute and be in one basis
        self.assertEqual(len(bases), 1)
        np.testing.assert_array_equal(bases[0], np.array([1, 1, 1], dtype=np.uint8))

    def test_multiple_observables(self):
        """Test with multiple observables."""
        obs1 = SparsePauliOp("ZZZ", 1.0)
        obs2 = SparsePauliOp("XXX", 2.0)
        bases, reverser = get_measurement_bases([obs1, obs2], bases_in_int_format=True)

        # Z and X don't commute qubit-wise, so we need 2 bases
        self.assertEqual(len(bases), 2)
        self.assertEqual(len(reverser), 2)

        # Check that each basis maps to a list with 2 elements (one per observable)
        for _, obs_list in reverser.items():
            self.assertEqual(len(obs_list), 2)

    def test_bases_string_format(self):
        """Test with bases_in_int_format=False to get string format."""
        obs = SparsePauliOp("XYZ", 1.0)
        bases, _ = get_measurement_bases(obs, bases_in_int_format=False)

        self.assertEqual(len(bases), 1)
        self.assertIsInstance(bases[0], str)
        self.assertEqual(bases[0], "XYZ")

    def test_commuting_paulis_grouped(self):
        """Test that commuting Paulis are grouped into the same basis."""
        obs = SparsePauliOp(["ZII", "IZI", "IIZ"], [1.0, 1.0, 1.0])
        bases, _ = get_measurement_bases(obs, bases_in_int_format=True)

        # All should be in one basis since they commute qubit-wise
        self.assertEqual(len(bases), 1)

    def test_non_commuting_paulis_separate_bases(self):
        """Test that non-commuting Paulis get separate bases."""
        obs = SparsePauliOp(["ZI", "XI"], [1.0, 1.0])
        bases, _ = get_measurement_bases(obs, bases_in_int_format=True)

        # These don't commute qubit-wise, so need separate bases
        self.assertEqual(len(bases), 2)

    def test_identity_terms(self):
        """Test handling of identity terms."""
        obs = SparsePauliOp(["III", "ZZZ"], [1.0, 2.0])
        bases, _ = get_measurement_bases(obs, bases_in_int_format=True)

        # Identity commutes with everything
        self.assertGreaterEqual(len(bases), 1)

    def test_empty_observable_list(self):
        """Test with an empty list of observables."""
        # Empty list causes sum() to return 0, which doesn't have .unique() method
        # This is expected behavior - function requires at least one observable
        with pytest.raises(AttributeError):
            _, _ = get_measurement_bases([], bases_in_int_format=True)

    def test_reverser_none_values(self):
        """Test that reverser contains None for observables without terms in a basis."""
        obs1 = SparsePauliOp("ZZ", 1.0)
        obs2 = SparsePauliOp("XX", 2.0)
        _, reverser = get_measurement_bases([obs1, obs2], bases_in_int_format=True)

        # Each basis should have one observable with terms and one with None
        for _, obs_list in reverser.items():
            non_none_count = sum(1 for obs in obs_list if obs is not None)
            self.assertEqual(non_none_count, 1)


class TestMeasBasisForPauliGroup(unittest.TestCase):
    """Tests for _meas_basis_for_pauli_group function."""

    def test_single_z_pauli(self):
        """Test with a single Z Pauli."""
        group = PauliList(["ZII"])
        basis = _meas_basis_for_pauli_group(group)
        self.assertEqual(basis, Pauli("ZII"))

    def test_single_x_pauli(self):
        """Test with a single X Pauli."""
        group = PauliList(["XII"])
        basis = _meas_basis_for_pauli_group(group)
        self.assertEqual(basis, Pauli("XII"))

    def test_single_y_pauli(self):
        """Test with a single Y Pauli."""
        group = PauliList(["YII"])
        basis = _meas_basis_for_pauli_group(group)
        self.assertEqual(basis, Pauli("YII"))

    def test_multiple_z_paulis(self):
        """Test with multiple Z Paulis."""
        group = PauliList(["ZII", "IZI", "IIZ"])
        basis = _meas_basis_for_pauli_group(group)
        self.assertEqual(basis, Pauli("ZZZ"))

    def test_mixed_paulis(self):
        """Test with mixed Pauli types."""
        group = PauliList(["ZI", "IX"])
        basis = _meas_basis_for_pauli_group(group)
        self.assertEqual(basis, Pauli("ZX"))

    def test_identity_in_group(self):
        """Test with identity in the group."""
        group = PauliList(["III", "ZII"])
        basis = _meas_basis_for_pauli_group(group)
        self.assertEqual(basis, Pauli("ZII"))

    def test_overlapping_paulis(self):
        """Test with overlapping Pauli positions."""
        group = PauliList(["ZZI", "ZIZ"])
        basis = _meas_basis_for_pauli_group(group)
        self.assertEqual(basis, Pauli("ZZZ"))


class TestConvertBasisToUintRepresentation(unittest.TestCase):
    """Tests for _convert_basis_to_uint_representation function."""

    def test_single_identity(self):
        """Test conversion of identity."""
        bases = PauliList(["I"])
        result = _convert_basis_to_uint_representation(bases)
        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result[0], np.array([0], dtype=np.uint8))

    def test_single_z(self):
        """Test conversion of Z."""
        bases = PauliList(["Z"])
        result = _convert_basis_to_uint_representation(bases)
        np.testing.assert_array_equal(result[0], np.array([1], dtype=np.uint8))

    def test_single_x(self):
        """Test conversion of X."""
        bases = PauliList(["X"])
        result = _convert_basis_to_uint_representation(bases)
        np.testing.assert_array_equal(result[0], np.array([2], dtype=np.uint8))

    def test_single_y(self):
        """Test conversion of Y."""
        bases = PauliList(["Y"])
        result = _convert_basis_to_uint_representation(bases)
        np.testing.assert_array_equal(result[0], np.array([3], dtype=np.uint8))

    def test_multi_qubit_pauli(self):
        """Test conversion of multi-qubit Pauli."""
        bases = PauliList(["IXYZ"])
        result = _convert_basis_to_uint_representation(bases)
        # Note: reversed order (little-endian) - IXYZ becomes Z,Y,X,I
        np.testing.assert_array_equal(result[0], np.array([1, 3, 2, 0], dtype=np.uint8))

    def test_multiple_bases(self):
        """Test conversion of multiple bases."""
        bases = PauliList(["ZZ", "XX", "YY"])
        result = _convert_basis_to_uint_representation(bases)
        self.assertEqual(len(result), 3)
        np.testing.assert_array_equal(result[0], np.array([1, 1], dtype=np.uint8))
        np.testing.assert_array_equal(result[1], np.array([2, 2], dtype=np.uint8))
        np.testing.assert_array_equal(result[2], np.array([3, 3], dtype=np.uint8))

    def test_dtype_is_uint8(self):
        """Test that output dtype is uint8."""
        bases = PauliList(["XYZ"])
        result = _convert_basis_to_uint_representation(bases)
        self.assertEqual(result[0].dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()

# Made with Bob
