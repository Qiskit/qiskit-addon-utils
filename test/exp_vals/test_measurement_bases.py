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
    _convert_to_pauli,
    _meas_basis_for_pauli_group,
    find_measure_basis_to_observable_mapping,
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


class TestConvertToPauli(unittest.TestCase):
    """Tests for _convert_to_pauli function."""

    def test_pauli_input(self):
        """Test with Pauli object as input."""
        pauli = Pauli("XYZ")
        result = _convert_to_pauli(pauli)
        self.assertEqual(result, pauli)

    def test_string_input(self):
        """Test with string as input."""
        result = _convert_to_pauli("XYZ")
        self.assertEqual(result, Pauli("XYZ"))

    def test_list_of_ints_input(self):
        """Test with list of integers as input."""
        result = _convert_to_pauli([2, 3, 1])  # X, Y, Z
        self.assertEqual(result, Pauli("XYZ"))

    def test_numpy_array_input(self):
        """Test with numpy array as input."""
        result = _convert_to_pauli(np.array([0, 1, 2, 3], dtype=np.uint8))
        # Array [0,1,2,3] maps to I,Z,X,Y
        self.assertEqual(result, Pauli("IZXY"))

    def test_tuple_input(self):
        """Test with tuple as input."""
        result = _convert_to_pauli((1, 2, 3))
        self.assertEqual(result, Pauli("ZXY"))

    def test_identity_conversion(self):
        """Test conversion of identity."""
        result = _convert_to_pauli([0, 0, 0])
        self.assertEqual(result, Pauli("III"))

    def test_invalid_input_type(self):
        """Test with invalid input type."""
        with pytest.raises(
            ValueError, match="basis must be a Pauli instance, str or a list of ints"
        ):
            _convert_to_pauli({"invalid": "type"})

    def test_invalid_list_content(self):
        """Test with list containing non-integers."""
        with pytest.raises((ValueError, KeyError)):
            _convert_to_pauli(["X", "Y", "Z"])


class TestFindMeasureBasisToObservableMapping(unittest.TestCase):
    """Tests for find_measure_basis_to_observable_mapping function."""

    def test_single_observable_single_basis(self):
        """Test with single observable and single basis."""
        obs = SparsePauliOp("ZZZ", 1.0)
        bases = ["ZZZ"]
        result = find_measure_basis_to_observable_mapping([obs], bases)

        self.assertEqual(len(result), 1)
        basis_pauli = Pauli("ZZZ")
        self.assertIn(basis_pauli, result)
        self.assertEqual(len(result[basis_pauli]), 1)
        self.assertIsInstance(result[basis_pauli][0], SparsePauliOp)

    def test_multiple_observables_single_basis(self):
        """Test with multiple observables and single basis."""
        obs1 = SparsePauliOp("ZZI", 1.0)
        obs2 = SparsePauliOp("IZZ", 2.0)
        bases = ["ZZZ"]
        result = find_measure_basis_to_observable_mapping([obs1, obs2], bases)

        basis_pauli = Pauli("ZZZ")
        self.assertEqual(len(result[basis_pauli]), 2)

    def test_single_observable_multiple_bases(self):
        """Test with single observable and multiple bases."""
        obs = SparsePauliOp(["ZI", "XI"], [1.0, 2.0])
        bases = ["ZI", "XI"]
        result = find_measure_basis_to_observable_mapping([obs], bases)

        self.assertEqual(len(result), 2)
        # Each term should be mapped to its commuting basis
        for _, obs_list in result.items():
            self.assertEqual(len(obs_list), 1)
            if obs_list[0] is not None:
                self.assertEqual(len(obs_list[0].paulis), 1)

    def test_basis_as_int_list(self):
        """Test with basis as list of integers."""
        obs = SparsePauliOp("ZZ", 1.0)
        bases = [[1, 1]]  # ZZ in int format
        result = find_measure_basis_to_observable_mapping([obs], bases)

        self.assertEqual(len(result), 1)

    def test_basis_as_pauli_list(self):
        """Test with basis as PauliList."""
        obs = SparsePauliOp("XX", 1.0)
        bases = PauliList(["XX"])
        result = find_measure_basis_to_observable_mapping([obs], bases)

        self.assertEqual(len(result), 1)

    def test_observable_term_not_commuting_with_any_basis(self):
        """Test error when observable term doesn't commute with any basis."""
        obs = SparsePauliOp("XY", 1.0)
        bases = ["ZZ"]  # Doesn't commute with XY

        with pytest.raises(
            ValueError, match="Some observable elements do not commute with any measurement basis"
        ):
            find_measure_basis_to_observable_mapping([obs], bases)

    def test_none_observable_in_result(self):
        """Test that None is returned for observables without terms in a basis."""
        obs1 = SparsePauliOp("ZZ", 1.0)
        obs2 = SparsePauliOp("XX", 2.0)
        bases = ["ZZ", "XX"]
        result = find_measure_basis_to_observable_mapping([obs1, obs2], bases)

        # Each basis should have one observable with terms and one None
        for _, obs_list in result.items():
            self.assertEqual(len(obs_list), 2)
            none_count = sum(1 for obs in obs_list if obs is None)
            self.assertEqual(none_count, 1)

    def test_first_commuting_basis_used(self):
        """Test that only the first commuting basis is used for each term."""
        obs = SparsePauliOp("ZZ", 1.0)
        bases = ["ZZ", "ZI", "IZ"]  # All commute with ZZ
        result = find_measure_basis_to_observable_mapping([obs], bases)

        # The observable should only be in the first basis
        first_basis = Pauli("ZZ")
        self.assertIsNotNone(result[first_basis][0])

        # Other bases should have None
        second_basis = Pauli("ZI")
        third_basis = Pauli("IZ")
        self.assertIsNone(result[second_basis][0])
        self.assertIsNone(result[third_basis][0])

    def test_complex_observable_with_multiple_terms(self):
        """Test with complex observable containing multiple terms."""
        obs = SparsePauliOp(["ZZI", "IZZ", "XII"], [1.0, 2.0, 3.0])
        bases = ["ZZZ", "XXX"]
        result = find_measure_basis_to_observable_mapping([obs], bases)

        # ZZI and IZZ should map to ZZZ, XII should map to XXX
        z_basis = Pauli("ZZZ")
        x_basis = Pauli("XXX")

        self.assertIsNotNone(result[z_basis][0])
        self.assertEqual(len(result[z_basis][0].paulis), 2)

        self.assertIsNotNone(result[x_basis][0])
        self.assertEqual(len(result[x_basis][0].paulis), 1)

    def test_identity_terms(self):
        """Test handling of identity terms."""
        obs = SparsePauliOp(["III", "ZZZ"], [1.0, 2.0])
        bases = ["ZZZ"]
        result = find_measure_basis_to_observable_mapping([obs], bases)

        # Both terms should commute with ZZZ
        basis_pauli = Pauli("ZZZ")
        self.assertIsNotNone(result[basis_pauli][0])
        self.assertEqual(len(result[basis_pauli][0].paulis), 2)

    def test_empty_observables(self):
        """Test with empty observables list."""
        bases = ["ZZ"]
        result = find_measure_basis_to_observable_mapping([], bases)

        # Should have entries for each basis but with empty lists
        self.assertEqual(len(result), 1)
        basis_pauli = Pauli("ZZ")
        self.assertEqual(len(result[basis_pauli]), 0)


if __name__ == "__main__":
    unittest.main()

# Made with Bob
