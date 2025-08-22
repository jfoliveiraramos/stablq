"""Test the LatticeView class."""

import numpy as np
import pytest

from surfq.lattice import Lattice
from surfq.pauli import Pauli


@pytest.fixture
def lattice():
    """Return a 3x3 lattice."""
    return Lattice(3)


def test_build_mask(lattice: Lattice):
    """Test the _build_mask method."""
    # Single-axis indexing
    np.testing.assert_array_equal(lattice[0].mask, [0])
    np.testing.assert_array_equal(lattice[8].mask, [8])
    np.testing.assert_array_equal(lattice[:].mask, np.arange(9))
    np.testing.assert_array_equal(lattice[1:4].mask, np.arange(1, 4))
    np.testing.assert_array_equal(lattice[[0, 8]].mask, [0, 8])

    # Double-axis indexing
    np.testing.assert_array_equal(lattice[0, 0].mask, [0])
    np.testing.assert_array_equal(lattice[2, 2].mask, [8])
    np.testing.assert_array_equal(lattice[:, :].mask, np.arange(9))
    np.testing.assert_array_equal(lattice[0, :].mask, [0, 3, 6])
    np.testing.assert_array_equal(lattice[:, 0].mask, [0, 1, 2])
    np.testing.assert_array_equal(lattice[1:3, 1:3].mask, [4, 5, 7, 8])
    np.testing.assert_array_equal(lattice[[0, 2], [0, 2]].mask, [0, 2, 6, 8])

    # List of tuples
    np.testing.assert_array_equal(lattice[[(0, 0), (2, 2)]].mask, [0, 8])


def test_h_gate(lattice: Lattice):
    """Test the H gate."""
    _ = lattice[0].H()
    assert lattice.paulis[0] == Pauli.I.value


def test_s_gate(lattice: Lattice):
    """Test the S gate."""
    _ = lattice[0].S()
    assert lattice.paulis[0] == Pauli.I.value


def test_cx_gate(lattice: Lattice):
    """Test the CX gate."""
    _ = lattice[0].CX(lattice[1])
    _ = lattice[0, :].CX(lattice[1, :])
    assert lattice.paulis[0] == Pauli.I.value
    assert lattice.paulis[1] == Pauli.I.value

    with pytest.raises(ValueError):
        _ = lattice[0].CX(lattice[[1, 2]])


def test_x_gate(lattice: Lattice):
    """Test the X gate."""
    _ = lattice[0].X()
    assert lattice.paulis[0] == Pauli.X.value


def test_y_gate(lattice: Lattice):
    """Test the Y gate."""
    _ = lattice[0].Y()
    assert lattice.paulis[0] == Pauli.Y.value


def test_z_gate(lattice: Lattice):
    """Test the Z gate."""
    _ = lattice[0].Z()
    assert lattice.paulis[0] == Pauli.Z.value
