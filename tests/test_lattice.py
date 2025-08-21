"""Test the Lattice class."""

import numpy as np
import pytest

from surfq.lattice import Lattice
from surfq.lattice_view import LatticeView


def test_lattice_init():
    """Test Lattice initialization."""
    lattice = Lattice(3)
    assert lattice.L == 3
    assert lattice.n == 9
    assert len(lattice.tableau) == 8
    assert len(lattice.stabilisers_coords) == 8
    assert len(lattice.paulis) == 9

    with pytest.raises(ValueError):
        Lattice(0)

    with pytest.raises(ValueError):
        Lattice(2)


def test_create_stabilisers():
    """Test the create_stabilisers method."""
    tableau, coords = Lattice.create_stabilisers(3)
    assert tableau.shape == (8, 19)
    assert coords.shape == (8, 2)


def test_x_z_stabilisers():
    """Test the X and Z stabilisers properties."""
    lattice = Lattice(3)
    x_stabs, x_coords = lattice.X_stabilisers
    z_stabs, z_coords = lattice.Z_stabilisers
    assert len(x_stabs) == 4
    assert len(x_coords) == 4
    assert len(z_stabs) == 4
    assert len(z_coords) == 4


def test_lattice_getitem():
    """Test the __getitem__ method."""
    lattice = Lattice(3)
    view = lattice[0, 0]
    assert isinstance(view, LatticeView)


def test_validate_qubit():
    """Test the _validate_qubit method."""
    lattice = Lattice(5)

    # Valid cases
    lattice._validate_qubit(0)
    lattice._validate_qubit(24)
    lattice._validate_qubit(slice(0, 25))
    lattice._validate_qubit(np.array([0, 24]))
    lattice._validate_qubit((0, 0))
    lattice._validate_qubit((slice(0, 5), slice(0, 5)))

    # Invalid cases
    with pytest.raises(ValueError):
        lattice._validate_qubit(-1)
    with pytest.raises(ValueError):
        lattice._validate_qubit(25)
    with pytest.raises(ValueError):
        lattice._validate_qubit(slice(-1, 5))
    with pytest.raises(ValueError):
        lattice._validate_qubit(slice(0, 26))
    with pytest.raises(ValueError):
        lattice._validate_qubit(np.array([-1, 24]))
    with pytest.raises(ValueError):
        lattice._validate_qubit(np.array([0, 25]))
    with pytest.raises(TypeError):
        lattice._validate_qubit(np.array([0.0, 24.0]))
    with pytest.raises(ValueError):
        lattice._validate_qubit((-1, 0))
    with pytest.raises(ValueError):
        lattice._validate_qubit((0, 5))
    with pytest.raises(ValueError):
        lattice._validate_qubit((slice(-1, 5), slice(0, 5)))
    with pytest.raises(ValueError):
        lattice._validate_qubit((slice(0, 6), slice(0, 5)))
