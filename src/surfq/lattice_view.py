"""A module for representing a view of a surface code lattice.

This module provides a `LatticeView` class that represents a view of a
surface code lattice. A lattice view is a subset of the qubits in a lattice,
and it can be used to apply quantum gates to specific qubits.

Typical usage example:

  >>> from surfq import Lattice
  >>>
  >>> lattice = Lattice(3)
  >>> view = lattice[0, 0]
  >>> view.X()
"""

from typing import final

import numpy as np
from numpy.typing import NDArray

from .pauli import Pauli
from .plotting import plot_lattice

QubitMask = NDArray[np.uint64]

QubitAxisIndex = int | slice | list[int] | QubitMask

QubitIndex = (
    QubitAxisIndex | tuple[QubitAxisIndex, QubitAxisIndex] | list[tuple[int, int]]
)


@final
class LatticeView:
    """A class representing a view of a surface code lattice.

    A lattice view is a subset of the qubits in a lattice, and it can be used
    to apply quantum gates to specific qubits.

    Attributes:
        L: The size of the original LxL lattice.
        n: The number of qubits in the original lattice.
        paulis: The Pauli operators applied on the original lattice's qubits.
        tableau: The tableau of the stabilizers of the original lattice.
        mask: A mask of indexes that selects the qubits in the view prior to operations.
    """

    def __init__(
        self,
        L: int,
        paulis: NDArray[np.uint8],
        tableau: NDArray[np.uint8],
        qubits: QubitIndex,
    ):
        """Initialise a new lattice view.

        Args:
            L: The size of the original LxL lattice.
            paulis: The Pauli operators applied on the original lattice's qubits.
            tableau: The tableau of the stabilizers.
            qubits: The qubits in the view.
        """
        self.L = L
        self.n = L**2
        self.paulis = paulis
        self.tableau = tableau
        self.mask = self._build_mask(qubits)

    def _build_mask(self, qubits: QubitIndex) -> QubitMask:
        """
        Transform qubit indices into a valid mask.

        Qubit indices can be specified using either single- or double-axis notation:

        - Single-axis: Qubits are indexed over the full lattice, i.e., [0, n)
        - Double-axis: Qubits are indexed along x and y axes, i.e., [0, L) for each axis

        Each index may be one of:
        - int: A single qubit within the respective range
        - list[int]: Multiple qubits within the respective range
        - slice: Multiple qubits within the respective range
        - np.ndarray: Multiple qubits within the respective range
        - list[tuple[int, int]]: Explicit list of (x, y) coordinates

        The lattice is read from left to right and top to bottom.

        Examples:
        - Single qubit:
            - lattice[6]           : 7th qubit in the lattice
            - lattice[2, 5]        : Qubit at column 3, row 6

        - Slicing:
            - lattice[:]            : All qubits in the lattice
            - lattice[:, :]         : All qubits in the lattice
            - lattice[6, :]         : All qubits in column 7
            - lattice[:, 2]         : All qubits in row 3
            - lattice[::2, 1::2]    : Qubits at odd rows and even columns

        - Lists and arrays:
            - lattice[[1, 5, 6]]               : Qubits 2, 6, and 7
            - lattice[np.array([1, 5, 6])]     : Qubits 2, 6, and 7
            - lattice[[1, 5, 6], 2]            : Qubits 2, 6, and 7 in row 3
            - lattice[:, np.array([1, 5, 6])]  : Qubits 2, 6, and 7 in each column

        - Coordinates list:
            - lattice[[(1, 0), (5, 2)]] : Qubits at coordinates (1, 0) and (5, 2)
        """
        if isinstance(qubits, tuple):
            xs = qubits[0]
            if isinstance(xs, slice):
                start = 0 if xs.start is None else int(xs.start)  # pyright: ignore[reportAny]
                stop = self.L if xs.stop is None else int(xs.stop)  # pyright: ignore[reportAny]
                step = 1 if xs.step is None else int(xs.step)  # pyright: ignore[reportAny]
                xs = np.arange(start, stop, step, dtype=np.uint64)

            ys = qubits[1]
            if isinstance(ys, slice):
                start = 0 if ys.start is None else int(ys.start)  # pyright: ignore[reportAny]
                stop = self.L if ys.stop is None else int(ys.stop)  # pyright: ignore[reportAny]
                step = 1 if ys.step is None else int(ys.step)  # pyright: ignore[reportAny]
                ys = np.arange(start, stop, step, dtype=np.uint64)

            xs_len = 1 if isinstance(xs, int) else len(xs)
            ys_len = 1 if isinstance(ys, int) else len(ys)

            return (np.repeat(xs, ys_len) + np.tile(ys, xs_len) * self.L).astype(
                np.uint64
            )
        elif isinstance(qubits, slice):
            start = 0 if qubits.start is None else qubits.start  # pyright: ignore[reportAny]
            stop = self.n if qubits.stop is None else qubits.stop  # pyright: ignore[reportAny]
            step = 1 if qubits.step is None else qubits.step  # pyright: ignore[reportAny]
            return np.arange(start, stop, step, dtype=np.uint64)
        elif isinstance(qubits, int):
            return np.array([qubits], dtype=np.uint64)
        elif isinstance(qubits, list):
            if all(isinstance(item, int) for item in qubits):
                return np.array(qubits, dtype=np.uint64)
            else:
                return np.array([x + y * self.L for x, y in qubits])  # pyright: ignore[reportUnknownVariableType]
        else:
            return qubits

    def H(self):
        """Apply a Hadamard gate to the qubits in the view."""
        self.paulis[self.mask] = (self.paulis[self.mask] >> 1) + (
            self.paulis[self.mask] << 1
        ) % 4

        mask = self.mask
        self.tableau[:, mask], self.tableau[:, mask + self.n] = (
            self.tableau[:, mask + self.n].copy(),
            self.tableau[:, mask].copy(),
        )
        self.tableau[:, -1] ^= np.bitwise_xor.reduce(
            self.tableau[:, mask] & self.tableau[:, mask + self.n], axis=1
        )
        return self

    def S(self):
        """Apply an S gate to the qubits in the view."""
        mask = self.mask
        self.tableau[:, mask + self.n] ^= self.tableau[:, mask]

        self.tableau[:, -1] ^= np.bitwise_xor.reduce(
            self.tableau[:, mask] & self.tableau[:, mask + self.n], axis=1
        )
        # self.tableau[:, -1] ^= self.tableau[:, mask] & self.tableau[:, mask + self.n]
        return self

    def CX(self, target: "LatticeView"):
        """Apply a CNOT gate to the qubits in the view.

        Args:
            target: The target qubits for the CNOT gate.

        Raises:
            ValueError: If the number of control and target qubits do not match.
        """
        c_mask = self.mask
        t_mask = target.mask
        if len(c_mask) != len(t_mask):
            raise ValueError(
                f"dimension mismatch between control and target indexes: \
                {self.mask} and {target.mask}"
            )
        self.tableau[:, t_mask] ^= self.tableau[:, c_mask]
        self.tableau[:, self.n + c_mask] ^= self.tableau[:, self.n + t_mask]
        self.tableau[:, -1] ^= (
            self.tableau[:, c_mask]
            & self.tableau[:, self.n + t_mask]
            & (self.tableau[:, t_mask] ^ self.tableau[:, self.n + c_mask] ^ 1)
        )
        return self

    def Z(self):
        """Apply a Z gate to the qubits in the view."""
        self.paulis[self.mask] ^= Pauli.Z.value
        return self.S().S()

    def X(self):
        """Apply an X gate to the qubits in the view."""
        return self.H().Z().H()

    def Y(self):
        """Apply a Y gate to the qubits in the view."""
        return self.X().Z()

    def show(self):
        """Plot the original lattice."""
        return plot_lattice(self.L, self.tableau, self.paulis)
