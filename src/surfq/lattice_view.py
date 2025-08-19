"""Operable View of Surface Code Lattice."""

from typing import final

import numpy as np
from numpy.typing import NDArray

from .pauli import Pauli
from .plotting import plot_lattice

QubitMask = NDArray[np.uint64]
QubitAxisIndex = int | slice | NDArray[np.uint64]
QubitIndex = int | slice | tuple[QubitAxisIndex, QubitAxisIndex] | QubitMask


@final
class LatticeView:
    """Define Operable View of Surface Code Lattice."""

    def __init__(
        self,
        L: int,
        paulis: NDArray[np.uint8],
        tableau: NDArray[np.uint8],
        qubits: QubitIndex,
    ):
        self.L = L
        self.n = L**2
        self.paulis = paulis
        self.tableau = tableau
        self.mask = self._build_mask(qubits)

    def _build_mask(self, qubits: QubitIndex) -> QubitMask:
        if isinstance(qubits, tuple):
            x = qubits[0]
            if isinstance(x, slice):
                start = 0 if x.start is None else int(x.start)  # pyright: ignore[reportAny]
                stop = self.L if x.stop is None else int(x.stop)  # pyright: ignore[reportAny]
                step = 1 if x.step is None else int(x.step)  # pyright: ignore[reportAny]
                x = np.arange(start, stop, step, dtype=np.uint64)

            y = qubits[1]
            if isinstance(y, slice):
                start = 0 if y.start is None else int(y.start)  # pyright: ignore[reportAny]
                stop = self.L if y.stop is None else int(y.stop)  # pyright: ignore[reportAny]
                step = 1 if y.step is None else int(y.step)  # pyright: ignore[reportAny]
                y = np.arange(start, stop, step, dtype=np.uint64)

            x_len = 1 if isinstance(x, int) else len(x)
            y_len = 1 if isinstance(y, int) else len(y)

            return (np.repeat(x, y_len) + np.tile(y, x_len) * self.L).astype(np.uint64)
        elif isinstance(qubits, slice):
            start = 0 if qubits.start is None else qubits.start  # pyright: ignore[reportAny]
            stop = self.n if qubits.stop is None else qubits.stop  # pyright: ignore[reportAny]
            step = 1 if qubits.step is None else qubits.step  # pyright: ignore[reportAny]
            return np.arange(start, stop, step, dtype=np.uint64)
        elif isinstance(qubits, int):
            return np.array([qubits], dtype=np.uint64)
        else:
            return qubits

    def H(self):
        """Apply Hadamard gate."""
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
        """Apply S gate."""
        mask = self.mask
        self.tableau[:, mask + self.n] ^= self.tableau[:, mask]

        self.tableau[:, -1] ^= np.bitwise_xor.reduce(
            self.tableau[:, mask] & self.tableau[:, mask + self.n], axis=1
        )
        # self.tableau[:, -1] ^= self.tableau[:, mask] & self.tableau[:, mask + self.n]
        return self

    def CX(self, target: "LatticeView"):
        """Apply Controlled-X gate."""
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
        """Apply Z gate."""
        self.paulis[self.mask] ^= Pauli.Z.value
        return self.S().S()

    def X(self):
        """Apply X gate."""
        return self.H().Z().H()

    def Y(self):
        """Apply Y gate."""
        return self.X().Z()

    def show(self):
        """Plot lattice."""
        return plot_lattice(self.L, self.tableau, self.paulis)
