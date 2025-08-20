"""Define surface code lattice class."""

from typing import final

import numpy as np
from numpy.typing import NDArray

from .lattice_view import LatticeView, QubitIndex
from .pauli import Pauli
from .plotting import plot_lattice


@final
class Lattice:
    """Surface Code Lattice class."""

    def __init__(self, L: int):
        """Initialise a square n×n lattice of qubits, all set to identity Pauli."""
        if L <= 0:
            raise ValueError("n must be a positive integer")
        elif L % 2 != 1:
            raise ValueError("n should be odd")
        self.L = L
        self.n = L**2
        self.tableau, self.stabilisers_coords = self.create_stabilisers(L)
        self.paulis: NDArray[np.uint8] = np.array(
            [Pauli.I.value for _ in range(self.n)], dtype=np.uint8
        )

    @property
    def X_stabilisers(self):
        """Retrieve X stablisers."""
        return self.tableau[: (self.n - 1) // 2], self.stabilisers_coords[
            : (self.n - 1) // 2
        ]

    @property
    def Z_stabilisers(self):
        """Retrieve Z stablisers."""
        return self.tableau[(self.n - 1) // 2 :], self.stabilisers_coords[
            (self.n - 1) // 2 :
        ]

    @staticmethod
    def create_stabilisers(L: int):
        """Define stabilisers."""
        n = L**2
        stabilisers: list[NDArray[np.uint8]] = []
        coordinates: list[tuple[float, float]] = []

        # Set up inner X stabilisers
        for row in range(L - 1):
            for col in range((L - 1) // 2):
                i = row * L
                j = 2 * col + (0 if row % 2 == 0 else 1)
                mask = i + j + np.array([0, 1, L, L + 1])
                stabilisers.append(
                    np.isin(np.arange(n * 2 + 1), mask).astype(np.uint8),
                )
                coordinates.append((j + 0.5, row + 0.5))

        # Set up top X stabilisers
        for col in range((L - 1) // 2):
            j = 2 * col + 1
            mask = j + np.array([0, 1])
            stabilisers.append(
                np.isin(np.arange(n * 2 + 1), mask).astype(np.uint8),
            )
            coordinates.append((j + 0.5, -0.5))

        # Set up bottom X stabilisers
        for col in range((L - 1) // 2):
            j = 2 * col
            mask = L * (L - 1) + j + np.array([0, 1])
            stabilisers.append(
                np.isin(np.arange(n * 2 + 1), mask).astype(np.uint8),
            )
            coordinates.append((j + 0.5, L - 1 + 0.5))

        # Set up inner Z stabilisers
        for row in range(L - 1):
            for col in range((L - 1) // 2):
                i = row * L
                j = 2 * col + (0 if row % 2 == 1 else 1)
                mask = n + i + j + np.array([0, 1, L, L + 1])
                stabilisers.append(
                    np.isin(np.arange(n * 2 + 1), mask).astype(np.uint8),
                )
                coordinates.append((j + 0.5, row + 0.5))

        # Set up left Z stabilisers
        for row in range((L - 1) // 2):
            i = 2 * row * L
            mask = n + i + np.array([0, L])
            stabilisers.append(
                np.isin(np.arange(n * 2 + 1), mask).astype(np.uint8),
            )
            coordinates.append((-0.5, 2 * row + 0.5))

        # Set up right Z stabilisers
        for row in range((L - 1) // 2):
            i = (2 * row + 1) * L
            mask = n + i + L - 1 + np.array([0, L])
            stabilisers.append(
                np.isin(np.arange(n * 2 + 1), mask).astype(np.uint8),
            )
            coordinates.append((L - 1 + 0.5, 2 * row + 1 + 0.5))

        return np.array(stabilisers, dtype=np.uint8), np.array(
            coordinates, dtype=np.float64
        )

    def _validate_qubit(self, qubits: QubitIndex):
        if isinstance(qubits, list):
            qubits = np.array(qubits, dtype=np.uint64)
        if isinstance(qubits, int):
            if not 0 <= qubits < self.n:
                raise ValueError(
                    f"invalid out-of-range qubit: \
                        index must be 0 <= qubit < {self.n}"
                )
        elif isinstance(qubits, slice):
            start = 0 if qubits.start is None else qubits.start  # pyright: ignore[reportAny]
            stop = self.n if qubits.stop is None else qubits.stop  # pyright: ignore[reportAny]
            if not (0 <= start < self.n) or not (0 <= stop <= self.n):
                raise ValueError(
                    f"invalid out-of-range qubit slice {qubits}: \
                        start and end must be 0 <= qubit < {self.n}"
                )
        elif isinstance(qubits, np.ndarray):
            if not np.issubdtype(qubits.dtype, np.integer):
                raise TypeError(
                    f"qubit array must contain integers, got {qubits.dtype}"
                )
            if not np.all((0 <= qubits) & (qubits < self.n)):
                raise ValueError(
                    f"invalid out-of-range qubits in array {qubits}: \
                        all elements must satisfy 0 <= qubit < {self.n}"
                )
        else:
            if isinstance(qubits[0], int) and not 0 <= qubits[0] < self.L:
                raise ValueError(
                    f"invalid out-of-range qubit x-coordinate: \
                        must be 0 <= qubit < {self.L}"
                )
            if isinstance(qubits[0], slice):
                start = 0 if qubits[0].start is None else int(qubits[0].start)  # pyright: ignore[reportAny]
                stop = self.L if qubits[0].stop is None else int(qubits[0].stop)  # pyright: ignore[reportAny]
                if not (0 <= start < self.L) or not (0 <= stop <= self.L):
                    raise ValueError(
                        f"invalid out-of-range qubit x-coordinate slice {qubits}: \
                            must be 0 <= qubit < {self.L}"
                    )

            # y-coordinate
            if isinstance(qubits[1], int) and not 0 <= qubits[1] < self.L:
                raise ValueError(f" {self.L}")
            if isinstance(qubits[1], slice):
                start = 0 if qubits[1].start is None else int(qubits[1].start)  # pyright: ignore[reportAny]
                stop = self.L if qubits[1].stop is None else int(qubits[1].stop)  # pyright: ignore[reportAny]
                if not (0 <= start < self.L) or not (0 <= stop <= self.L):
                    raise ValueError(
                        f"invalid out-of-range qubit y-coordinate slice {qubits}: \
                            must be 0 <= qubit < {self.L}"
                    )

    def __getitem__(self, qubits: QubitIndex) -> "LatticeView":
        """Retrieve mutable view of Lattice."""
        self._validate_qubit(qubits)
        return LatticeView(self.L, self.paulis, self.tableau, qubits)

    def show(self) -> None:
        """Plot lattice."""
        plot_lattice(self.L, self.tableau, self.paulis)
