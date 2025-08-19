"""Surface Code Lattice."""

from typing import Callable, final

import numpy as np
from numpy.typing import NDArray

from .lattice_view import LatticeView, QubitIndex
from .pauli import Pauli
from .plotting import plot_lattice


@final
class Lattice:
    """Define Surface Code Lattice class."""

    def __init__(self, L: int):
        """Initialise a square n×n lattice of qubits, all set to identity Pauli."""
        if L <= 0:
            raise ValueError("n must be a positive integer")
        elif L % 2 != 1:
            raise ValueError("n should be odd")
        self.L = L
        self.n = L**2
        self.stabilisers = self.surface_code_stabilisers(L)
        self.tableau = np.hstack(
            [
                self.stabilisers,
                np.zeros((self.stabilisers.shape[0], 1), dtype=self.stabilisers.dtype),
            ]
        )
        self.paulis: NDArray[np.uint8] = np.array(
            [Pauli.I.value for _ in range(self.n)], dtype=np.uint8
        )

    @staticmethod
    def surface_code_stabilisers(L: int):
        """Define stabilisers."""
        n = L**2
        stabilisers = np.zeros((n - 1, 2 * n), dtype=np.uint8)

        s = 0

        # Set up inner X stabilisers
        for row in range(L - 1):
            for col in range((L - 1) // 2):
                i = row * L
                j = 2 * col + (0 if row % 2 == 0 else 1)
                mask = i + j + np.array([0, 1, L, L + 1])
                stabilisers[s, mask] = 1
                s += 1

        # Set up top X stabilisers
        for col in range((L - 1) // 2):
            j = 2 * col + 1
            mask = j + np.array([0, 1])
            stabilisers[s, mask] = 1
            s += 1

        # Set up bottom X stabilisers
        for col in range((L - 1) // 2):
            j = 2 * col
            mask = L * (L - 1) + j + np.array([0, 1])
            stabilisers[s, mask] = 1
            s += 1

        # Set up inner Z stabilisers
        for row in range(L - 1):
            for col in range((L - 1) // 2):
                i = row * L
                j = 2 * col + (0 if row % 2 == 1 else 1)
                mask = n + i + j + np.array([0, 1, L, L + 1])
                stabilisers[s, mask] = 1
                s += 1

        # Set up left Z stabilisers
        for row in range((L - 1) // 2):
            j = 2 * row * L
            mask = n + j + np.array([0, L])
            stabilisers[s, mask] = 1
            s += 1

        # Set up right Z stabilisers
        for row in range((L - 1) // 2):
            j = (2 * row + 1) * L
            mask = n + j + L - 1 + np.array([0, L])
            stabilisers[s, mask] = 1
            s += 1

        return stabilisers

    @staticmethod
    def _validate_qubit(
        func: Callable[..., "LatticeView"],
    ) -> Callable[..., "LatticeView"]:
        def wrapper(self: "Lattice", *qubits: QubitIndex) -> "LatticeView":
            for q in qubits:
                if isinstance(q, int):
                    if not 0 <= q < self.n:
                        raise ValueError(
                            f"invalid out-of-range qubit: \
                            index must be 0 <= qubit < {self.n}"
                        )
                elif isinstance(q, slice):
                    start = 0 if q.start is None else q.start  # pyright: ignore[reportAny]
                    stop = self.n if q.stop is None else q.stop  # pyright: ignore[reportAny]
                    if not (0 <= start < self.n) or not (0 <= stop <= self.n):
                        raise ValueError(
                            f"invalid out-of-range qubit slice {q}: \
                            start and end must be 0 <= qubit < {self.n}"
                        )
                elif isinstance(q, np.ndarray):
                    if not np.issubdtype(q.dtype, np.integer):
                        raise TypeError(
                            f"qubit array must contain integers, got {q.dtype}"
                        )
                    if not np.all((0 <= q) & (q < self.n)):
                        raise ValueError(
                            f"invalid out-of-range qubits in array {q}: \
                            all elements must satisfy 0 <= qubit < {self.n}"
                        )
                else:
                    if isinstance(q[0], int) and not 0 <= q[0] < self.L:
                        raise ValueError(
                            f"invalid out-of-range qubit x-coordinate: \
                            must be 0 <= qubit < {self.L}"
                        )
                    if isinstance(q[0], slice):
                        start = 0 if q[0].start is None else int(q[0].start)  # pyright: ignore[reportAny]
                        stop = self.L if q[0].stop is None else int(q[0].stop)  # pyright: ignore[reportAny]
                        if not (0 <= start < self.L) or not (0 <= stop <= self.L):
                            raise ValueError(
                                f"invalid out-of-range qubit x-coordinate slice {q}: \
                                must be 0 <= qubit < {self.L}"
                            )

                    # y-coordinate
                    if isinstance(q[1], int) and not 0 <= q[1] < self.L:
                        raise ValueError(f" {self.L}")
                    if isinstance(q[1], slice):
                        start = 0 if q[1].start is None else int(q[1].start)  # pyright: ignore[reportAny]
                        stop = self.L if q[1].stop is None else int(q[1].stop)  # pyright: ignore[reportAny]
                        if not (0 <= start < self.L) or not (0 <= stop <= self.L):
                            raise ValueError(
                                f"invalid out-of-range qubit y-coordinate slice {q}: \
                                must be 0 <= qubit < {self.L}"
                            )
            return func(self, *qubits)

        return wrapper

    @_validate_qubit
    def __getitem__(self, qubits: QubitIndex) -> "LatticeView":
        """Retrieve mutable view of Lattice."""
        return LatticeView(self.L, self.paulis, self.tableau, qubits)

    def show(self) -> None:
        """Plot lattice."""
        plot_lattice(self.L, self.tableau, self.paulis)
