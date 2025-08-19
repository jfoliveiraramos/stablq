"""Define Surface Code Lattice."""

from enum import Enum
from typing import Callable, final, override

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Polygon, Wedge
from numpy.typing import NDArray


class Pauli(Enum):
    """Single-Qubit Pauli Operator."""

    I = 0
    Z = 1
    X = 2
    Y = 3

    def __mul__(self, other: "Pauli") -> "Pauli":
        """Define Pauli Mulitplication."""
        return Pauli(self.value ^ other.value)

    @override
    def __repr__(self) -> str:
        match self:
            case Pauli.I:
                return "I"
            case Pauli.Z:
                return "Z"
            case Pauli.X:
                return "X"
            case Pauli.Y:
                return "Y"

    @override
    def __str__(self) -> str:
        match self:
            case Pauli.I:
                return "I"
            case Pauli.Z:
                return "Z"
            case Pauli.X:
                return "X"
            case Pauli.Y:
                return "Y"


QubitIndex = int | slice | tuple[int | slice, int | slice]
QubitMask = NDArray[np.uint64]


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
        func: Callable[..., "MutableLattice"],
    ) -> Callable[..., "MutableLattice"]:
        def wrapper(self: "Lattice", *qubits: QubitIndex) -> "MutableLattice":
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
                else:
                    if isinstance(q[0], int) and not 0 <= q[0] < self.L:
                        raise ValueError(
                            f"invalid out-of-range qubit x-coordinate: \
                            must be 0 <= qubit < {self.L}"
                        )
                    if isinstance(q[0], slice):
                        start = 0 if q[0].start is None else q[0].start  # pyright: ignore[reportAny]
                        stop = self.L if q[0].stop is None else q[0].stop  # pyright: ignore[reportAny]
                        if not (0 <= start < self.L) or not (0 <= stop <= self.L):
                            raise ValueError(
                                f"invalid out-of-range qubit x-coordinate slice {q}: \
                                must be 0 <= qubit < {self.L}"
                            )

                    # y-coordinate
                    if isinstance(q[1], int) and not 0 <= q[1] < self.L:
                        raise ValueError(
                            f"invalid out-of-range qubit y-coordinate: \
                            must be 0 <= qubit < {self.L}"
                        )
                    if isinstance(q[1], slice):
                        start = 0 if q[1].start is None else q[1].start  # pyright: ignore[reportAny]
                        stop = self.L if q[1].stop is None else q[1].stop  # pyright: ignore[reportAny]
                        if not (0 <= start < self.L) or not (0 <= stop <= self.L):
                            raise ValueError(
                                f"invalid out-of-range qubit y-coordinate slice {q}: \
                                must be 0 <= qubit < {self.L}"
                            )
            return func(self, *qubits)

        return wrapper

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

            return np.repeat(x, len(y)) + np.tile(y, len(x)) * self.L
        elif isinstance(qubits, slice):
            start = 0 if qubits.start is None else qubits.start  # pyright: ignore[reportAny]
            stop = self.n if qubits.stop is None else qubits.stop  # pyright: ignore[reportAny]
            step = 1 if qubits.step is None else qubits.step  # pyright: ignore[reportAny]
            return np.arange(start, stop, step, dtype=np.uint64)
        return np.array([qubits], dtype=np.uint64)

    @_validate_qubit
    def __getitem__(self, qubits: QubitIndex) -> "MutableLattice":
        """Retrieve mutable view of Lattice."""
        return MutableLattice(self, self._build_mask(qubits))

    def plot(self) -> None:
        """Plot lattice."""
        L: int = self.L
        n: int = self.n
        tableau: NDArray[np.uint8] = self.tableau
        x_grid, y_grid = np.meshgrid(np.arange(L), np.arange(L))
        grid_x, grid_y = x_grid.flatten(), y_grid.flatten()
        ax = plt.gca()  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

        def draw_polygon_or_wedge(
            indices: NDArray[np.intp],
            color: str,
            alpha: float,
        ) -> None:
            x = indices % L
            y = L - 1 - indices // L

            if len(indices) == 4:
                pts = np.column_stack((x, y))
                pts[[-2, -1]] = pts[[-1, -2]]
                patch = Polygon(list(pts), color=color, alpha=alpha)
            elif len(indices) == 2:
                cx, cy = float(np.mean(x)), float(np.mean(y))
                if y[0] == y[1] == 0:
                    theta1, theta2 = (180, 0)
                elif y[0] == y[1] == L - 1:
                    theta1, theta2 = (0, 180)
                elif x[0] == x[1] == 0:
                    theta1, theta2 = (90, -90)
                else:
                    theta1, theta2 = (-90, 90)
                patch = Wedge(
                    center=(cx, cy),
                    r=0.5,
                    theta1=theta1,
                    theta2=theta2,
                    color=color,
                    alpha=alpha,
                )
            else:
                return

            ax.add_patch(patch)  # pyright: ignore[reportUnknownMemberType]

        for stab in tableau[: (n - 1) // 2]:  # pyright: ignore[reportAny]
            indices = np.where(stab[:n] == 1)[0]  # pyright: ignore[reportAny]
            if stab[-1] != 1:
                draw_polygon_or_wedge(indices, "orange", 0.5)
            else:
                draw_polygon_or_wedge(indices, "red", 0.5)

        for stab in tableau[(n - 1) // 2 :]:  # pyright: ignore[reportAny]
            indices = np.where(stab[n:-1] == 1)[0]  # pyright: ignore[reportAny]
            if stab[-1] != 1:
                draw_polygon_or_wedge(indices, "cyan", 0.5)
            else:
                draw_polygon_or_wedge(indices, "red", 0.3)

        ax.scatter(grid_x, grid_y, s=100, color="k")  # pyright: ignore[reportUnknownMemberType]
        ax.set_xticks(np.arange(L))  # pyright: ignore[reportUnknownMemberType]
        ax.set_yticks(np.arange(L))  # pyright: ignore[reportUnknownMemberType]
        ax.set_aspect("equal")  # pyright: ignore[reportUnknownMemberType]
        ax.set_xlim(-1, L)  # pyright: ignore[reportUnknownMemberType]
        ax.set_ylim(-1, L)  # pyright: ignore[reportUnknownMemberType]
        ax.grid(True)  # pyright: ignore[reportUnknownMemberType]

        ax.legend(  # pyright: ignore[reportUnknownMemberType]
            handles=[
                Patch(
                    facecolor="orange",
                    alpha=0.5,
                    edgecolor="black",
                    label="X-Stabiliser",
                ),
                Patch(
                    facecolor="cyan",
                    alpha=0.5,
                    edgecolor="black",
                    label="Z-Stabiliser",
                ),
                Patch(
                    facecolor="red",
                    alpha=0.5,
                    edgecolor="black",
                    label="X-Stabiliser Syndrome",
                ),
                Patch(
                    facecolor="red",
                    alpha=0.3,
                    edgecolor="black",
                    label="Z-Stabiliser Syndrome",
                ),
            ],
            handlelength=1,
            handleheight=1,
            borderpad=0.5,
            loc="upper left",
            bbox_to_anchor=(0.01, -0.05),
            fontsize=11,
            markerscale=2.0,
        )

        ops = [(Pauli(op), q % L, L - q // L - 1) for q, op in enumerate(self.paulis)]  # pyright: ignore[reportAny]
        for op, x, y in ops:
            if op == Pauli.I:
                continue
            _ = plt.plot(x, y, marker="o", color="red", markersize=10)
            _ = plt.text(  # pyright: ignore[reportUnknownMemberType]
                x + 0.1, y + 0.1, f"${op}$", ha="left", va="bottom"
            )

        plt.show()  # pyright: ignore[reportUnknownMemberType]


@final
class MutableLattice:
    """Operable View of Surface Code Lattice."""

    def __init__(self, lattice: Lattice, mask: QubitMask):
        self.lattice = lattice
        self.L = lattice.L
        self.n = lattice.n
        self.paulis = lattice.paulis
        self.tableau = lattice.tableau

        self.mask = mask

    def H(self):
        """Apply Hadamard gate."""
        self.lattice.paulis[self.mask] = (self.lattice.paulis[self.mask] >> 1) + (
            self.lattice.paulis[self.mask] << 1
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

    def CX(self, target: "MutableLattice"):
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

    def __getitem__(self, qubits: QubitIndex) -> "MutableLattice":
        """Retrieve mutable view of Lattice."""
        return self.lattice[qubits]
