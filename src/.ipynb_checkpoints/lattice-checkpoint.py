"""Define Surface Code Lattice."""

from enum import Enum
from typing import Callable, final

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


QubitIndex = int | slice[int, int, int]
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
        self.paulis = np.array([Pauli.I.value for _ in range(self.n)], dtype=np.uint8)

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
    def _validate_qubit(func: Callable[..., "Lattice"]) -> Callable[..., "Lattice"]:
        def wrapper(self: "Lattice", *qubits: QubitIndex) -> "Lattice":
            for q in qubits:
                if isinstance(q, int) and not 0 <= q < self.n:
                    raise ValueError(
                        f"invalid out-of-range qubit {q}: \
                        index must be 0 <= qubit < {self.n} \
                    "
                    )
                elif (
                    isinstance(q, slice)
                    and not 0 <= q.start <= self.n
                    and 0 <= q.stop < self.n
                ):
                    raise ValueError(
                        f"invalid out-of-range qubit {q}: \
                        index must be 0 <= qubit < {self.n} \
                    "
                    )
            return func(self, *qubits)

        return wrapper

    @staticmethod
    def _mask_index(qubits: QubitIndex) -> QubitMask:
        return np.array(qubits, dtype=np.uint64)

    @_validate_qubit
    def H(self, qubits: QubitIndex):
        """Apply Hadamard gate."""
        mask = self._mask_index(qubits)
        self.tableau[:, -1] ^= self.tableau[:, mask] & self.tableau[:, mask + self.n]
        self.tableau[:, mask], self.tableau[:, mask + self.n] = (
            self.tableau[:, mask + self.n].copy(),
            self.tableau[:, mask].copy(),
        )
        return self

    @_validate_qubit
    def S(self, qubits: QubitIndex):
        """Apply S gate."""
        mask = self._mask_index(qubits)
        self.tableau[:, -1] ^= self.tableau[:, qubits] & self.tableau[:, mask + self.n]
        self.tableau[:, mask + self.n] ^= self.tableau[:, mask]
        return self

    @_validate_qubit
    def CX(self, controls: QubitIndex, targets: QubitIndex):
        """Apply Controlled-X gate."""
        c_mask = self._mask_index(controls)
        t_mask = self._mask_index(targets)
        if len(c_mask) != len(t_mask):
            raise ValueError(
                f"dimension mismatch between control and target indexes: \
                {controls} and {targets}"
            )
        self.tableau[:, t_mask] ^= self.tableau[:, c_mask]
        self.tableau[:, self.n + c_mask] ^= self.tableau[:, self.n + t_mask]
        return self

    @_validate_qubit
    def Z(self, qubits: QubitIndex):
        """Apply Z gate."""
        self.paulis[qubits] = Pauli.Z.value
        return self.S(qubits).S(qubits)

    @_validate_qubit
    def X(self, qubits: QubitIndex):
        """Apply X gate."""
        return self.H(qubits).Z(qubits).H(qubits)

    @_validate_qubit
    def Y(self, qubits: QubitIndex):
        """Apply Y gate."""
        return self.X(qubits).Z(qubits)

    def plot(self) -> None:
        """Plot lattice."""
        L: int = self.L
        n: int = self.n
        tableau: NDArray[np.uint8] = self.tableau
        x_grid, y_grid = np.meshgrid(np.arange(L), np.arange(L))
        grid_x, grid_y = x_grid.flatten(), y_grid.flatten()
        ax = plt.gca()  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

        def draw_polygon_or_wedge(
            indices: NDArray[np.uint64],
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
            indices: np.ndarray = np.where(stab[:n] == 1)[0]  # pyright: ignore[reportAny]
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

        ops = [(op, q % L, L - q // L - 1) for op, q in self.paulis]
        for op, x, y in ops:
            _ = plt.plot(x, y, marker="o", color="red", markersize=10)
            _ = plt.text(  # pyright: ignore[reportUnknownMemberType]
                x + 0.1, y + 0.1, f"${op}$", ha="left", va="bottom"
            )

        plt.show()  # pyright: ignore[reportUnknownMemberType]
