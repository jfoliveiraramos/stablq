"""Define Surface Code Lattice."""

from typing import Callable, final

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Polygon, Wedge
from numpy.typing import NDArray


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
        self.operations: list[tuple[str, int]] = []

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
        def wrapper(self: "Lattice", *qubits: int) -> "Lattice":
            for q in qubits:
                if not 0 <= q < self.n:
                    raise ValueError(
                        f"invalid out-of-range qubit {q}: \
                        index must be 0 <= qubit < {self.n} \
                    "
                    )
            return func(self, *qubits)

        return wrapper

    @_validate_qubit
    def H(self, qubit: int):
        """Apply Hadamard gate."""
        self.tableau[:, -1] ^= self.tableau[:, qubit] & self.tableau[:, qubit + self.n]
        self.tableau[:, qubit], self.tableau[:, qubit + self.n] = (
            self.tableau[:, qubit + self.n].copy(),
            self.tableau[:, qubit].copy(),
        )
        return self

    @_validate_qubit
    def S(self, qubit: int):
        """Apply S gate."""
        self.tableau[:, -1] ^= self.tableau[:, qubit] & self.tableau[:, qubit + self.n]
        self.tableau[:, qubit + self.n] ^= self.tableau[:, qubit]
        return self

    @_validate_qubit
    def CX(self, control: int, target: int):
        """Apply Controlled-X gate."""
        self.tableau[:, target] ^= self.tableau[:, control]
        self.tableau[:, self.n + control] ^= self.tableau[:, self.n + target]
        return self

    @_validate_qubit
    def Z(self, qubit: int):
        """Apply Z gate."""
        self.operations.append(("Z", qubit))
        return self.S(qubit).S(qubit)

    @_validate_qubit
    def X(self, qubit: int):
        """Apply X gate."""
        self.operations.append(("X", qubit))
        return self.H(qubit).Z(qubit).H(qubit)

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
            color_base: str,
            is_syndrome: bool,
        ) -> None:
            x = indices % L
            y = L - 1 - indices // L
            color: str = "red" if is_syndrome else color_base

            if len(indices) == 4:
                pts = np.column_stack((x, y))
                pts[[-2, -1]] = pts[[-1, -2]]
                patch = Polygon(list(pts), color=color, alpha=0.5)
            elif len(indices) == 2:
                cx, cy = float(np.mean(x)), float(np.mean(y))
                if color_base == "orange":
                    theta1, theta2 = (180, 0) if y[0] == 0 else (0, 180)
                else:  # cyan
                    theta1, theta2 = (90, -90) if x[0] == 0 else (-90, 90)
                patch = Wedge(
                    center=(cx, cy),
                    r=0.5,
                    theta1=theta1,
                    theta2=theta2,
                    color=color,
                    alpha=0.5,
                )
            else:
                return

            ax.add_patch(patch)  # pyright: ignore[reportUnknownMemberType]

        for stab in tableau[: (n - 1) // 2]:  # pyright: ignore[reportAny]
            indices: np.ndarray = np.where(stab[:n] == 1)[0]  # pyright: ignore[reportAny]
            draw_polygon_or_wedge(indices, "orange", stab[-1] == 1)  # pyright: ignore[reportAny]

        for stab in tableau[(n - 1) // 2 :]:  # pyright: ignore[reportAny]
            indices = np.where(stab[n:-1] == 1)[0]  # pyright: ignore[reportAny]
            draw_polygon_or_wedge(indices, "cyan", stab[-1] == 1)  # pyright: ignore[reportAny]

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
                    facecolor="cyan", alpha=0.5, edgecolor="black", label="Z-Stabiliser"
                ),
            ],
            handlelength=1,
            handleheight=1,
            borderpad=0.5,
            loc="upper left",
            bbox_to_anchor=(1.05, 1),
            fontsize=14,
            markerscale=2.0,
        )

        plt.show()  # pyright: ignore[reportUnknownMemberType]
