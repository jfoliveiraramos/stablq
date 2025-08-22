# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "surfq",
# ]
#
# [tool.uv.sources]
# surfq = { path = "..", editable = true }
# ///
"""Entry-point script."""

import math
from typing import Literal

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from surfq import Lattice


def main():
    """Entry-point."""
    L = 11
    # lattice = Lattice(L)
    # _ = lattice[1 : L - 1, 1].X()
    # _ = lattice[[0, L - 1], 3].Z()
    # lattice.show()
    # _ = lattice[4, 2].Z()
    # _ = lattice[0, 0].Z()
    # lattice.show()
    # _ = lattice[[0, L - 1], 1].Z()
    # _ = lattice[[0, L - 1], 3].Z()
    # lattice.show()
    #
    # lattice = Lattice(L)
    # _ = lattice[0, [1, 2]].X()
    # _ = lattice[L - 1, 2].X()
    # _ = lattice[L - 2, 3].X()
    # lattice.show()
    #
    # lattice = Lattice(L)
    # _ = lattice[[0, L - 1], 1].Z()
    # _ = lattice[[0, L - 1], 3].Z()
    # lattice.show()
    # _ = lattice[[0, L - 1], 1].Z()
    # _ = lattice[[0, L - 1], 3].Z()
    # lattice.show()

    lattice = Lattice(L)
    # _ = lattice[0, [0, 3]].Z()
    # _ = lattice[L - 1, [2, 4]].Z()
    _ = lattice[[(4, 3), (5, 4), (6, 5), (6, 6)]].X()
    _ = lattice[[(3, 8), (4, 8)]].X()
    lattice.show()

    G: nx.Graph[int] = nx.Graph()
    indices = np.where(lattice.tableau[:, -1])[0]
    c: NDArray[np.float64] = lattice.stabilisers_coords[indices]
    X_edges = [
        (indices[i], indices[j], distance(L, c[i], c[j], "X"))
        for j in range(len(c))
        for i in range(j)
    ]
    print(X_edges)
    G.add_weighted_edges_from(X_edges)  # pyright: ignore[reportUnknownMemberType]
    matching = list(nx.algorithms.min_weight_matching(G))
    import matplotlib.pyplot as plt

    print(np.array(matching))
    for s1, s2 in matching:
        print(lattice.stabilisers_coords[s1])
        plt.plot(
            [lattice.stabilisers_coords[s1][0], lattice.stabilisers_coords[s2][0]],
            [
                L - lattice.stabilisers_coords[s1][1] - 1,
                L - lattice.stabilisers_coords[s2][1] - 1,
            ],
            color="k",
            marker="o",
            markerfacecolor="white",
        )
    lattice.show()
    print(matching)

    for s1, s2 in matching:
        c1, c2 = lattice.stabilisers_coords[[s1, s2]]

        # Compute integer bounds for coordinates
        x1, x2 = (
            (math.ceil(c1[0]), math.floor(c2[0]))
            if c1[0] < c2[0]
            else (math.floor(c1[0]), math.ceil(c2[0]))
        )
        y1, y2 = (
            (math.ceil(c1[1]), math.floor(c2[1]))
            if c1[1] < c2[1]
            else (math.floor(c1[1]), math.ceil(c2[1]))
        )

        dx, dy = abs(x1 - x2), abs(y1 - y2)

        if dx <= dy:
            # Move along x first, then y
            qd = (
                np.arange(min(x1, x2), max(x1, x2))
                + np.arange(min(y1, y2), min(y1, y2) + dx) * L
            )
            arr = max(x1, x2) + np.arange(min(y1, y2) + dx, max(y1, y2) + 1) * L
        else:
            # Move along y first, then x
            qd = np.arange(min(y1, y2), max(y1, y2)) * L + np.arange(
                min(x1, x2), min(x1, x2) + dy
            )
            arr = max(y1, y2) * L + np.arange(min(x1, x2) + dy, max(x1, x2) + 1)

        q = np.concatenate([qd, arr])

        # Plot the path

        x, y = q % L, L - 1 - q // L
        plt.plot(
            x,
            y,
            linestyle="--",
            color="k",
            marker="o",
            markerfacecolor="#9467bd",
            markeredgecolor="black",
        )

        # Apply the X operation on the lattice
        lattice[q].X()

    for s in set(indices) - {u for edge in matching for u in edge}:
        sc = lattice.stabilisers_coords[s]
        if sc[0] <= 0.5:
            lattice[0, 0 if sc[1] <= 0 else math.floor(sc[1])].X()
        elif sc[0] >= L - 1.5:
            lattice[L - 1, 0 if sc[1] <= 0 else math.floor(sc[1])].X()
        elif sc[1] <= 0.5:
            lattice[0 if sc[1] <= 0 else math.floor(sc[1]), 0].X()
        elif sc[1] >= L - 1.5:
            lattice[0 if sc[1] <= 0 else math.floor(sc[1]), L - 1].X()
    lattice.show()
    lattice.show()


def distance(L, c1, c2, stab: Literal["X"] | Literal["Z"]):
    dx = np.abs((c1[0] - c2[0]) * (np.abs(c1[0] - c2[0]) > 1))
    dx = dx if stab != "X" else min(dx, L - dx)
    dy = np.abs((c1[1] - c2[1]) * (np.abs(c1[1] - c2[1]) > 1))
    dy = min(dy, np.inf if stab != "Z" else L - dy)
    return np.sqrt(dx**2 + dy**2)


if __name__ == "__main__":
    main()
