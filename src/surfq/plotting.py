"""A module for plotting surface code lattices.

This module provides a set of functions for plotting surface code lattices,
including the qubits, the stabilizers, and the Pauli operators.

It uses the `matplotlib` library to generate the plots, and it provides a
set of custom styles and colors to make the plots easy to read and
understand.

Typical usage example:

  >>> from surfq import Lattice
  >>>
  >>> lattice = Lattice(3)
  >>> lattice.qubits[0, 0].X()
  >>> lattice.show()
"""

import importlib.resources
from logging import warning
from pathlib import Path, PosixPath

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch, Polygon, Wedge
from numpy.typing import NDArray

from .pauli import Pauli

sns.set_style("darkgrid")
font_resource = importlib.resources.files("surfq").joinpath("fonts/Roboto-Regular.ttf")

with importlib.resources.as_file(font_resource) as font_path:
    roboto_font = fm.FontProperties(fname=PosixPath(Path(font_path)))
    fm.fontManager.addfont(str(font_path))  # pyright: ignore[reportAny]
mpl.rcParams.update(  # pyright: ignore[reportUnknownMemberType]
    {
        "font.family": roboto_font.get_name(),
        "font.size": 12,
        "grid.color": "0.5",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "xtick.color": "black",
        "ytick.color": "black",
    }
)


def plot_lattice(L: int, tableau: NDArray[np.uint8], paulis: NDArray[np.uint8]) -> None:
    """Plot a surface code lattice.

    This function plots a surface code lattice, including the qubits, the
    stabilizers, and the Pauli operators.

    Args:
        L: The size of the lattice.
        tableau: The tableau of the stabilizers.
        paulis: The Pauli operators on the qubits.
    """
    n = L**2

    if L >= 20:
        warning("Lattice dimension is too big to show")
        return

    s = 60 if L <= 10 else 25
    fontsize = 10 if L <= 10 else 7

    x_grid, y_grid = np.meshgrid(np.arange(L), np.arange(L))
    grid_x, grid_y = x_grid.flatten(), y_grid.flatten()
    ax = plt.gca()  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

    def draw_polygon_or_wedge(
        indices: NDArray[np.intp],
        color: str,
    ) -> None:
        x = indices % L
        y = L - 1 - indices // L

        if len(indices) == 4:
            pts = np.column_stack((x, y))
            pts[[-2, -1]] = pts[[-1, -2]]
            patch = Polygon(
                list(pts),
                facecolor=color,
                edgecolor="k",
                linewidth=0.75,
            )
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
                facecolor=color,
                edgecolor="k",
                linewidth=0.75,
            )
        else:
            return

        ax.add_patch(patch)  # pyright: ignore[reportUnknownMemberType]

    for stab in tableau[: (n - 1) // 2]:  # pyright: ignore[reportAny]
        indices = np.where(stab[:n] == 1)[0]  # pyright: ignore[reportAny]
        if stab[-1] != 1:
            draw_polygon_or_wedge(indices, "#6188b2")
        else:
            draw_polygon_or_wedge(indices, "#D15567")

    for stab in tableau[(n - 1) // 2 :]:  # pyright: ignore[reportAny]
        indices = np.where(stab[n:-1] == 1)[0]  # pyright: ignore[reportAny]
        if stab[-1] != 1:
            draw_polygon_or_wedge(indices, "#87b1d3")
        else:
            draw_polygon_or_wedge(indices, "#E17C88")

    ax.scatter(grid_x, grid_y, s=s, color="white", edgecolor="black")  # pyright: ignore[reportUnknownMemberType]

    ax.set_aspect("equal")  # pyright: ignore[reportUnknownMemberType]
    ax.set_yticks(np.arange(L))  # pyright: ignore[reportUnknownMemberType]
    ax.set_xticks(np.arange(L))  # pyright: ignore[reportUnknownMemberType]
    ax.set_xlim(-1, L)  # pyright: ignore[reportUnknownMemberType]
    ax.set_ylim(-1, L)  # pyright: ignore[reportUnknownMemberType]
    ax.set_yticklabels(np.arange(L)[::-1])  # pyright: ignore[reportUnknownMemberType]
    ax.tick_params(axis="x", labeltop=True, labelbottom=False, labelsize=fontsize)  # pyright: ignore[reportUnknownMemberType]
    ax.tick_params(axis="y", labelsize=fontsize)  # pyright: ignore[reportUnknownMemberType]
    ax.grid(True)  # pyright: ignore[reportUnknownMemberType]

    ax.legend(  # pyright: ignore[reportUnknownMemberType]
        handles=[
            Patch(
                facecolor="#6188b2",
                edgecolor="black",
                label="X-Stabiliser",
            ),
            Patch(
                facecolor="#87b1d3",
                edgecolor="black",
                label="Z-Stabiliser",
            ),
            Patch(
                facecolor="#D15567",
                edgecolor="black",
                label="X-Stabiliser Syndrome",
            ),
            Patch(
                facecolor="#E17C88",
                edgecolor="black",
                label="Z-Stabiliser Syndrome",
            ),
        ],
        handlelength=1,
        handleheight=1,
        borderpad=0.5,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.17),
        fontsize=10,
        markerscale=2.0,
        ncol=2,
    )

    for q, op in enumerate(paulis):  # pyright: ignore[reportAny]
        if not op:
            continue
        p = Pauli(op)
        x, y = q % L, L - q // L - 1
        _ = plt.scatter(x, y, color=p.color, edgecolor="black", s=s)
        _ = plt.text(  # pyright: ignore[reportUnknownMemberType]
            x + 0.1,
            y + 0.1,
            f"$\\mathrm{{{p}}}_{{{q}}}$",
            ha="left",
            va="bottom",
            fontsize=fontsize,
        )

    plt.show()  # pyright: ignore[reportUnknownMemberType]
