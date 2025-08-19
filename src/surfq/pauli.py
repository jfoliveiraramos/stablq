"""Pauli frame definition."""

from enum import Enum
from functools import cached_property
from typing import override


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

    @cached_property
    def color(self) -> str:
        """Define color per Pauli operator for plotting."""
        match self:
            case Pauli.I:
                return "#FFFFFF"  # white for identity
            case Pauli.Z:
                return "#D15567"  # muted purple, complements red/pink
            case Pauli.X:
                return "#E17C88"  # muted red, in tone with syndrome reds
            case Pauli.Y:
                return "#F28EBF"  # softer pink, between X and Z
