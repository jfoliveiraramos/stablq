"""Pauli Frame definition."""

from enum import Enum
from functools import lru_cache


class Pauli(Enum):
    """Define Pauli Operators."""

    I = 0
    X = 1
    Z = 2
    Y = 3

    @lru_cache
    def __mul__(self, other: "Pauli") -> "Pauli":
        """Define Pauli Multiplication."""
        return Pauli(self.value ^ other.value)
