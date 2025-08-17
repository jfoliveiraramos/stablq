"""Pauli frame definition."""

from collections.abc import Iterator
from enum import Enum
from functools import cached_property, lru_cache
from typing import final, overload, override

import numpy as np
from numpy._typing import NDArray


class Pauli(Enum):
    """Single-qubit Pauli operators."""

    I = 0
    X = 1
    Z = 2
    Y = 3

    @lru_cache
    def __mul__(self, other: "Pauli") -> "Pauli":
        """Return the product with another Pauli operator."""
        return Pauli(self.value ^ other.value)

    @lru_cache
    def commutes(self, other: "Pauli") -> bool:
        """Return True if this operator commutes with another."""
        return self == other or self == Pauli.I or other == Pauli.I

    @lru_cache
    def anti_commutes(self, other: "Pauli") -> bool:
        """Return True if this operator anti-commutes with another."""
        return not self.commutes(other)


@final
class PauliOperator:
    """Multi-qubit Pauli operator."""

    def __init__(self, ops: list[Pauli] | np.ndarray):
        if isinstance(ops, list):
            self._op = np.array([op.value for op in ops], dtype=np.uint8)
        else:
            self._op = ops.astype(np.uint8)

    @property
    def values(self) -> NDArray[np.uint8]:
        """List of single-qubit Pauli operators."""
        return self._op

    @cached_property
    def n(self) -> int:
        """Number of qubits (length of the operator)."""
        return len(self._op)

    def check_dimension_match(self, other: "PauliOperator") -> None:
        """Raise ValueError if dimensions do not match."""
        if self.n != other.n:
            raise ValueError(
                f"Pauli operators {self} and {other} do not match in dimension"
            )

    @overload
    def __eq__(self, other: "PauliOperator") -> bool: ...
    @overload
    def __eq__(self, other: object) -> bool: ...

    @override
    def __eq__(self, other: object) -> bool:
        """Return True if this operator is equal to another Pauli operator."""
        if not isinstance(other, PauliOperator):
            return NotImplemented
        return np.array_equal(self.values, other.values)

    @override
    def __hash__(self) -> int:
        return hash(self.values.tobytes())

    def __iter__(self) -> Iterator[Pauli]:
        """Iterate over the single-qubit Pauli operators."""
        return (Pauli(v) for v in self.values)  # pyright: ignore[reportAny]

    def __mul__(self, other: "PauliOperator") -> "PauliOperator":
        """Return the product with another Pauli operator."""
        self.check_dimension_match(other)
        return PauliOperator(self.values ^ other.values)

    @overload
    def __getitem__(self, key: int) -> Pauli: ...

    @overload
    def __getitem__(self, key: slice) -> list[Pauli]: ...

    def __getitem__(self, key: int | slice) -> Pauli | list[Pauli]:
        """Return a single operator or a slice of operators."""
        if isinstance(key, int):
            return Pauli(self.values[key])
        else:
            return [Pauli(value) for value in self.values[key]]  # pyright: ignore[reportAny]

    @lru_cache
    def commutes(self, other: "PauliOperator") -> bool:
        """Return True if this operator commutes with another."""
        comm_flags: NDArray[np.bool_] = (  # pyright: ignore[reportAny]
            (self.values == other.values) | (self.values == 0) | (other.values == 0)
        )
        return int(np.sum(~comm_flags)) % 2 == 0

    @lru_cache
    def anti_commutes(self, other: "PauliOperator") -> bool:
        """Return True if this operator anti-commutes with another."""
        return not self.commutes(other)
