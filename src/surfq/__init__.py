"""A library for simulating the surface code.

This library provides a set of tools for simulating the behavior of the surface code.

It includes classes for representing the lattice, the qubits, and the
stabilizers, as well as functions for applying quantum gates and simulating
errors.

Typical usage example:

  >>> from surfq import Lattice
  >>>
  >>> lattice = Lattice(3)
  >>> lattice.qubits[0, 0].X()
  >>> lattice.qubits[0, 1].Z()
  >>> lattice.show()
"""

from .lattice import Lattice

__all__ = ["Lattice"]
