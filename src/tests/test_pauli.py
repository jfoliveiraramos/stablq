"""Test Pauli Frame."""

from pauli import Pauli, PauliOperator


def test_single_pauli_operations():
    """Test Single-Qubit Pauli operations."""
    assert Pauli.I * Pauli.I == Pauli.I
    assert Pauli.X * Pauli.I == Pauli.X
    assert Pauli.Z * Pauli.I == Pauli.Z
    assert Pauli.Y * Pauli.I == Pauli.Y
    assert Pauli.I * Pauli.X == Pauli.X
    assert Pauli.X * Pauli.X == Pauli.I
    assert Pauli.Z * Pauli.X == Pauli.Y
    assert Pauli.Y * Pauli.X == Pauli.Z
    assert Pauli.I * Pauli.Z == Pauli.Z
    assert Pauli.X * Pauli.Z == Pauli.Y
    assert Pauli.Z * Pauli.Z == Pauli.I
    assert Pauli.Y * Pauli.Z == Pauli.X
    assert Pauli.I * Pauli.Y == Pauli.Y
    assert Pauli.X * Pauli.Y == Pauli.Z
    assert Pauli.Z * Pauli.Y == Pauli.X
    assert Pauli.Y * Pauli.Y == Pauli.I

    assert Pauli.X.commutes(Pauli.X)
    assert Pauli.Y.commutes(Pauli.Y)
    assert Pauli.Z.commutes(Pauli.Z)
    assert Pauli.I.commutes(Pauli.X)
    assert Pauli.I.commutes(Pauli.Y)
    assert Pauli.I.commutes(Pauli.Z)

    assert not Pauli.X.commutes(Pauli.Z)
    assert not Pauli.Z.commutes(Pauli.X)
    assert not Pauli.X.commutes(Pauli.Y)
    assert not Pauli.Y.commutes(Pauli.X)
    assert not Pauli.Y.commutes(Pauli.Z)
    assert not Pauli.Z.commutes(Pauli.Y)

    assert (Pauli.X * Pauli.Y) * Pauli.Z == Pauli.X * (Pauli.Y * Pauli.Z)
    assert (Pauli.Z * Pauli.Y) * Pauli.X == Pauli.Z * (Pauli.Y * Pauli.X)
    assert (Pauli.X * Pauli.Z) * Pauli.Y == Pauli.X * (Pauli.Z * Pauli.Y)

    assert Pauli.X == Pauli.X
    assert Pauli.X != Pauli.Y
    assert not (Pauli.X == Pauli.Y)


def test_multi_pauli_operations():
    """Test multi-qubit Pauli operations."""
    op = PauliOperator([Pauli.X, Pauli.Y, Pauli.Z])
    assert op * PauliOperator([Pauli.I, Pauli.I, Pauli.I]) == op

    op1 = PauliOperator([Pauli.X, Pauli.Z, Pauli.Y])
    op2 = PauliOperator([Pauli.X, Pauli.Z, Pauli.Y])
    assert op1 * op2 == PauliOperator([Pauli.I, Pauli.I, Pauli.I])

    assert PauliOperator([Pauli.X, Pauli.Y]) == PauliOperator([Pauli.X, Pauli.Y])
    assert PauliOperator([Pauli.X, Pauli.Y]) != PauliOperator([Pauli.Y, Pauli.X])

    op3 = PauliOperator([Pauli.X, Pauli.I])
    op4 = PauliOperator([Pauli.Z, Pauli.I])
    assert not op3.commutes(op4)
    assert op3.commutes(PauliOperator([Pauli.X, Pauli.I]))
    assert op3.commutes(PauliOperator([Pauli.I, Pauli.I]))
    assert PauliOperator([Pauli.X, Pauli.Z]).commutes(PauliOperator([Pauli.Z, Pauli.X]))

    op5 = PauliOperator([Pauli.X, Pauli.Y, Pauli.Z])
    assert op5[0] == Pauli.X
    assert op5[1] == Pauli.Y

    assert op5[0:2] == [Pauli.X, Pauli.Y]
    assert op5[1:] == [Pauli.Y, Pauli.Z]

    collected = [p for p in op5]
    assert collected == [Pauli.X, Pauli.Y, Pauli.Z]
