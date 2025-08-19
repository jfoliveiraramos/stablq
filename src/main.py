"""Entry-point script."""

from lattice import Lattice


def main():
    """Entry-point."""
    lattice = Lattice(7)
    _ = lattice[0:4].Y()
    _ = lattice[5, :].X()
    _ = lattice[:, 5].Z()
    lattice.plot()


if __name__ == "__main__":
    main()
