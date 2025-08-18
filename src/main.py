"""Entry-point script."""

from lattice import Lattice


def main():
    """Entry-point."""
    lattice = Lattice(5)
    _ = lattice.X(9)
    _ = lattice.X(6)
    _ = lattice.plot()


if __name__ == "__main__":
    main()
