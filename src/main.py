"""Entry-point script."""

from lattice import Lattice


def main():
    """Entry-point."""
    lattice = Lattice(3)
    _ = lattice.plot()


if __name__ == "__main__":
    main()
