"""Entry-point script."""

from lattice import Lattice


def main():
    """Entry-point."""
    lattice = Lattice(7)
    # _ = lattice[0:4].Y()
    _ = lattice[0, 0].X()[2, 2].Z()
    _ = lattice[:, 5].Z()
    lattice.show()


if __name__ == "__main__":
    main()
