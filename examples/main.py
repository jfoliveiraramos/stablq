# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "surfq",
# ]
#
# [tool.uv.sources]
# surfq = { path = "..", editable = true }
# ///
"""Entry-point script."""

from surfq import Lattice


def main():
    """Entry-point."""
    lattice = Lattice(7)
    _ = lattice[0, 0].X()
    _ = lattice[2, 2].Z()
    _ = lattice[:, 5].Z()
    lattice.show()


if __name__ == "__main__":
    main()
