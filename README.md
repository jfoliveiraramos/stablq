# SurfQ

**SurfQ** is a Python framework for efficiently simulating surface codes under noise. The aim is to enable detailed investigation of the behavior of surface codes under different physical error models, supporting advanced fault-tolerant protocols such as lattice surgery, state injection, and magic state distillation as future extensions.

## Features

- Intuitive surface code representation
- Apply quantum operations: X, Z, H, CNOT, and syndrome measurement
- Simulate various noise channels: Pauli noise, depolarizing, bit-flip, phase-flip errors
- Analyse logical error rates and syndrome measurement statistics for error correction
- Modular and extensible framework for surface code simulation and fault-tolerant quantum protocols

## Usage

This package is available on PyPi. To use it you can simply install on your environment with *pip*:

```shell
pip install surfq
```


Alternatively, if you are using [uv](https://docs.astral.sh/uv/) package manager, you can get started by simply:

```shell
uvx juv init notebook.ipynb
uvx juv run notebook.ipynb --with surfq
```

Then to get started in python:

```python
from surfq import Lattice

l = Lattice(5)
l[1,1:4].X()
l.show()
```

## Examples

This project contains a set of example scripts to experiment with. We recommend using [juv](https://github.com/manzt/juv) for seamless dependency management.
```shell
uv examples/main.py
```

## Notebooks

This project contains a set of notebooks to experiment with. We recommend using [juv](https://github.com/manzt/juv) for seamless dependency management.

- **Tutorial:** This notebook goes through the basics of Quantum Error Correction, Stabilizer Formalism and the Surface Code, while leveraging **surq** to make the surface code more interactive and more easily illustrated.

```shell
uvx juv run notebooks/tutorial/notebook.py
```

## References

- [Improved Simulation of Stabilizer Circuits](https://arxiv.org/pdf/quant-ph/0406196v5) – Aaronson and Gottesman, 2004
- [Stim: a fast stabilizer circuit simulator](https://arxiv.org/abs/2103.02202) – Gidney, 2021
- [STABSim: A Parallelized Clifford Simulator with Features Beyond Direct Simulation](https://arxiv.org/abs/2507.03092) – Garner et al., 2025
- [The Heisenberg Representation of Quantum Computers](https://arxiv.org/abs/quant-ph/9807006) - Gottesman, 1998
