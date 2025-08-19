# SurfQ

**SurfQ** is a Python framework for efficiently simulating surface codes under noise. The aim is to enable detailed investigation of the behavior of surface codes under different physical error models, supporting advanced fault-tolerant protocols such as lattice surgery, state injection, and magic state distillation as future extensions.

## Features:

- Intuitive surface code representation
- Apply quantum operations: X, Z, H, CNOT, and syndrome measurement
- Simulate various noise channels: Pauli noise, depolarizing, bit-flip, phase-flip errors
- Analyse logical error rates and syndrome measurement statistics for error correction
- Modular and extensible framework for surface code simulation and fault-tolerant quantum protocols

## Examples

```shell
uv run examples/main.py
```

## Notebooks

```shell
uvx juv run notebooks/main.py
```

## References

- [Improved Simulation of Stabilizer Circuits](https://arxiv.org/pdf/quant-ph/0406196v5) – Aaronson and Gottesman, 2008
- [Stim: a fast stabilizer circuit simulator](https://arxiv.org/abs/2103.02202) – Gidney, 2021
- [STABSim: A Parallelized Clifford Simulator with Features Beyond Direct Simulation](https://arxiv.org/abs/2507.03092) – Garner et al., 2025
- [The Heisenberg Representation of Quantum Computers](https://arxiv.org/abs/quant-ph/9807006) - Gottesman, 2024
