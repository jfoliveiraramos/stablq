# StablQ

**StablQ** is a Python framework for efficiently simulating stabiliser quantum codes and their behaviour under noise. The aim is to provide a modular architecture to implement different codes, including small stabiliser codes and surface codes, and analyse their robustness under various physical error models. Future extensions include lattice surgery, state injection and magic state distillation.

Key features:
- Represent stabiliser codes as Pauli strings or check matrices
- Apply quantum operations (X, Z, H, CNOT) and measurements
- Simulate different noise channels: Pauli, depolarising, bit-flip, phase-flip
- Analyse logical error rates and syndrome statistics
- Extendable framework for surface codes and fault-tolerant protocols


## References

- [Stim: a fast stabilizer circuit simulator](https://arxiv.org/abs/2103.02202) – Gidney, 2021
- [STABSim: A Parallelized Clifford Simulator with Features Beyond Direct Simulation](https://arxiv.org/abs/2507.03092) – Garner et al., 2025
