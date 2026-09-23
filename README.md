# MatchCake

<p align="center"><img src="https://github.com/MatchCake/MatchCake/blob/main/images/logo/Logo.svg?raw=true" width="60%" /></p>
<p align="center"><sub>Logo by Tarik El-Khateeb / Xanadu</sub></p>

[![Star on GitHub](https://img.shields.io/github/stars/MatchCake/MatchCake.svg?style=social)](https://github.com/MatchCake/MatchCake/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/MatchCake/MatchCake?style=social)](https://github.com/MatchCake/MatchCake/network/members)
[![Python 3.11 to 3.14](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/downloads/)
[![downloads](https://img.shields.io/pypi/dm/MatchCake)](https://pypi.org/project/MatchCake)
[![PyPI version](https://img.shields.io/pypi/v/MatchCake)](https://pypi.org/project/MatchCake)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/MatchCake/MatchCake/blob/dev/LICENSE)
[![status](https://joss.theoj.org/papers/91f77a47cbd519daac9794a1d2144361/status.svg)](https://joss.theoj.org/papers/91f77a47cbd519daac9794a1d2144361)
[![DOI](https://zenodo.org/badge/699543422.svg)](https://doi.org/10.5281/zenodo.22917735)

![Tests Workflow](https://github.com/MatchCake/MatchCake/actions/workflows/tests.yml/badge.svg)
![Dist Workflow](https://github.com/MatchCake/MatchCake/actions/workflows/build_dist.yml/badge.svg)
![Doc Workflow](https://github.com/MatchCake/MatchCake/actions/workflows/docs.yml/badge.svg)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![codecov](https://codecov.io/github/MatchCake/MatchCake/branch/main/graph/badge.svg?token=Yz44IcMdVx)](https://codecov.io/github/MatchCake/MatchCake)


# Description

MatchCake is a Python package that provides a new PennyLane device for simulating a specific class of quantum
circuits called Matchgate circuits or matchcircuits. These circuits are made with matchgates, a class of restricted
quantum unitaries that are parity-preserving and operate on nearest-neighbor qubits. These constraints lead to
matchgates being classically simulable in polynomial time.

Additionally, this package provides quantum kernels made with [scikit-learn](https://scikit-learn.org/stable/) API allowing the
use matchcircuits as kernels in quantum machine learning algorithms. One way to use these kernels could be in a Support Vector
Machine (SVM).

Note that this package is built on PennyLane and PyTorch. This means that only the NumPy and PyTorch backends are compatible.
Other backends provided by Autoray, such as JAX and TensorFlow, are not supported.
We highly recommend using PyTorch as the backend when working with MatchCake.


# Requirements

| Requirement   | Supported                  |
|---------------|----------------------------|
| **Python**    | 3.11, 3.12, 3.13, 3.14     |
| **Platforms** | Linux, Windows             |

Every one of these Python versions is tested on Linux (x86-64 and aarch64) and on Windows on each pull
request, in [continuous integration](https://github.com/MatchCake/MatchCake/actions/workflows/tests.yml).
On Linux, a glibc based distribution is required. PyTorch publishes no musl wheels, so MatchCake cannot be
installed on Alpine.

macOS is not supported. MatchCake contains no platform specific code and may well run there, but it
is not tested. If you need macOS support, please
[open an issue](https://github.com/MatchCake/MatchCake/issues/new/choose) and we will look into
adding it to the continuous integration matrix.


# Installation

MatchCake is published on PyPI as `matchcake`.

### Set up a project or an environment

Install MatchCake into a virtual environment and not into the system interpreter, which recent Linux
distributions refuse with `error: externally-managed-environment`. If you do not have one yet, create it:

```bash
uv init myproject && cd myproject                    # uv
poetry new myproject && cd myproject                 # poetry (then cap requires-python, see below)
python -m venv .venv && source .venv/bin/activate    # plain virtual environment
```

On Windows, activate the virtual environment with `.venv\Scripts\activate` instead.

### Install MatchCake

| Method     | Commands                                                          |
|------------|-------------------------------------------------------------------|
| **pip**    | `pip install matchcake`                                           |
| **uv**     | `uv add matchcake` in a uv project, or `uv pip install matchcake` |
| **poetry** | `poetry add matchcake` in a poetry project                        |
| **source** | `pip install "git+https://github.com/MatchCake/MatchCake@main"`   |

A few things useful to know during the installation:

- `uv add` and `poetry add` are project commands: they record a dependency in the `pyproject.toml` of the
  project you are standing in. In a directory without one they stop with
  `No pyproject.toml found in current directory or any parent directory`. Note that `uv add` also searches
  parent directories, so running it inside an unrelated project modifies that project. Use
  `uv pip install matchcake` when you only want the package in an environment.
- `poetry add matchcake` additionally requires your project to put an upper bound on its Python range, for
  example `requires-python = ">=3.11,<3.15"`. `poetry new` writes an open ended range such as `>=3.12`, and
  poetry then declines to resolve because `torchpfaffian` supports Python `<3.15` only.
- The `source` row needs `git` available on your `PATH`. It installs the latest release from the `main`
  branch; see below for the development branch.


### Last unstable version

To install the development branch, use the `@dev` reference:

```bash
pip install "git+https://github.com/MatchCake/MatchCake@dev"
```

The uv equivalent is `uv add "matchcake @ git+https://github.com/MatchCake/MatchCake@dev"`.


### PyTorch build and installation size

MatchCake depends on PyTorch, and on Linux the default PyTorch wheel on PyPI is the CUDA build. A plain
`pip install matchcake` therefore downloads PyTorch together with around twenty NVIDIA packages and occupies
roughly 6 GB, whether or not the machine has a GPU. This comes from how PyTorch is packaged, not from
MatchCake.

For a CPU only environment, install PyTorch from the PyTorch CPU index first, then install MatchCake on top.
The requirement is already satisfied at that point, so nothing pulls the CUDA build in, and the result is
about 1.6 GB with no NVIDIA packages:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install matchcake
```

Swap the index for a specific CUDA version, for instance `https://download.pytorch.org/whl/cu128` for
CUDA 12.8 or `https://download.pytorch.org/whl/cu130` for CUDA 13.0.

With uv, declare the index in your own `pyproject.toml` and add `torch` alongside `matchcake`:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
```

```bash
uv add matchcake torch
```

Listing `torch` explicitly is important: `[tool.uv.sources]` only redirects dependencies that your own project
declares, so a `torch` pulled in solely through `matchcake` still comes from PyPI.

MatchCake declares `cpu`, `cu128` and `cu130` extras, but they only select these indexes when MatchCake
itself is built from a clone, through `[tool.uv.sources]` in its `pyproject.toml`. That mechanism belongs to
uv and is not part of the metadata published to PyPI, so `pip install "matchcake[cpu]"` and
`uv add matchcake --extra cpu` resolve exactly the same PyTorch as no extra at all. Use the index based
recipe above. The extras are documented for contributors in
[CONTRIBUTING.md](https://github.com/MatchCake/MatchCake/blob/dev/CONTRIBUTING.md).


# Quick Usage Preview

## Quantum Circuit Simulation with MatchCake
```python
import matchcake as mc
import pennylane as qml
import numpy as np
from pennylane.ops.qubit.observables import BasisStateProjector

# Create a Non-Interacting Fermionic Device
nif_device = mc.NonInteractingFermionicDevice(wires=4)
initial_state = np.zeros(len(nif_device.wires), dtype=int)

# Define a quantum circuit
def circuit(params, wires, initial_state=None):
    qml.BasisState(initial_state, wires=wires)
    for i, even_wire in enumerate(wires[:-1:2]):
        idx = list(wires).index(even_wire)
        curr_wires = [wires[idx], wires[idx + 1]]
        mc.operations.CompRxRx(params, wires=curr_wires)
        mc.operations.CompRyRy(params, wires=curr_wires)
        mc.operations.CompRzRz(params, wires=curr_wires)
    for i, odd_wire in enumerate(wires[1:-1:2]):
        idx = list(wires).index(odd_wire)
        mc.operations.fSWAP(wires=[wires[idx], wires[idx + 1]])
    projector: BasisStateProjector = qml.Projector(initial_state, wires=wires)
    return qml.expval(projector)

# Create a QNode
nif_qnode = qml.QNode(circuit, nif_device)
qml.draw_mpl(nif_qnode)(np.array([0.1, 0.2]), wires=nif_device.wires, initial_state=initial_state)

# Evaluate the QNode
expval = nif_qnode(np.random.random(2), wires=nif_device.wires, initial_state=initial_state)
print(f"Expectation value: {expval}")
```

## Data Classification with MatchCake

```python
from matchcake.ml.kernels import FermionicPQCKernel
from matchcake.ml.visualisation import ClassificationVisualizer
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Load the iris dataset
X, y = datasets.load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and fit the model
pipeline = Pipeline([
    ('scaler', MinMaxScaler(feature_range=(0, 1))),
    ('kernel', FermionicPQCKernel(n_qubits=4, rotations="X,Z").freeze()),
    ('classifier', SVC(kernel='precomputed')),
])
pipeline.fit(x_train, y_train)

# Evaluate the model
test_accuracy = pipeline.score(x_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Visualize the classification
viz = ClassificationVisualizer(x=X, n_pts=1_000)
viz.plot_2d_decision_boundaries(model=pipeline, y=y, show=True)

```


# Tutorials
- [MatchCake Basics](https://github.com/MatchCake/MatchCake/blob/main/tutorials/matchcake_basics.ipynb)
- [Compute Expectation Values with MatchCake](https://github.com/MatchCake/MatchCake/blob/main/tutorials/expectation_values.ipynb)
- [Iris Classification with MatchCake](https://github.com/MatchCake/MatchCake/blob/main/tutorials/iris_classification.ipynb)
- [Nystroem Kernel Approximation](https://github.com/MatchCake/MatchCake/blob/main/tutorials/nystroem_kernel_approximation.ipynb)


# For Developers

To contribute to the development of MatchCake, please refer to the [contributing guidelines](https://github.com/MatchCake/MatchCake/blob/dev/CONTRIBUTING.md).



# Notes
- This package is still in development and some features may not be available yet.
- The documentation is still in development and may not be complete yet.



# About
This work was supported by the Ministère de l'Économie, de l'Innovation et de l'Énergie du Québec
through its Research Chair in Quantum Computing, an NSERC Discovery grant, and the Canada First Research Excellence
Fund.


# Important Links
- Documentation at [https://MatchCake.github.io/MatchCake/](https://MatchCake.github.io/MatchCake/).
- Github at [https://github.com/MatchCake/MatchCake/](https://github.com/MatchCake/MatchCake/).




# Found a bug or have a feature request?
- [Click here to create a new issue.](https://github.com/MatchCake/MatchCake/issues/new/choose)



# License
[Apache License 2.0](https://github.com/MatchCake/MatchCake/blob/dev/LICENSE)



# Citations

Software, archived on [Zenodo](https://doi.org/10.5281/zenodo.22917737) (this DOI resolves to the latest version):
```
@software{matchcake_Gince2026,
  title={MatchCake: A Python Simulator for Non-Interacting Fermionic Quantum Circuits with Machine Learning Applications},
  author={Gince, Jérémie},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.22917737},
  url={https://doi.org/10.5281/zenodo.22917737},
}
```

## Fermionic Machine Learning Paper

[Fermionic Machine Learning](https://ieeexplore.ieee.org/document/10821385) is a work presented at the 2024 IEEE
International Conference on Quantum Computing and Engineering (QCE). The paper compares unconstrained quantum kernel
methods with constraint-based kernels derived from matchgate (free-fermionic) circuits, and benchmarks their
performance on supervised classification tasks. All free-fermionic kernels considered in this work were simulated
using MatchCake.

[IEEE Xplore paper](https://ieeexplore.ieee.org/document/10821385):
```
@INPROCEEDINGS{10821385,
  author={Gince, Jérémie and Pagé, Jean-Michel and Armenta, Marco and Sarkar, Ayana and Kourtis, Stefanos},
  booktitle={2024 IEEE International Conference on Quantum Computing and Engineering (QCE)},
  title={Fermionic Machine Learning},
  year={2024},
  volume={01},
  number={},
  pages={1672-1678},
  keywords={Runtime;Quantum entanglement;Computational modeling;Benchmark testing;Rendering (computer graphics);Hardware;Kernel;Integrated circuit modeling;Quantum circuit;Standards;Quantum machine learning;quantum kernel methods;matchgate circuits;fermionic quantum computation;data classification},
  doi={10.1109/QCE60285.2024.00195}
}
```


[ArXiv paper](https://arxiv.org/abs/2404.19032):
```
@misc{gince2024fermionic,
      title={Fermionic Machine Learning},
      author={Jérémie Gince and Jean-Michel Pagé and Marco Armenta and Ayana Sarkar and Stefanos Kourtis},
      year={2024},
      eprint={2404.19032},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}
```
