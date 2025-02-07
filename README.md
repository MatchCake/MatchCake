# MatchCake

<div style="text-align:center"><img src="https://github.com/MatchCake/MatchCake/blob/main/images/logo/Logo.svg?raw=true" width="40%" /></div>

[![Star on GitHub](https://img.shields.io/github/stars/MatchCake/MatchCake.svg?style=social)](https://github.com/MatchCake/MatchCake/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/MatchCake/MatchCake?style=social)](https://github.com/MatchCake/MatchCake/network/members)
[![Python 3.6](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![downloads](https://img.shields.io/pypi/dm/MatchCake)](https://pypi.org/project/MatchCake)
[![PyPI version](https://img.shields.io/pypi/v/MatchCake)](https://pypi.org/project/MatchCake)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

![Tests Workflow](https://github.com/MatchCake/MatchCake/actions/workflows/tests.yml/badge.svg)
![Dist Workflow](https://github.com/MatchCake/MatchCake/actions/workflows/build_dist.yml/badge.svg)
![Doc Workflow](https://github.com/MatchCake/MatchCake/actions/workflows/docs.yml/badge.svg)
![Publish Workflow](https://github.com/MatchCake/MatchCake/actions/workflows/publish.yml/badge.svg)
![Code coverage](https://raw.githubusercontent.com/MatchCake/MatchCake/coverage-badge/coverage.svg)



# Description

MatchCake is a Python package that provides a new PennyLane device for simulating a specific class of quantum 
circuits called Matchgate circuits or matchcircuits. These circuits are made with matchgates, a class of restricted 
quantum unitaries that are parity-preserving and operate on nearest-neighbor qubits. These constraints lead to 
matchgates being classically simulable in polynomial time.

Additionally, this package provides quantum kernels made with [scikit-learn](https://scikit-learn.org/stable/) API allowing the 
use matchcircuits as kernels in quantum machine learning algorithms. One way to use these kernels could be in a Support Vector 
Machine (SVM). In the [benchmark/classification](benchmark/classification/README.md) folder, you can 
find some scripts that use SVM with matchcircuits as a kernel to classify the Iris dataset, the Breast Cancer dataset, 
and the Digits dataset in polynomial time with high accuracy.



# Installation

| Method     | Commands                                                                                                                                                                     |
|------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **poetry** | `poetry install`                                                                                                                                                             |
| **PyPi**   | `pip install MatchCake`                                                                                                                                                      |
| **source** | `pip install git+https://github.com/MatchCake/MatchCake`                                                                                                                     |
| **wheel**  | 1.Download the .whl file [here](https://github.com/MatchCake/MatchCake/tree/main/dist);<br> 2. Copy the path of this file on your computer; <br> 3. `pip install [path].whl` |


### Last unstable version
To install the latest unstable version, download the latest version of the .whl file and follow the instructions above.



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
        mc.operations.fRXX(params, wires=curr_wires)
        mc.operations.fRYY(params, wires=curr_wires)
        mc.operations.fRZZ(params, wires=curr_wires)
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
from matchcake.ml.svm import FixedSizeSVC
from matchcake.ml.visualisation import ClassificationVisualizer
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load the iris dataset
X, y = datasets.load_iris(return_X_y=True)
X = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and fit the model
model = FixedSizeSVC(kernel_cls=FermionicPQCKernel, kernel_kwargs=dict(size=4), random_state=0)
model.fit(x_train, y_train)

# Evaluate the model
test_accuracy = model.score(x_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Visualize the classification
viz = ClassificationVisualizer(x=X, n_pts=1_000)
viz.plot_2d_decision_boundaries(model=model, y=y, show=True)

```


# Tutorials
- [MatchCake Basics](https://github.com/MatchCake/MatchCake/blob/main/tutorials/matchcake_basics.ipynb)
- [Iris Classification with MatchCake](https://github.com/MatchCake/MatchCake/blob/main/tutorials/iris_classification.ipynb)



# Notes
- This package is still in development and some features may not be available yet.
- The documentation is still in development and may not be complete yet.
- The package works only with the forked version of PennyLane available 
[here](https://github.com/JeremieGince/pennylane.git@v0_32_0-dev1). In the future, the package will be compatible with
the official version of PennyLane.



# About
This work was supported by the Ministère de l'Économie, de l'Innovation et de l'Énergie du Québec
through its Research Chair in Quantum Computing, an NSERC Discovery grant, and the Canada First Research Excellence 
Fund.


# Important Links
- Documentation at [https://MatchCake.github.io/MatchCake/](https://MatchCake.github.io/MatchCake/).
- Github at [https://github.com/MatchCake/MatchCake/](https://github.com/MatchCake/MatchCake/).




# Found a bug or have a feature request?
- [Click here to create a new issue.](https://github.com/MatchCake/MatchCake/issues/new)



# License
[Apache License 2.0](LICENSE)



# Citation
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

Repository:
```
@misc{matchcake_Gince2023,
  title={Fermionic Machine learning},
  author={Jérémie Gince},
  year={2023},
  publisher={Université de Sherbrooke},
  url={https://github.com/MatchCake/MatchCake},
}
```

