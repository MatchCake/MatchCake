---
title: 'MatchCake: A Python Simulator for Non-Interacting Fermionic Quantum Circuits with Machine Learning Applications'
tags:
  - quantum computing
  - machine learning
  - quantum circuit simulation
  - non-interacting fermions
  - matchgates
  - python
authors:
  - name: Jérémie Gince
    orcid: 0009-0002-7179-375X
    affiliation: 1
affiliations:
  - name: Institut Quantique & Département de Physique, Université de Sherbrooke, Sherbrooke, QC J1K 2R1, Canada
    index: 1
date: "`r format(Sys.time(), '%d %B, %Y')`"
bibliography: paper.bib
header-includes: |
  <script>
  window.MathJax = {
    loader: {load: ['[tex]/physics']},
    tex: {packages: {'[+]': ['physics']}}
  };
  </script>
  <script type="text/javascript" src="cdn.jsdelivr.net"></script>
  \newcommand{\Pf}{\mathrm{Pf}}
  \newcommand{\del}[2]{\frac{\partial #1}{\partial #2}}
  \newcommand{\delp}[1]{\frac{\partial }{\partial #1}}
  \newcommand{\ddfrac}[2]{\frac{\dd #1}{\dd #2}}
  \newcommand{\ddfracp}[1]{\frac{\dd }{\dd #1}}
  \newcommand{\braOket}[3]{\left\langle#1\left|#2\right|#3\right\rangle}
  \newcommand{\Ket}[1]{\left|#1\right\rangle}
  \newcommand{\Proj}[1]{\left|#1\right\rangle\left\langle#1\right|}
  \newcommand{\bigO}{\mathcal{O}}
---

# Summary

We introduce `MatchCake`, a Python package that simulates matchgate quantum circuits (matchcircuits). Matchgates are a
parity-preserving class of nearest-neighbour unitaries whose dynamics are equivalent to those of non-interacting
Majorana fermions, making matchcircuits classically simulable in polynomial time while retaining rich quantum structure.
Built on PennyLane [@bergholm2022pennylane] and PyTorch [@Ansel_PyTorch_2_Faster_2024], `MatchCake` brings automatic
differentiation to this class of circuits, enabling research on classically simulable quantum machine learning models.
We demonstrate it on quantum kernel methods for classification [@10821385] and on the simulation of physical systems.

# Statement of need

Quantum circuits describe the discrete time evolution of a quantum state under a sequence of gates, and are the backbone
of quantum algorithms and simulations. Since quantum hardware is still maturing, efficiently simulating and analysing
such circuits on classical computers is essential, both for designing quantum algorithms and for mapping the boundary
between classical and quantum computational power. In general, the state space of an $N$-qubit system has
dimension $\bigO(2^N)$, so generic simulators scale exponentially in the number of qubits. Efficient simulation is
possible only for restricted circuit families whose states or evolutions admit compact, polynomial-size representations.
Matchcircuits are one such family.

A two-qubit matchgate written in the computational basis

\begin{align}
M(A,W) &= \left[
\begin{matrix}
a & 0 & 0 & b \\
0 & w & x & 0 \\
0 & y & z & 0 \\
c & 0 & 0 & d
\end{matrix}
\right],
\end{align}

is a parity-preserving nearest-neighbour unitary gate defined by two submatrices

\begin{align}
A = \left[
\begin{matrix}
a & b\\
c & d\\
\end{matrix}
\right],
W = \left[
\begin{matrix}
w & x\\
y & z
\end{matrix}
\right],
\end{align}

subject to the constraint $\det(A) = \det(W)$. On a linear register, such matchcircuits evolve as non-interacting
Majorana fermions [@nielsen2005fermionic; @dai_extracting_2015]. Recent work [@projansky2025gaussianity] connects them
to Clifford circuits through fermionic Gaussian operations, framing matchcircuits as a natural generalization that
interpolates between stabilizer dynamics and more general fermionic evolutions. Their extensions are also practically
relevant: augmenting matchcircuits with $ZZ$ interactions enables efficient simulation of interaction-limited
Fermi–Hubbard systems [@mocherla2023extending]. More broadly, matchgate circuits have found applications spanning
quantum thermodynamics [@PhysRevA.98.012309], quantum machine learning (QML) [@laar2022quantum; @10821385], and
many-body
simulation.

# State of the field

Simulators differ by the circuit class they handle efficiently. Stim [@gidney2021stim] targets stabilizer (Clifford)
circuits with $\bigO(N^2)$ complexity, while quimb [@gray2018quimb] uses tensor-network methods [@biamonte_tensor_2017]
such as matrix product states to simulate low-entanglement circuits at a cost of $\bigO(N\chi^3)$ in the bond
dimension $\chi$. PennyLane [@bergholm2022pennylane] is a widely used Python framework for quantum circuit simulation
and machine learning, supporting many backends and hardware interfaces, but it lacks efficient native support for
matchgate circuits.

Its universal state-vector simulators (SVS) handle arbitrary circuits but scale exponentially in $N$, and its wrappers
around
specialized backends such as Stim and quimb do not target matchgates (matchcircuits [@valiant_quantum_2001]). Users are
therefore forced onto exponentially scaling simulators, even though matchcircuits admit polynomial-time $\bigO(N^3)$
classical simulation [@brod_efficient_2016; @bravyi_contraction_2008; @PhysRevA.102.052604]. Unlike tensor-network
methods, whose efficiency relies on limited entanglement, matchcircuits derive their tractability from an underlying
free-fermionic (Gaussian) structure and so support arbitrarily large entanglement. And unlike the discrete Clifford
gate set, matchgates [@jozsa_matchgates_2008] form a continuously parametrized family of two-qubit gates corresponding
to free-fermionic operations. `MatchCake` fills the remaining tooling gap with a dedicated matchcircuit simulator
integrated with
PennyLane and PyTorch, whose automatic differentiation supports learning-based models that exploit matchgate dynamics.

# Software design

Using `MatchCake`, the user provides a matchcircuit $C$ (a list of matchgates), a product initial
state $\Ket{\psi_\text{initial}}$, and an observable

\begin{align}
\mathcal{F} = \sum\limits_j \beta_j F_j
\end{align}

with real coefficients $\beta_j$, whose terms are computational-basis projectors or Pauli words,

\begin{align}
F_j \in \left\{\Proj{y}: y \in \{0,1\}^N \right\} \cup \left\{ P \right\},
\end{align}

where $\Ket{y}$ is a computational-basis state and $P$ any Pauli word on the $N$-qubit register. `MatchCake` then
computes, in time polynomial in $N$, the expectation value

\begin{align}
\text{Output} = \braOket{\psi_\text{initial}}{C^\dagger \mathcal{F} C}{\psi_\text{initial}}.
\end{align}

As shown in \autoref{fig:matchgate-algorithm}, $\mathcal{F}$ is evaluated term by term. The circuit $C$ is mapped to
its transition matrix $R_C$ [@brod_efficient_2016], and each term is Majorana-decomposed through the Jordan–Wigner
transformation [@jordan_uber_1928; @nielsen2005fermionic]. When $\Ket{\psi_\text{initial}}$ is a computational-basis
state and $F_j$ is diagonal in that basis, the per-term expectation $\varepsilon_j$ is read from a lookup table of
Wick's contractions [@wick_evaluation_1950; @brod_efficient_2016]; otherwise it is obtained from the covariance matrix
of the circuit-evolved state. Both routes reduce to Pfaffian computation, and the per-term values are accumulated
as $\sum_j \beta_j \varepsilon_j$.

![
The
`MatchCake` algorithm, which evaluates the expectation value $\left\langle\mathcal{F}\right\rangle = \sum_j \beta_j \varepsilon_j$ of the observable $\mathcal{F}$, one Pauli string $F_j$ at a time. When the initial state $\Ket{\psi_\text{initial}}$ is a computational-basis state
and $F_j$ is diagonal in the computational basis (Yes), the per-term expectation $\varepsilon_j$ is read from a
lookup table; otherwise (No), it is obtained from the covariance matrix of the circuit-evolved state. Both routes
share a common Pfaffian evaluation. \label{fig:matchgate-algorithm}
](images/matchcake-algorithm-detailed-v2.svg){
width="55%" fig-env="figure" fig-align="center"
}

## Minimal Example - Quantum Circuit Simulation

The following example builds a matchcircuit and computes the expectation value of a computational-basis projector after
applying the circuit to an initial state.

```python
import matchcake as mc
import pennylane as qml
import numpy as np
from pennylane.ops.qubit.observables import BasisStateProjector

# Create a Non-Interacting Fermionic Device with 4 qubits/wires
nif_device = mc.NonInteractingFermionicDevice(wires=4)
# Define the initial state as the all-zero computational basis state: |0000>
initial_state = np.zeros(len(nif_device.wires), dtype=int)


# Define a quantum circuit
def circuit(params, wires, initial_state):
    # Prepare the initial state
    qml.BasisState(initial_state, wires=wires)
    for i, even_wire in enumerate(wires[:-1:2]):
        idx = list(wires).index(even_wire)
        curr_wires = [wires[idx], wires[idx + 1]]
        # Apply the matchgate M(Rx(params), Rx(params))
        mc.operations.CompRxRx(params, wires=curr_wires)
        # Apply the matchgate M(Ry(params), Ry(params))
        mc.operations.CompRyRy(params, wires=curr_wires)
        # Apply the matchgate M(Rz(params), Rz(params))
        mc.operations.CompRzRz(params, wires=curr_wires)
    for i, odd_wire in enumerate(wires[1:-1:2]):
        idx = list(wires).index(odd_wire)
        mc.operations.fSWAP(wires=[wires[idx], wires[idx + 1]])
    projector: BasisStateProjector = qml.Projector(initial_state, wires=wires)
    return qml.expval(projector)


# Create a QNode
nif_qnode = qml.QNode(circuit, nif_device)
qml.draw_mpl(nif_qnode)(
    params=np.array([0.1, 0.2]),
    wires=nif_device.wires,
    initial_state=initial_state
)

# Evaluate the QNode
expval = nif_qnode(
    params=np.array([0.1, 0.2]),
    wires=nif_device.wires,
    initial_state=initial_state
)
print(f"Expectation value: {expval:.4f}")
```

Output: ```Expectation value: 0.9901```

![Quantum circuit generated by the
`circuit` function. \label{fig:minimal_example_quantum_circuit_simulation}](./images/minimal_example_quantum_circuit_simulation.svg){
width="80%" height="20%" fig-env="figure" fig-align="center"
}

## Minimal Example - QML

The next example builds quantum kernel classifiers with the `FermionicPQCKernel` and `LinearNIFKernel` classes and
evaluates them on the iris dataset [@iris_53] using 20-fold cross-validation. `MatchCake.ml` builds on
Scikit-learn [@scikit-learn; @sklearn_api], giving a familiar interface for machine learning practitioners.

```python
import matplotlib as mpl
from matchcake.ml import CrossValidation, CrossValidationVisualizer
from matchcake.ml.kernels import FermionicPQCKernel
from matchcake.ml.kernels.linear_nif_kernel import LinearNIFKernel
from matchcake.ml.visualisation import ClassificationVisualizer
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

# Load the iris dataset
dataset = datasets.load_iris(as_frame=True)
X, y = dataset.data, dataset.target

# Define estimators using different kernels for comparison
n_qubits = 8
estimators = {
    "Fermionic PQC": Pipeline(
        [
            ("scaler", MinMaxScaler(feature_range=(0, 1))),
            # Example of kernel without alignment, which is the default behaviour
            ("kernel", FermionicPQCKernel(n_qubits=n_qubits, rotations="X,Z")),
            ("classifier", SVC(kernel="precomputed")),
        ]
    ),
    "Fermionic Linear": Pipeline(
        [
            ("scaler", MinMaxScaler(feature_range=(0, 1))),
            # Example of kernel with alignment, which improves the performance
            ("kernel", LinearNIFKernel(n_qubits=n_qubits, alignment=True)),
            ("classifier", SVC(kernel="precomputed")),
        ]
    ),
}

# Run cross-validation
cvo = CrossValidation(estimators, X, y).run()

# Create subplots for visualization
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
# Visualize the classification boundaries for a selected model and fold
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)
viz = ClassificationVisualizer(x=X, n_pts=25_000)
viz.plot_2d_decision_boundaries(
    model=estimators["Fermionic Linear"].fit(x_train, y_train),
    title="",
    legend_labels=dataset.target_names,
    y=y,
    fig=fig,
    ax=axes[0],
    legend_loc="lower right",
)
# Visualize cross-validation results with violin plots
cv_viz = CrossValidationVisualizer(cvo)
cv_viz.plot(
    ax=axes[1],
    score_name="Accuracy",
    estimator_name_key="Kernel",
    score_name_map={"train_score": "Train", "test_score": "Test"},
    palette="pastel",
)
```

![ Output of the machine learning example, using 8-qubit kernels and 20-fold cross-validation. Left: decision
boundaries for the Fermionic Linear classifier on the iris dataset. Right: violin plots of training and testing
accuracies across the cross-validation folds for the Fermionic PQC and Fermionic Linear classifiers.
\label{fig:minimal_example_quantum_machine_learning}](./images/minimal_example_quantum_machine_learning.svg){
width="100%" height="20%" fig-env="figure" fig-align="center"
}

# Research impact statement

`MatchCake` was central to “Fermionic Machine Learning” [@10821385], enabling the benchmarking of kernel-based
free-fermionic classifiers at qubit counts infeasible with SVS.

That work underscores the need for classically simulable baselines when benchmarking QML,
to test whether claimed quantum resources are actually necessary before asserting a quantum advantage. It also
highlights a key strength of free-fermionic models: the provable absence of barren
plateaus [@diaz2023showcasing], critical for efficient training.
`MatchCake` will therefore be an invaluable tool for the quantum simulation and QML communities, supporting fundamental
research and the empirical validation of quantum advantage claims.

# AI usage disclosure

The author used AI-assisted tools to improve the grammar, style, and clarity of the manuscript, and to help draft
limited parts of the source-code documentation. No scientific content, results, or conclusions were generated by AI
tools. The author has reviewed, edited, and validated all AI-assisted content and assumes full responsibility for the
final version of the manuscript and the software.

# Acknowledgements

The author thanks Victor Drouin-Touchette for comments on the main text and Stefanos Kourtis for his support on the
topic. This research was funded by the research Chair in Quantum Computing funded by Ministère de l'Économie, de
l'Innovation et de l'Énergie, the QSciTech CREATE program funded by the Natural Sciences and Engineering Research
Council of Canada (NSERC), and NSERC Discovery and Alliance grants.

# References
