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

We introduce `MatchCake`, a new Python package that implements a simulator of matchgate quantum circuits 
(also known as matchcircuits). Matchgates form a restricted class of unitaries that preserves parity and act 
on nearest-neighbor qubits. These circuits are equivalent to a hamiltonian evolution of 
non-interacting Majorana fermions, which makes them 
classically simulable in polynomial time while retaining rich quantum structure. `MatchCake` provides a practical 
framework for exploring this class of circuits within PennyLane [@bergholm2022pennylane] and 
PyTorch [@Ansel_PyTorch_2_Faster_2024], enabling research on classically simulable 
quantum machine learning models with automatic differentiation. We demonstrate its capabilities through applications 
in quantum kernel methods, including classification [@10821385], regression, and beyond.



# Statement of need

In quantum computing, quantum circuits provide a formal framework for representing the discrete time evolution of a 
quantum state under a sequence of operations known as quantum gates. They constitute the backbone of quantum algorithms, 
quantum system simulations, and broader quantum information processing tasks. They are therefore central to the research 
and development of methods that exploit quantum resources such as superposition, entanglement, and interference. 
However, since quantum hardware is still under active development, it is crucial to have efficient tools for simulating 
and analyzing quantum circuits on classical computers. Indeed, the classical simulation of quantum circuits is 
essential for the study of quantum systems, the design of quantum algorithms, and the development of quantum computers 
themselves. Moreover, the study of classically simulable circuit families is itself a key avenue for understanding the 
boundary between classical and quantum computational power. As a result, a large number of quantum circuit simulators 
have been developed to address this need. In general, the state space of an $N$-qubit system has dimension $\bigO(2^N)$, 
so an explicit representation of a generic quantum state requires exponentially many parameters, which leads most 
simulation approaches to scale exponentially in the number of qubits. Nonetheless, efficient simulation becomes 
possible for restricted families of circuits whose states or evolutions admit compact, polynomial-size representations.

Quantum circuit simulators are differentiated by the classes of circuits they can simulate efficiently. For example, Stim [@gidney2021stim] is designed for stabilizer circuits and achieves polynomial complexity, $\bigO(N^2)$ where $N$ is the system size, when simulating circuits composed of Clifford gates. On the other hand, quimb [@gray2018quimb] leverages tensor network methods [@biamonte_tensor_2017], such as matrix product states (MPS), to efficiently simulate circuits with limited entanglement at a cost of $\bigO(N\chi^3)$, where $\chi$ is the bond dimension, which is a proxy for the amount of entanglement in the system.

At a higher level of abstraction, PennyLane [@bergholm2022pennylane] provides a widely used Python framework for quantum circuit simulation and quantum machine learning. PennyLane enables the definition, execution, and differentiation of quantum circuits across a variety of backends, making it particularly well suited for hybrid quantum–classical algorithms and learning-based applications. While PennyLane supports multiple simulators and hardware interfaces, efficient native support for matchgate-based circuits is currently lacking.

Specifically, PennyLane primarily relies on universal state-vector simulators (SVS), which, although capable of simulating arbitrary quantum circuits, suffer from an exponential scaling in the number of qubits, thereby are fundamentally not scalable. PennyLane also provides wrappers around specialized simulators such as Stim and quimb, enabling more efficient simulation for certain restricted classes of circuits. However, none of these backends target the subclass of quantum circuits composed of matchgates (matchcircuits [@valiant_quantum_2001]). Consequently, users interested in simulating matchgate-based circuits are forced to rely on exponentially scaling simulators, despite the fact that such circuits theoretically admit polynomial-time $\bigO(N^3)$ classical simulation [@brod_efficient_2016; @bravyi_contraction_2008; @PhysRevA.102.052604].

Unlike tensor-network-based methods, whose efficiency relies on restricting entanglement growth, the efficient simulability of matchcircuits arises from an underlying free-fermionic (Gaussian) structure, allowing them to support arbitrarily large entanglement while remaining tractable. In contrast to Clifford circuits, which are efficiently simulable due to a discrete and non-parametrized gate set, matchcircuits are composed of matchgates [@jozsa_matchgates_2008], a continuously parametrized class of two-qubit unitary gates corresponding to free-fermionic operations. Despite their theoretical importance and favorable resource trade-offs, there is currently no clear, dedicated tool for their simulation within general-purpose quantum software frameworks. 

On a technical side, a two-qubit matchgate written in the computational basis

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

subject to the constraint $\det(A) = \det(W)$. Indeed, when restricted to such matchcircuits on a linear register of qubits, the system's dynamics become equivalent to those of non-interacting Majorana fermions [@nielsen2005fermionic; @dai_extracting_2015].

Recent theoretical work has further clarified their position within the broader landscape of efficiently simulable quantum circuits. In particular, [@projansky2025gaussianity] established a formal connection between matchcircuits and Clifford circuits through fermionic Gaussian operations, highlighting matchcircuits as a natural generalization that interpolates between stabilizer dynamics and more general fermionic evolutions.

Beyond their theoretical interest, matchcircuits and their extensions have practical relevance for quantum simulation. Notably, [@mocherla2023extending] showed that augmenting matchcircuits with additional $ZZ$ interactions enables the efficient simulation of an interaction-limited Fermi–Hubbard system. This extension significantly broadens the class of physically relevant problems accessible to matchgate-based simulation techniques, encompassing applications in quantum thermodynamics [@PhysRevA.98.012309], quantum machine learning [@laar2022quantum; @10821385], and many-body quantum system simulation [@mocherla2023extending].

`MatchCake` is then designed to fill the gap in efficient simulation tools for matchcircuits. By providing a dedicated simulator integrated with PennyLane and PyTorch [@Ansel_PyTorch_2_Faster_2024], `MatchCake` enables researchers to explore the capabilities of matchcircuits in quantum machine learning and other applications. Its automatic differentiation support facilitates the development of learning-based models that leverage the unique properties of matchgate dynamics, opening new avenues for research in classically simulable quantum algorithms.


# Research Impact Statement

`MatchCake` has already demonstrated its usefulness in the context of the paper “Fermionic Machine Learning” [@10821385], whose goal was to benchmark different kernel-based machine learning models for data classification. By using the present package, the authors were able to demonstrate the performance of free-fermionic models on a large number of qubits, an analysis that would not have been feasible with a state-vector simulator (SVS).

In that work, the authors emphasize the necessity of relying on such classically simulable models when benchmarking quantum machine learning (QML) approaches, in order to assess whether the claimed quantum resources are truly required to achieve strong performance. This type of verification is essential in the current landscape, prior to making any claims of quantum advantage in QML. From another perspective, the authors also discuss a key advantage of free-fermionic models over other QML approaches: the absence of barren plateaus, a property of critical importance for the efficient optimization of machine learning models.

It therefore follows that MatchCake is an essential tool for the quantum simulation and quantum machine learning communities. It is not only valuable for fundamental research in quantum computing and for the development of QML models, but also plays a central role in validating the claimed advantages of such models.



# Functionality

Using `MatchCake`, the user is expected to provide a matchcircuit $C$, which is a list of matchgates, the initial state $\Ket{\psi_\text{initial}}$ of the quantum system, required to be a product state, and the observable $\mathcal{F}$ to be measured. The observable $\mathcal{F}$ must be a linear combination

\begin{align}
    \mathcal{F} = \sum\limits_j \beta_j F_j
\end{align}

of observables $F_j$ that can be efficiently decomposed into Gaussian Majorana observables which take the form

\begin{align}
    F_j \in &\left\{\Proj{y}: y \in \{0,1\}^N \right\}
    \\ \nonumber
    &\cup
    \left\{ 
        \bigcup_{i=0}^{N-1} 
        \left\{ Z_{i}I_{i+1}, X_{i}X_{i+1}, Y_{i}Y_{i+1}, Y_{i}X_{i+1}, X_{i}Y_{i+1}, I_{i}Z_{i+1} \right\}
    \right\}
    \\ \nonumber
    &\cup
    \left\{ 
        \bigcup_{i<k=0}^{N-1} 
        \left\{ Z_{i}Z_{k} \right\} 
    \right\}
\end{align}

where $\Ket{y}$ is any computational-basis state, $X_i$, $Y_i$, and $Z_i$ denote the Pauli matrices 
acting on qubits $i$ of a linear register of $N$ qubits, $i \in \{0,\dots,N-2\}$ labels nearest-neighbour sites, 
and $k \in \{1,\dots,N-1\}$ denotes an arbitrary qubit index. In return, `MatchCake` computes the expectation value

\begin{align}
    \text{Output} = \braOket{\psi_\text{initial}}{C^\dagger \mathcal{F} C}{\psi_\text{initial}}
\end{align}

in time polynomial in $N$, the number of qubits in the system.


More specifically, the `MatchCake` algorithm presented in \autoref{fig:matchgate-algorithm} is divided into three distinct branches, depending on the observable $\mathcal{F}$ to be measured. Branch (A) handles projectors in the computational basis. Branch (B) treats the $Z_{i}I_{i+1}, I_{i}Z_{i+1}, Z_{i}Z_{k}$ Pauli words by expressing them as a linear combination of branch-(A) evaluations. Branch (C) addresses the remaining Pauli words.

Across the different stages of the algorithm, several intermediate procedures are employed:

1. the Majorana decomposition of the observable,
2. the use of a lookup table to obtain a special tensor $\mathcal{A}$ which represents the Wick's contraction [@wick_evaluation_1950] of a Majorana string [@brod_efficient_2016],
3. the mapping of the matchgate circuit $C$ to its corresponding transition matrix $R_C$ [@brod_efficient_2016],
4. the Jordan–Wigner transformation [@jordan_uber_1928; @nielsen2005fermionic], and
5. the use of a Clifford subroutine based on PennyLane’s stabilizer simulator [@bergholm2022pennylane] to obtain the final expectation value.


![
The `MatchCake` algorithm. Algorithm inputs are shown as red rounded rectangles, intermediate results as yellow rounded rectangles, and the final output as a green rounded rectangle; processing steps are represented by grey rectangles. The algorithm branches into three distinct paths depending on the observable $\mathcal{F}$ being measured: (A) a computational-basis projector, (B) a Pauli word diagonal in the computational basis, and (C) other specific Pauli words that can be decomposed into Gaussian Majorana observables. The coefficients $\beta$ are real scalars arising from the Majorana decomposition. \label{fig:matchgate-algorithm}
](images/matchcake-algorithm-detailed.svg){
width="100%" height="20%" fig-env="figure" fig-align="center"
}



## Minimal Example - Quantum Circuit Simulation

In the following code block, we demonstrate a simple example on how to use `MatchCake` to create a matchcircuit and obtain the expectation value of a computational-basis projector after applying the circuit to an initial state.

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

![Quantum circuit generated by the `circuit` function. \label{fig:minimal_example_quantum_circuit_simulation}](./images/minimal_example_quantum_circuit_simulation.svg){
width="80%" height="20%" fig-env="figure" fig-align="center"
}

## Minimal Example - Quantum Machine Learning

In the following code block, we demonstrate a simple example on how to use `MatchCake` to create quantum kernel-based classification models using the `FermionicPQCKernel` and `LinearNIFKernel` classes. We then evaluate its performance on the iris dataset [@iris_53] using cross-validation with 20 random folds. The framework used in the `MatchCake.ml` is built on top of Scikit-learn [@scikit-learn; @sklearn_api], providing a familiar and consistent interface for machine learning practitioners.

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

![ Output of the machine learning example code. On the left:
Example of a decision boundaries for the Fermionic Linear classifier on the iris dataset. 
On the right: Violin plots showing the distribution of training and testing accuracies across different cross-validation folds for both the Fermionic PQC and Fermionic Linear classifiers.
\label{fig:minimal_example_quantum_machine_learning}](./images/minimal_example_quantum_machine_learning.svg){
width="100%" height="20%" fig-env="figure" fig-align="center"
}


# Acknowledgements

The author thanks Victor Drouin-Touchette for comments on the main text and Stefanos Kourtis for his support on the topic. This research was funded by the research Chair in Quantum Computing funded by Ministère de l'Économie, de l'Innovation et de l'Énergie, the QSciTech CREATE program funded by the Natural Sciences and Engineering Research Council of Canada (NSERC), and NSERC Discovery and Alliance grants.


# AI usage disclosure

The author acknowledges the use of AI-assisted tools during the preparation of this manuscript to improve grammar, 
style, and clarity of presentation. AI tools were also used to assist in drafting limited portions of the source code 
documentation for the purpose of improving readability. No scientific content, results, or conclusions were generated 
by AI tools. The author has reviewed, edited, and validated all AI-assisted content and assumes full responsibility 
for the final version of the manuscript and the software.

# References