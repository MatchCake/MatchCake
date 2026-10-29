# Tracing out wires

This document explains how MatchCake traces out wires of a matchgate circuit, why the
reduced state stays classically tractable although it is mixed, and how expectation values,
probabilities and the entanglement entropy are read from it. It builds on the covariance-matrix formalism of
[Computing expectation values using the Covariance-Matrix](expectation_values_theory.md):
the state of the device is a computational-basis state $`|x\rangle`$ evolved by the
matchgate circuit $`V`$, and it is fully described by its Majorana covariance matrix
$`\Lambda = R^\top \Lambda_0 R`$, with $`R`$ the single-particle transition matrix of $`V`$.

## 1. The reduced state is Gaussian

Split the $`n`$ wires into a kept set $`A`$ and a traced-out set $`B`$. The reduced
state $`\rho_A = \mathrm{Tr}_B |\psi\rangle\langle\psi|`$ is characterized by the
expectation values of the operators supported on $`A`$. In the fermionic picture these
are the products of the Majorana operators $`c_\mu`$ with $`\mu \in I_A = \{2a, 2a+1 : a \in A\}`$,
and their expectation values in $`\rho_A`$ are the same as in $`|\psi\rangle`$. Wick's theorem
therefore still applies, with the principal submatrix of the covariance matrix,

```math
\langle c_{\mu_1} \cdots c_{\mu_{2k}} \rangle_{\rho_A}
= i^{-k}\, \mathrm{Pf}\big(\Lambda_A|_{\{\mu_1, \ldots, \mu_{2k}\}}\big),
\qquad
\Lambda_A = \Lambda|_{I_A \times I_A},
```

and the odd products vanish because $`|x\rangle`$ has a definite fermion parity. The
reduced state is thus a fermionic Gaussian state with covariance matrix $`\Lambda_A`$.
Unlike $`\Lambda`$, the submatrix $`\Lambda_A`$ does not square to $`-\mathbb{1}`$ in
general: the reduced state is mixed.

**Qubits versus modes.** The Majorana operators of the kept wires, relabeled as a
system of $`|A|`$ wires, are $`c'_{2j} = Z'_{<j} X'_j`$ and $`c'_{2j+1} = Z'_{<j} Y'_j`$,
whose Jordan-Wigner strings run through the kept wires only. A Pauli word on the reduced
system is a product of the $`c'_\mu`$, and the reduction above identifies $`c'_\mu`$ with the
original $`c_{\mu'}`$ of the corresponding kept wire. In the original system this Pauli word
is therefore dressed with a $`Z`$ on every traced-out wire lying between two of its $`X`$ or $`Y`$
factors, which is the fermionic (mode) partial trace. When $`A`$ is a contiguous block of wires
no traced-out wire lies between two kept wires, and the reduction coincides with the usual
qubit partial trace. For a non-contiguous $`A`$ the two differ, and MatchCake implements the
fermionic one, which keeps the reduced state Gaussian and lets nearest-neighbour matchgates
of the reduced system act as free-fermion hoppings between the kept modes.

## 2. Normal modes, entropy and purity

Every real antisymmetric matrix can be brought to a block-diagonal form by a special
orthogonal matrix. Writing the decomposition in the convention of the single-particle
transition matrices,

```math
\Lambda_A = R_A^\top \Lambda_D R_A,
\qquad
\Lambda_D = \bigoplus_{j=1}^{|A|} (-\lambda_j) \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix},
\qquad
R_A \in SO(2|A|),
```

the numbers $`\lambda_j \in [-1, 1]`$ are the polarizations $`\langle Z_j \rangle`$ of the
**normal modes**. The block-diagonal matrix $`\Lambda_D`$ is the covariance matrix of a
product of single-mode mixtures, and $`R_A`$ is the transition matrix of a Gaussian unitary
$`V_A`$, so that

```math
\rho_A = V_A \Big( \bigotimes_j \rho_j \Big) V_A^\dagger,
\qquad
\rho_j = \frac{1 + \lambda_j}{2}\, |0\rangle\langle 0| + \frac{1 - \lambda_j}{2}\, |1\rangle\langle 1| .
```

The entanglement entropy of the cut and the purity follow from the polarizations alone,
with $`p_j = (1 + \lambda_j) / 2`$,

```math
S(\rho_A) = -\sum_j \big[ p_j \ln p_j + (1 - p_j) \ln (1 - p_j) \big],
\qquad
\mathrm{Tr}[\rho_A^2] = \prod_j \frac{1 + \lambda_j^2}{2} .
```

**Computing the decomposition.** The Hermitian matrix $`i\Lambda_A`$ has the eigenvalues
$`\pm\lambda_j`$. For an eigenvector $`v_j`$ of eigenvalue $`\lambda_j > 0`$, the real
vectors $`u_j = \sqrt{2}\,\mathrm{Re}(v_j)`$ and $`w_j = \sqrt{2}\,\mathrm{Im}(v_j)`$ satisfy
$`\Lambda_A u_j = \lambda_j w_j`$ and $`\Lambda_A w_j = -\lambda_j u_j`$, and they form the
rows $`2j`$ and $`2j+1`$ of $`R_A`$. Zero eigenvalues are degenerate between the two
sectors, so their vectors are replaced by an orthonormal basis of the kernel of $`\Lambda_A`$.
A sign flip of one row enforces $`\det R_A = 1`$, and the polarizations are read from the
product $`R_A \Lambda_A R_A^\top`$ so that the result is insensitive to the signs chosen by
the eigensolver.

## 3. Reading the reduced state

Wick's theorem holds for mixed Gaussian states, so the Pfaffian formulas of the expectation
values of pure states apply to $`\Lambda_A`$ unchanged. Likewise, the probability of an outcome
$`y`$ on the measured wires $`W`$ is the overlap of two Gaussian states,

```math
p(y) = \frac{1}{2^{|W|}} \Big| \mathrm{Pf}\big( \Lambda_A|_W + \Lambda_y \big) \Big| ,
```

with $`\Lambda_y`$ the block-diagonal covariance matrix of $`|y\rangle`$, and samples are drawn
wire block by wire block from these probabilities. Tracing out further wires of the reduced
state is again a principal submatrix, so reductions chain.

## 4. Gradients

The reduced covariance matrix $`\Lambda_A`$ is a differentiable function of the parameters
of the circuit, and so are the expectation values and probabilities read from it. The
normal-mode decomposition, from which the entropy and the purity are computed, is made
differentiable by a custom backward pass.
Writing a perturbation of the transition matrix as $`dR_A = \Omega R_A`$ with $`\Omega`$
antisymmetric, the perturbation of the covariance matrix reads

```math
R_A\, d\Lambda_A\, R_A^\top = [\Lambda_D, \Omega] + d\Lambda_D ,
```

which is inverted block by block: the diagonal $`2 \times 2`$ blocks give $`d\lambda_j`$, and
an off-diagonal block $`(j, k)`$ gives the four components of $`\Omega_{jk}`$ divided by
$`\lambda_j - \lambda_k`$ or $`\lambda_j + \lambda_k`$. The directions with a vanishing
denominator (rotations inside a mode, and rotations mixing two modes of equal or opposite
polarization) do not change $`\Lambda_A`$: they are gauge freedoms of the decomposition and
are projected out, which is exact for any loss that only depends on the reduced state. Pure
modes are always degenerate, $`\lambda_j = \pm 1`$, but physical perturbations cannot push a
polarization beyond $`1`$, so only the well-defined components appear and the gradient is
exact there. At an exact degeneracy between two *mixed* modes the decomposition itself is not
differentiable; the readouts of Section 3, which never build the decomposition, have a
well-defined gradient at every point.

## 5. Limitations

- The initial state of the first circuit must be a computational-basis state. A product
  state with superposed qubits carries a nonzero displacement vector, and its reduced state is
  Gaussian only in the extended Majorana algebra of the whole system, not in the algebra of the
  kept wires alone.
- The reduction is the fermionic partial trace. It equals the qubit partial trace for a
  contiguous block of kept wires, and differs from it by Jordan-Wigner strings through the
  traced-out wires otherwise.

## References

[1] Bravyi, Sergey. "Lagrangian representation for fermionic linear optics." arXiv preprint
quant-ph/0404180 (2004).

[2] Peschel, Ingo. "Calculation of reduced density matrices from correlation functions."
Journal of Physics A: Mathematical and General 36, no. 14 (2003): L205.

[3] Brod, Daniel J. "Efficient classical simulation of matchgate circuits with generalized
inputs and measurements." Physical Review A 93, no. 6 (2016): 062332.
