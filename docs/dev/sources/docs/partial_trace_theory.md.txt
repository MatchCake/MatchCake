# Tracing out wires and mixed Gaussian states

This document explains how MatchCake traces out wires of a matchgate circuit, why the
reduced state stays classically tractable although it is mixed, and how a circuit is run
from such a mixed state. It builds on the covariance-matrix formalism of
[Computing expectation values using the Covariance-Matrix](expectation_values_theory.md):
the state of the device is a computational-basis state $`|x\rangle`$ evolved by the
matchgate circuit $`V`$, and it is fully described by its Majorana covariance matrix
$`\Lambda = R^\top \Lambda_0 R`$, with $`R`$ the single-particle transition matrix of $`V`$.
Section 7 treats the initial product states with superposed qubits, which also carry a
displacement vector.

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

## 2. Qubits versus fermionic modes

### 2.1 The reduced covariance matrix

Let $`\mu_0 < \mu_1 < \cdots < \mu_{2|A|-1}`$ be the Majorana indices $`I_A`$ of the kept wires,
and $`\Lambda`$ the covariance matrix of the full register. The covariance matrix of the reduced
state of the qubits of $`A`$ is $`(\Lambda_A)_{jk} = i\, \mathrm{Tr}(\rho_A\, c'_j c'_k)`$, with
$`c'_j`$ the Majorana operators of the reduced system. Written in the full system, the Pauli
word $`c'_j c'_k`$ is the product $`c_{\mu_j} c_{\mu_k}`$ times a $`Z_b = -i\, c_{2b} c_{2b+1}`$ on
every traced-out wire $`b`$ located between the wires of $`\mu_j`$ and $`\mu_k`$. Wick's theorem
then gives every entry as a single Pfaffian of a submatrix of $`\Lambda`$:

```math
(\Lambda_A)_{jk} = (-1)^{r_{jk}}\, \mathrm{Pf}\big(\Lambda|_{S_{jk}}\big),
\qquad
S_{jk} = \{\mu_j, \mu_k\} \cup \bigcup_{b \in B_{jk}} \{2b, 2b+1\},
\qquad j < k,
```

where $`B_{jk} \subseteq B`$ is the set of traced-out wires located between the wires of
$`\mu_j`$ and $`\mu_k`$, and $`r_{jk} = |B_{jk}|`$. The other entries follow from
$`\Lambda_A^\top = -\Lambda_A`$. The submatrix $`\Lambda|_{S_{jk}}`$ gathers the entry
$`\Lambda_{\mu_j \mu_k}`$, the block of the traced-out wires $`\Lambda_B|_{B_{jk}}`$, and the
correlations between the two kept Majorana operators and those wires.

**Contiguous kept wires.** No traced-out wire lies between two kept wires, so $`r_{jk} = 0`$
and $`S_{jk} = \{\mu_j, \mu_k\}`$ for every pair. The Pfaffian of a $`2 \times 2`$ antisymmetric
matrix is its upper off-diagonal entry, and the formula reduces to

```math
(\Lambda_A)_{jk} = \Lambda_{\mu_j \mu_k}
\qquad \Longleftrightarrow \qquad
\Lambda_A = \Pi_A\, \Lambda\, \Pi_A^\top ,
```

with $`\Pi_A`$ the $`2|A| \times 2n`$ matrix that selects the rows of $`I_A`$. The block
$`\Lambda_B`$ of the traced-out wires and the correlations between $`A`$ and $`B`$ drop out.
This is all that MatchCake computes when it traces out wires: `mc.covariance_matrix(wires)`,
`reduced_covariance_matrix` and `partial_trace` return this principal submatrix, and
`qml.vn_entropy`, `qml.purity` and `qml.density_matrix` are evaluated from it (Section 3).

**One traced-out wire in between.** When a single traced-out wire $`b`$ separates the wires of
$`\mu_j`$ and $`\mu_k`$, the Pfaffian is the one of a $`4 \times 4`$ matrix:

```math
(\Lambda_A)_{jk} = -\Big( \Lambda_{\mu_j \mu_k}\, \Lambda_{2b,\, 2b+1}
- \Lambda_{\mu_j,\, 2b+1}\, \Lambda_{2b,\, \mu_k}
+ \Lambda_{\mu_j,\, 2b}\, \Lambda_{2b+1,\, \mu_k} \Big) .
```

Since $`\Lambda_{2b,\, 2b+1} = -\langle Z_b \rangle`$, the first term is the submatrix entry
weighted by the polarization of the traced-out wire, and the two others are the correlations
of the kept Majorana operators with that wire. The entry differs from $`\Lambda_{\mu_j \mu_k}`$
in general.

### 2.2 Non-contiguous kept wires

For kept wires that are not a contiguous block, two different matrices can be called the
reduced covariance matrix:

- the matrix $`\Lambda_A`$ of Section 2.1, with its Pfaffians. It holds the two-point functions
  of the reduced state of the qubits. That state is not Gaussian in general, so $`\Lambda_A`$
  does not determine it: its higher-order correlations are not Pfaffians of $`\Lambda_A`$, and
  it cannot be given to `GaussianStatePrep`.
- the principal submatrix $`\Pi_A \Lambda \Pi_A^\top`$, that is the formula of Section 2.1 with
  the Jordan-Wigner strings on the traced-out wires dropped ($`r_{jk} = 0`$). It is the
  covariance matrix of the reduced state of the fermionic modes, which is Gaussian but is not
  the reduced state of the qubits.

When a traced-out wire $`b`$ lies between two kept wires, the product of two kept Majorana
operators on each side of $`b`$ keeps a factor $`Z_b`$. For the kept wires $`\{0, 2\}`$ of a
three-wire register, for example,

```math
c_1 c_4 = (Y_0)(Z_0 Z_1 X_2) = i\, X_0 Z_1 X_2 ,
```

so the entry $`(1, 2)`$ of the principal submatrix, $`\Lambda_{1,4} = i \langle c_1 c_4 \rangle`$, holds
$`-\langle X_0 Z_1 X_2 \rangle`$ and not $`-\langle X_0 X_2 \rangle`$. A Pauli word on the reduced system then stands in the original
system for the same word dressed with a $`Z`$ on every traced-out wire located between two of
its $`X`$ or $`Y`$ factors. This is what the reduced state of the fermionic modes describes. It is a valid
Gaussian state, but it differs from the reduced state of the qubits, and so do its entropy
and its purity. The reduced state of the qubits on such wires is not Gaussian in general and
has no covariance-matrix description.

The measurements that return a matrix or a number (`mc.covariance_matrix`, the entropy, the
purity and the density matrix) therefore accept only contiguous kept wires, and raise an error
otherwise. With `gate_barrier=False`, `mc.covariance_matrix` returns the principal submatrix,
the reduced state of the fermionic modes, with a warning. The Pfaffian formula of Section 2.1
for non-contiguous wires is not implemented. `mc.reduced_state`, `reduced_state` and
`partial_trace` accept any kept wires: the state they return remembers the removed wires with
gate barriers, described in Section 8.
Expectation values and probabilities on arbitrary wires are not affected by this restriction:
they are computed from the full covariance matrix.

### 2.3 Worked example on three wires

This example follows every step from a state on three wires to the covariance matrix of two
of them. All the numbers can be checked by hand.

#### Step 1: the state and its Majorana operators

Take the three-wire state

```math
|\psi\rangle = \tfrac{1}{2}\big( |000\rangle + |110\rangle + |101\rangle + |011\rangle \big) ,
```

where the bits are those of wires $`0`$, $`1`$ and $`2`$, in this order. Every basis state in
it has an even number of ones, so the state has a definite fermion parity. It can be prepared
by a matchgate circuit from $`|000\rangle`$, so it is a fermionic Gaussian state.

Three wires carry six Majorana operators, two per wire. Wire $`k`$ owns $`c_{2k}`$ and
$`c_{2k+1}`$, and each one carries a string of $`Z`$ on the wires before it:

| Wire | Majorana indices | Operators |
| --- | --- | --- |
| $`0`$ | $`0, 1`$ | $`c_0 = X_0`$, $`c_1 = Y_0`$ |
| $`1`$ | $`2, 3`$ | $`c_2 = Z_0 X_1`$, $`c_3 = Z_0 Y_1`$ |
| $`2`$ | $`4, 5`$ | $`c_4 = Z_0 Z_1 X_2`$, $`c_5 = Z_0 Z_1 Y_2`$ |

#### Step 2: the covariance matrix of the full state

The covariance matrix is the $`6 \times 6`$ matrix
$`\Lambda_{\mu\nu} = i \langle\psi| c_\mu c_\nu |\psi\rangle`$ for $`\mu \neq \nu`$, with zeros on
the diagonal. It is antisymmetric, so only the entries with $`\mu < \nu`$ need to be computed.
Each product $`c_\mu c_\nu`$ is a Pauli word, obtained with $`Y Z = i X`$, $`X Z = -i Y`$ and
$`Z^2 = \mathbb{1}`$. Three of them have a non-zero expectation value in $`|\psi\rangle`$:

```math
\begin{aligned}
c_1 c_2 &= Y_0 Z_0 X_1 = i\, X_0 X_1
&&\Rightarrow\quad \Lambda_{1,2} = i \cdot i\, \langle X_0 X_1 \rangle = -\langle X_0 X_1 \rangle = -1 ,
\\ c_3 c_4 &= Z_0 Y_1 Z_0 Z_1 X_2 = i\, X_1 X_2
&&\Rightarrow\quad \Lambda_{3,4} = -\langle X_1 X_2 \rangle = -1 ,
\\ c_0 c_5 &= X_0 Z_0 Z_1 Y_2 = -i\, Y_0 Z_1 Y_2
&&\Rightarrow\quad \Lambda_{0,5} = i \cdot (-i)\, \langle Y_0 Z_1 Y_2 \rangle = \langle Y_0 Z_1 Y_2 \rangle = -1 .
\end{aligned}
```

The expectation values are read from the state. $`X_0 X_1`$ flips the bits of wires $`0`$ and
$`1`$: it exchanges $`|000\rangle`$ with $`|110\rangle`$ and $`|101\rangle`$ with $`|011\rangle`$,
which leaves $`|\psi\rangle`$ unchanged, so $`\langle X_0 X_1 \rangle = 1`$. In the same way
$`\langle X_1 X_2 \rangle = 1`$. The operator $`Y_0 Z_1 Y_2`$ flips the bits of wires $`0`$ and
$`2`$ and gives a factor $`-1`$ to each of the four basis states, so
$`Y_0 Z_1 Y_2 |\psi\rangle = -|\psi\rangle`$ and $`\langle Y_0 Z_1 Y_2 \rangle = -1`$. The twelve
other products with $`\mu < \nu`$ have a vanishing expectation value. For example
$`c_2 c_3 = i Z_1`$ and $`\langle Z_1 \rangle = 0`$, because wire $`1`$ is $`0`$ in half of the
basis states and $`1`$ in the other half.

Placing the three values at their positions, and their opposites at the transposed positions,
gives

```math
\Lambda = \begin{pmatrix}
0 & 0 & 0 & 0 & 0 & -1
\\ 0 & 0 & -1 & 0 & 0 & 0
\\ 0 & 1 & 0 & 0 & 0 & 0
\\ 0 & 0 & 0 & 0 & -1 & 0
\\ 0 & 0 & 0 & 1 & 0 & 0
\\ 1 & 0 & 0 & 0 & 0 & 0
\end{pmatrix} .
```

Rows and columns are numbered from $`0`$ to $`5`$. Every row holds a single $`\pm 1`$, and one
can check that $`\Lambda^2 = -\mathbb{1}`$: the state is pure.

#### Step 3: keeping wires 0 and 1, tracing out wire 2

**The selection matrix.** The kept wires $`A = \{0, 1\}`$ own the Majorana indices
$`I_A = \{0, 1, 2, 3\}`$, which we list in increasing order as
$`(\mu_0, \mu_1, \mu_2, \mu_3) = (0, 1, 2, 3)`$. The selection matrix $`\Pi_A`$ has one row per
kept Majorana index and one column per Majorana index of the full register, so it is
$`4 \times 6`$. Row $`j`$ is zero everywhere except for a $`1`$ in column $`\mu_j`$:

```math
\Pi_A = \begin{pmatrix}
1 & 0 & 0 & 0 & 0 & 0
\\ 0 & 1 & 0 & 0 & 0 & 0
\\ 0 & 0 & 1 & 0 & 0 & 0
\\ 0 & 0 & 0 & 1 & 0 & 0
\end{pmatrix} .
```

The wires form a contiguous block, so Section 2.1 gives
$`\Lambda_A = \Pi_A\, \Lambda\, \Pi_A^\top`$.

**The product.** Multiplying by $`\Pi_A`$ on the left keeps the rows $`0, 1, 2, 3`$ of
$`\Lambda`$, and multiplying by $`\Pi_A^\top`$ on the right keeps the columns $`0, 1, 2, 3`$:

```math
\Pi_A\, \Lambda = \begin{pmatrix}
0 & 0 & 0 & 0 & 0 & -1
\\ 0 & 0 & -1 & 0 & 0 & 0
\\ 0 & 1 & 0 & 0 & 0 & 0
\\ 0 & 0 & 0 & 0 & -1 & 0
\end{pmatrix},
\qquad
\Lambda_A = \Pi_A\, \Lambda\, \Pi_A^\top = \begin{pmatrix}
0 & 0 & 0 & 0
\\ 0 & 0 & -1 & 0
\\ 0 & 1 & 0 & 0
\\ 0 & 0 & 0 & 0
\end{pmatrix} .
```

In other words, $`(\Lambda_A)_{jk} = \Lambda_{\mu_j \mu_k}`$: the reduced covariance matrix is
$`\Lambda`$ with the rows and the columns of the traced-out wire, $`4`$ and $`5`$, deleted.

**What was lost.** The entry $`\Lambda_{1,2}`$ links two kept Majorana operators and survives.
The entries $`\Lambda_{0,5}`$ and $`\Lambda_{3,4}`$ link a kept Majorana operator to one of the
traced-out wire, in column $`5`$ and column $`4`$, and are deleted. Rows $`0`$ and $`3`$ of
$`\Lambda_A`$ are now empty, so $`\Lambda_A^2 \neq -\mathbb{1}`$: the reduced state is mixed.

**Entropy and purity.** Section 3 computes them from the polarizations $`|\lambda_j|`$, the
non-negative eigenvalues of $`i \Lambda_A`$. Here the only non-zero block of $`\Lambda_A`$ sits
on the indices $`(1, 2)`$:

```math
i \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}
\ \text{has eigenvalues}\ \pm 1,
\qquad
\text{and the empty rows } 0 \text{ and } 3 \text{ give the eigenvalue } 0 \text{ twice.}
```

The eigenvalues of $`i \Lambda_A`$ are therefore $`(-1, 0, 0, 1)`$, and the two polarizations
are $`|\lambda_0| = 1`$ and $`|\lambda_1| = 0`$: one pure mode and one maximally mixed mode.
With $`p_j = (1 + |\lambda_j|)/2`$, that is $`p_0 = 1`$ and $`p_1 = 1/2`$,

```math
S(\rho_A) = \underbrace{0}_{\text{mode } 0} + \underbrace{\ln 2}_{\text{mode } 1} = \ln 2,
\qquad
\mathrm{Tr}\, \rho_A^2 = \frac{1 + 1^2}{2} \cdot \frac{1 + 0^2}{2} = \frac{1}{2} .
```

**In MatchCake.** The same steps with the toolbox, starting from the state vector:

```python
import numpy as np
from matchcake.utils import CovarianceMatrixToolbox

state = np.zeros(8)
state[[0b000, 0b110, 0b101, 0b011]] = 0.5
covariance = CovarianceMatrixToolbox.from_density_matrix(np.outer(state, state))  # Step 2
reduced = CovarianceMatrixToolbox.reduce(covariance, [0, 1])                      # Step 3
print(CovarianceMatrixToolbox.mode_polarizations(reduced))   # [0. 1.]
print(CovarianceMatrixToolbox.von_neumann_entropy(reduced))  # 0.6931... = ln 2
print(CovarianceMatrixToolbox.purity(reduced))               # 0.5
```

In a circuit that prepares this state on `nif.qubit`, `mc.covariance_matrix(wires=[0, 1])`
returns the same `reduced` matrix.

#### Step 4: keeping wires 0 and 2, tracing out wire 1

The kept wires are not contiguous any more: wire $`1`$ sits between them. This is the case of
Section 2.2, with two different matrices.

**The selection matrix.** The kept wires own the Majorana indices
$`(\mu_0, \mu_1, \mu_2, \mu_3) = (0, 1, 4, 5)`$, so the $`1`$ of rows $`2`$ and $`3`$ moves to the
columns $`4`$ and $`5`$:

```math
\Pi_A = \begin{pmatrix}
1 & 0 & 0 & 0 & 0 & 0
\\ 0 & 1 & 0 & 0 & 0 & 0
\\ 0 & 0 & 0 & 0 & 1 & 0
\\ 0 & 0 & 0 & 0 & 0 & 1
\end{pmatrix} .
```

**The fermionic matrix.** Keeping the rows and the columns $`0, 1, 4, 5`$ of $`\Lambda`$ gives

```math
\Pi_A\, \Lambda\, \Pi_A^\top = \begin{pmatrix}
\Lambda_{0,0} & \Lambda_{0,1} & \Lambda_{0,4} & \Lambda_{0,5}
\\ \Lambda_{1,0} & \Lambda_{1,1} & \Lambda_{1,4} & \Lambda_{1,5}
\\ \Lambda_{4,0} & \Lambda_{4,1} & \Lambda_{4,4} & \Lambda_{4,5}
\\ \Lambda_{5,0} & \Lambda_{5,1} & \Lambda_{5,4} & \Lambda_{5,5}
\end{pmatrix}
= \begin{pmatrix}
0 & 0 & 0 & -1
\\ 0 & 0 & 0 & 0
\\ 0 & 0 & 0 & 0
\\ 1 & 0 & 0 & 0
\end{pmatrix} .
```

This is what MatchCake returns with `gate_barrier=False`.

**The qubit matrix.** The reduced system has two wires, $`0'`$ and $`1'`$, which are the
original wires $`0`$ and $`2`$, and its own Majorana operators
$`c'_0 = X_{0'}`$, $`c'_1 = Y_{0'}`$, $`c'_2 = Z_{0'} X_{1'}`$, $`c'_3 = Z_{0'} Y_{1'}`$. Its
covariance matrix is $`(\Lambda_A)_{jk} = i \langle c'_j c'_k \rangle`$, given by the Pfaffian
formula of Section 2.1. The six entries with $`j < k`$ fall in two groups.

- Two entries stay within one kept wire, $`(j, k) = (0, 1)`$ and $`(2, 3)`$. No traced-out wire
  lies in between, so they are copied: $`(\Lambda_A)_{0,1} = \Lambda_{0,1} = 0`$ and
  $`(\Lambda_A)_{2,3} = \Lambda_{4,5} = 0`$.
- Four entries link the two kept wires, with $`j \in \{0, 1\}`$ and $`k \in \{2, 3\}`$. The
  traced-out wire $`1`$, with Majorana indices $`\{2, 3\}`$, lies in between, so
  $`r_{jk} = 1`$ and $`S_{jk} = \{\mu_j, 2, 3, \mu_k\}`$.

The Pfaffian of a $`4 \times 4`$ antisymmetric matrix $`M`$ is
$`\mathrm{Pf}(M) = M_{0,1} M_{2,3} - M_{0,2} M_{1,3} + M_{0,3} M_{1,2}`$. Applied to the submatrix of
$`\Lambda`$ on the indices $`S_{jk}`$ in increasing order, and with the sign $`(-1)^{r_{jk}} = -1`$,
the four entries are

```math
\begin{aligned}
(\Lambda_A)_{0,2} &= -\mathrm{Pf}\big(\Lambda|_{\{0, 2, 3, 4\}}\big)
= -\big( \Lambda_{0,2} \Lambda_{3,4} - \Lambda_{0,3} \Lambda_{2,4} + \Lambda_{0,4} \Lambda_{2,3} \big)
= -\big( 0 \cdot (-1) - 0 \cdot 0 + 0 \cdot 0 \big) = 0 ,
\\ (\Lambda_A)_{0,3} &= -\mathrm{Pf}\big(\Lambda|_{\{0, 2, 3, 5\}}\big)
= -\big( \Lambda_{0,2} \Lambda_{3,5} - \Lambda_{0,3} \Lambda_{2,5} + \Lambda_{0,5} \Lambda_{2,3} \big)
= -\big( 0 \cdot 0 - 0 \cdot 0 + (-1) \cdot 0 \big) = 0 ,
\\ (\Lambda_A)_{1,2} &= -\mathrm{Pf}\big(\Lambda|_{\{1, 2, 3, 4\}}\big)
= -\big( \Lambda_{1,2} \Lambda_{3,4} - \Lambda_{1,3} \Lambda_{2,4} + \Lambda_{1,4} \Lambda_{2,3} \big)
= -\big( (-1) \cdot (-1) - 0 \cdot 0 + 0 \cdot 0 \big) = -1 ,
\\ (\Lambda_A)_{1,3} &= -\mathrm{Pf}\big(\Lambda|_{\{1, 2, 3, 5\}}\big)
= -\big( \Lambda_{1,2} \Lambda_{3,5} - \Lambda_{1,3} \Lambda_{2,5} + \Lambda_{1,5} \Lambda_{2,3} \big)
= -\big( (-1) \cdot 0 - 0 \cdot 0 + 0 \cdot 0 \big) = 0 .
\end{aligned}
```

Filling the lower triangle by antisymmetry,

```math
\Lambda_A = \begin{pmatrix}
0 & 0 & 0 & 0
\\ 0 & 0 & -1 & 0
\\ 0 & 1 & 0 & 0
\\ 0 & 0 & 0 & 0
\end{pmatrix} .
```

**Comparing the two.** In the reduced system $`c'_1 c'_2 = i\, X_{0'} X_{1'}`$, exactly as
$`c_1 c_2`$ in Step 2, so the entry $`(1, 2)`$ of a covariance matrix on these two wires
means $`-\langle X_0 X_2 \rangle`$, and the entry $`(0, 3)`$ means $`\langle Y_0 Y_2 \rangle`$.

- The qubit matrix has $`(\Lambda_A)_{1,2} = -1`$, that is $`\langle X_0 X_2 \rangle = 1`$. This
  is correct: $`X_0 X_2`$ exchanges $`|000\rangle`$ with $`|101\rangle`$ and $`|110\rangle`$ with
  $`|011\rangle`$, which leaves $`|\psi\rangle`$ unchanged. The value comes from the product
  $`\Lambda_{1,2} \Lambda_{3,4}`$ of the two correlations that go through the traced-out wire,
  which the selection matrix alone never sees.
- The fermionic matrix has $`0`$ there, and $`-1`$ at the entry $`(0, 3)`$. That entry is
  $`\Lambda_{0,5} = \langle Y_0 Z_1 Y_2 \rangle`$, with the $`Z_1`$ of the Jordan-Wigner string
  still inside, whereas $`\langle Y_0 Y_2 \rangle = 0`$ for the qubits.

MatchCake raises an error for `wires=[0, 2]`, because the principal submatrix is not the
reduced state of the qubits, and returns the fermionic matrix when `gate_barrier=False` is passed.

## 3. Normal modes, entropy and purity

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

A mode with $`|\lambda_j| = 1`$ is pure and a mode with $`\lambda_j = 0`$ is maximally mixed.
At most $`\min(|A|, |B|)`$ modes are mixed when the full state is pure. The magnitudes
$`|\lambda_j|`$ are the non-negative eigenvalues of the Hermitian matrix $`i \Lambda_A`$.

The spectrum of $`\rho_A`$ is the one of the product of the $`\rho_j`$, which gives the
Von Neumann entropy and the purity in terms of the occupation probabilities
$`p_j = (1 + \lambda_j)/2`$:

```math
S(\rho_A) = -\sum_j \big[ p_j \ln p_j + (1 - p_j) \ln (1 - p_j) \big],
\qquad
\mathrm{Tr}\, \rho_A^2 = \prod_j \frac{1 + \lambda_j^2}{2}
= 2^{-|A|} \sqrt{\det\big(\mathbb{1} - \Lambda_A^2\big)} .
```

For a pure full state, $`S(\rho_A)`$ is the entanglement entropy between $`A`$ and $`B`$.

The density matrix itself follows from the Wick expansion over the even subsets $`S`$ of
Majorana indices, with $`c'_S`$ the ordered product of the Majorana operators of $`S`$ in the
reduced system:

```math
\rho_A = \frac{1}{2^{|A|}} \sum_{|S|\ \mathrm{even}} i^{|S|/2}\, \mathrm{Pf}\big(\Lambda_A|_S\big)\, c'_S .
```

The sum has $`2^{2|A|-1}`$ terms, so the dense density matrix is only available for a small
number of kept wires.

## 4. Starting a circuit from a mixed Gaussian state

Nothing in the covariance-matrix formalism requires the initial state to be pure. A matchgate
circuit $`U`$ with transition matrix $`R_2`$ maps any Gaussian state with covariance matrix
$`\Lambda_A`$ to the Gaussian state with covariance matrix

```math
\Lambda_A \mapsto R_2^\top \Lambda_A R_2 .
```

Expectation values of Pauli words are Pfaffians of submatrices of the evolved matrix, by
Wick's theorem. The probability of a computational-basis outcome $`y`$ on a set of wires
$`W`$ is the overlap of the state with a pure Gaussian projector,

```math
p(y) = 2^{-|W|}\, \Big| \mathrm{Pf}\big( \Lambda|_W + \Lambda_y \big) \Big| ,
```

with $`\Lambda_y`$ the covariance matrix of $`|y\rangle`$. Only the projector needs to be pure
for this formula. Sampling measures the qubits one at a time, updating the covariance matrix
after each outcome, which also holds for a mixed state.

The device therefore runs a circuit from a mixed Gaussian state with the same code paths as
from a product state: the initial covariance matrix is the only difference, and the
displacement vector of the extended covariance matrix is zero unless the state preparation
carries one (Section 7).

## 5. In MatchCake

Inside a `QNode` on `nif.qubit`, the subsystem is read with measurements, and the wires given
to a measurement are the wires that are kept:

| Measurement | Result |
| --- | --- |
| `mc.covariance_matrix(wires)` | $`\Lambda_A`$ |
| `mc.reduced_state(wires)` | `GaussianStatePrep` of the reduced state |
| `qml.vn_entropy(wires)` | $`S(\rho_A)`$ |
| `qml.purity(wires)` | $`\mathrm{Tr}\, \rho_A^2`$ |
| `qml.density_matrix(wires)` | $`\rho_A`$, for a small number of wires |

A mixed Gaussian state enters a circuit through the state preparation
`mc.GaussianStatePrep(covariance_matrix, wires)`, queued as the first operation. Feeding it the
result of `mc.covariance_matrix` continues the computation on the reduced system, typically on
a device with fewer wires. `mc.reduced_state` returns that state preparation directly, to be
queued with `qml.apply`. It also holds the displacement vector of Section 7 and the gate
barriers of Section 8, which a matrix cannot carry. Outside of a `QNode`, the device methods
`reduced_covariance_matrix`, `reduced_state` and `partial_trace` give the same results.

The functions behind these features take a covariance matrix and are the static methods of
`matchcake.utils.CovarianceMatrixToolbox`: `von_neumann_entropy`, `purity`, `mode_polarizations`,
`reduce`, `to_density_matrix` and `block_diagonalize`, which returns $`R_A`$ and the
polarizations.

## 6. Gradients

The reduced covariance matrix is a linear function of $`\Lambda`$, and every output of a
circuit started from a `GaussianStatePrep` is a polynomial function of its covariance matrix.
Gradients therefore flow from the outputs of a second circuit back to the parameters of the
first one without any special treatment.

The entropy and the purity depend on $`\Lambda_A`$ through its spectrum only. The purity is
computed from a determinant and is smooth everywhere. The entropy has an infinite slope at
$`|\lambda_j| = 1`$, where a mode becomes pure. MatchCake drops the vanishing terms there, so
the gradient stays finite, but it is not informative at such a point.

The decomposition $`\Lambda_A = R_A^\top \Lambda_D R_A`$ is not unique: rotations inside a
mode, and rotations between two modes with equal or opposite polarizations, leave
$`\Lambda_A`$ unchanged. Writing $`dR_A = \Omega R_A`$ with $`\Omega`$ antisymmetric gives

```math
R_A\, d\Lambda_A\, R_A^\top = [\Lambda_D, \Omega] + d\Lambda_D ,
```

which is inverted block by block, dividing by $`\lambda_j - \lambda_k`$ or
$`\lambda_j + \lambda_k`$. The components along the gauge directions are projected out, so the
gradient of `CovarianceMatrixToolbox.block_diagonalize` is exact for any quantity that depends on the state
only.

## 7. Initial states with a displacement vector

A product state with superposed qubits is not an even Gaussian state. Next to its covariance
matrix it has a displacement vector $`d_\mu = \langle c_\mu \rangle`$, and it obeys Wick's
theorem with the extended covariance matrix of
[Section 5 of the expectation-value document](expectation_values_theory.md):

```math
\widetilde\Lambda = \begin{pmatrix} \Lambda & d \\ -d^\top & 0 \end{pmatrix},
\qquad
\langle c_{\mu_1} \cdots c_{\mu_m} \rangle \propto
\begin{cases}
\mathrm{Pf}\big(\Lambda|_S\big) & m \text{ even}, \\ \mathrm{Pf}\big(\widetilde\Lambda|_{S \cup \{2n\}}\big) & m \text{ odd},
\end{cases}
```

with $`S = \{\mu_1, \ldots, \mu_m\}`$ and $`2n`$ the index of the parity mode. A matchgate
circuit maps $`d`$ to $`R^\top d`$.

### 7.1 The Jordan-Wigner string of the removed wires

Let $`A`$ be a contiguous block of kept wires, with $`p`$ removed wires before it. The Majorana
operators of the reduced system have Jordan-Wigner strings that start at the first kept wire, so

```math
c'_j = Z_\text{pre}\, c_{\mu_j},
\qquad
Z_\text{pre} = \prod_{b < \min A} Z_b ,
\qquad
c'_{j_1} \cdots c'_{j_m} = (Z_\text{pre})^m\, c_{\mu_{j_1}} \cdots c_{\mu_{j_m}} ,
```

since $`Z_\text{pre}`$ commutes with the Majorana operators of the kept wires and squares to the
identity.

**Even products.** The strings cancel. The even expectation values of the reduced state are
those of the full state, so its covariance matrix is the principal submatrix
$`\Lambda_A = \Pi_A \Lambda \Pi_A^\top`$ for every contiguous block, with or without a
displacement vector.

**Odd products, kept wires first.** For $`p = 0`$ there is no string, and the odd expectation
values of the reduced state are Pfaffians of $`\widetilde\Lambda`$ on the indices of $`A`$ and the
parity mode. The reduced state obeys Wick's theorem with

```math
\widetilde\Lambda_A = \begin{pmatrix} \Lambda_A & d_A \\ -d_A^\top & 0 \end{pmatrix},
```

where $`d_A`$ is the part of $`d`$ on the Majorana indices of $`A`$. Tracing out the last wires is
deleting their rows and columns from $`\widetilde\Lambda`$.

**Odd products, other kept wires.** For $`p > 0`$ one string survives. With
$`Z_b = -i\, c_{2b} c_{2b+1}`$, the displacement vector of the reduced state is a Pfaffian that
includes the removed wires,

```math
d'_j = \langle Z_\text{pre}\, c_{\mu_j} \rangle \propto
\mathrm{Pf}\big( \widetilde\Lambda|_{I_\text{pre} \cup \{\mu_j,\, 2n\}} \big) ,
```

with $`I_\text{pre}`$ the Majorana indices of the $`p`$ removed wires. It is not the part of $`d`$
on $`A`$, and the higher odd expectation values are not given by Wick's theorem with
$`\Lambda_A`$ and $`d'`$: the reduced state has no description by one covariance matrix and one
displacement vector. For a product state without gates, for example, the entry of $`d`$ for
$`X`$ on wire $`k`$ is $`x_k \prod_{l<k} z_l`$, while the reduced state has
$`\langle X_k \rangle = x_k`$.

### 7.2 The even part of the reduced state

The even expectation values define a state on their own,

```math
\rho_A^\text{even} = \tfrac{1}{2} \big( \rho_A + P_A\, \rho_A\, P_A \big) ,
```

with $`P_A`$ the parity of the kept wires. It is the Gaussian state with covariance matrix
$`\Lambda_A`$ and no displacement vector. A matchgate circuit maps even products to even
products, so it never mixes the even and the odd parts of a state. Starting a circuit from
$`\rho_A^\text{even}`$ therefore gives the exact value of every quantity that is even in the
Majorana operators: the probabilities and the samples in the computational basis, and the
expectation values of the Pauli words with an even number of $`X`$ and $`Y`$. The odd
expectation values, such as $`\langle X_k \rangle`$, vanish in it.

### 7.3 In MatchCake

| Feature | State with a displacement vector |
| --- | --- |
| `mc.covariance_matrix(wires)`, `reduced_covariance_matrix` | $`\Lambda_A`$, on the same wires as without displacement |
| `mc.reduced_state`, `reduced_state`, `partial_trace`, kept wires from the first wire | `GaussianStatePrep` with $`\Lambda_A`$ and $`d_A`$ |
| `mc.reduced_state`, `reduced_state`, `partial_trace`, other kept wires | `GaussianStatePrep` with $`\Lambda_A`$ alone, the state $`\rho_A^\text{even}`$, and a barrier before its first wire (Section 8) |
| `qml.vn_entropy`, `qml.purity`, `qml.density_matrix` | `NotImplementedError` |

A reduced state with a displacement vector enters another circuit through
`mc.GaussianStatePrep(covariance_matrix, wires, displacement_vector=...)`. The entropy, the
purity and the density matrix of Section 3 are those of a state without displacement, and they
are not evaluated from $`\Lambda_A`$ alone when the state has one.

## 8. Gate barriers

Sections 2.2 and 7.1 have the same cause: a Majorana operator of a kept wire still holds a
$`Z`$ on every removed wire before it. MatchCake records where with barriers, so that the
reduced state on any wires can be used without returning a wrong result.

### 8.1 The rule

Let $`W`$ be a Pauli word on the kept wires and $`b`$ a removed wire. Each $`X`$ or $`Y`$ of
$`W`$ is an odd number of Majorana operators of its wire, and each $`Z`$ an even number. The
principal submatrix of Section 2.1 keeps the Jordan-Wigner strings on the removed wires, so the
expectation value it gives for $`W`$ is

```math
\langle W \rangle_{\Pi_A \Lambda \Pi_A^\top} =
\Big\langle W \prod_{b} Z_b^{\,m_b(W)} \Big\rangle ,
```

where $`m_b(W)`$ is the number of $`X`$ and $`Y`$ of $`W`$ on the kept wires after $`b`$. The
strings cancel, and the value is the one of the reduced state of the qubits, exactly when
$`m_b(W)`$ is even for every removed wire $`b`$. The removed wires between the same two kept
wires give the same condition, and the removed wires after the last kept wire give none. A
**barrier** before a kept wire stands for the removed wires located just before it:

- a barrier before a kept wire other than the first one, for removed wires between two kept
  wires (Section 2.2);
- a barrier before the first kept wire, for removed wires in front of the kept ones. Its
  condition is an even number of $`X`$ and $`Y`$ in the whole word. It is only needed for a
  state with a displacement vector (Section 7), since the other words vanish otherwise.

**A Pauli word is exact when it has an even number of $`X`$ and $`Y`$ on the right of every
barrier.** The words made of $`Z`$ only always are, hence the probabilities and the samples in
the computational basis.

### 8.2 Gates

A matchgate on two neighbouring kept wires on the same side of a barrier is quadratic in the
Majorana operators of that side. It maps a word to words with the same parities $`m_b`$, so
the exact words stay exact after any number of such gates. A gate with one wire on each side of
a barrier does not keep these parities. In the reduced register it is the gate on the qubits
with a $`Z`$ on the removed wires, which is another gate.

### 8.3 In MatchCake

`mc.reduced_state`, `reduced_state` and `partial_trace` return a `GaussianStatePrep` with the
positions of its barriers in `barriers`. A device started from it raises a `DeviceError` for

- a gate that acts on both sides of a barrier;
- an observable with a Pauli word that breaks the rule;
- samples in another basis than the computational one on a wire on the right of a barrier;
- an entropy, a purity or a density matrix on wires with a barrier between two of them, or on
  wires that are not all on the left of every barrier when a displacement vector was dropped.

Everything else is exact. The barriers follow the state: a further reduction inherits them, and
a wire added in $`|0\rangle`$ in the place of removed wires is on the left of their barrier. A
state with a barrier before its first wire is returned without displacement vector, since the
words that would use it are refused. Such a state is marked (`odd_part_dropped`): its barrier is
kept for as long as a wire remains on its right, also after wires are added and removed again on
its left, because the words with an odd number of $`X`$ and $`Y`$ on its right do not vanish in
the reduced state of the qubits.

With `gate_barrier=False`, the state is returned without barrier and with a warning. It is
then the reduced state of the fermionic modes of Section 2.2, and nothing checks the gates and
the observables any more. `mc.covariance_matrix` and `reduced_covariance_matrix` return a
matrix, which cannot carry a barrier: they refuse the kept wires with a removed wire or a
barrier between them, unless `gate_barrier=False` is passed.

## 9. Limitations

- **Initial states with a displacement vector.** See Section 7. The entropy, the purity and the
  density matrix are not implemented for them, and their reduced state on wires that do not
  start at the first wire only gives the quantities that are even in the Majorana operators.
- **Gate barriers.** See Section 8. A reduced state on wires that are not a contiguous block
  cannot be evolved by gates across the removed wires, and only gives the observables that
  follow the rule of its barriers.
- **Wire order of a Gaussian state preparation.** A `GaussianStatePrep` can act on part of the
  wires of the device, the other wires starting in $`|0\rangle`$, but it must list its wires
  in the order of the device: permuting fermionic modes is not a relabeling of qubits.
- **Analytic only.** The reduced-state measurements are not estimated from shots.

## References

- S. Bravyi, "Lagrangian representation for fermionic linear optics," Quantum Inf. Comput. 5,
  216 (2005), [arXiv:quant-ph/0404180](https://arxiv.org/abs/quant-ph/0404180).
- I. Peschel, "Calculation of reduced density matrices from correlation functions,"
  J. Phys. A 36, L205 (2003), [arXiv:cond-mat/0212631](https://arxiv.org/abs/cond-mat/0212631).
- F. Iglói and I. Peschel, "On reduced density matrices for disjoint subsystems,"
  EPL 89, 40001 (2010), [arXiv:0910.5671](https://arxiv.org/abs/0910.5671).
- D. J. Brod, "Efficient classical simulation of matchgate circuits with generalized inputs
  and measurements," Phys. Rev. A 93, 062332 (2016),
  [arXiv:1602.03539](https://arxiv.org/abs/1602.03539).
