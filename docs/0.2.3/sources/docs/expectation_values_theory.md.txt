# Computing expectation values using the Covariance-Matrix

This document explains how MatchCake turns a matchgates circuit into a Pauli expectation
value through the Majorana covariance matrix. First, we explain how it works with
computational-basis initial states and then show how it works with arbitrary product states.

For references, the Gaussian formalism is due to Bravyi [3]; the
extension to arbitrary product states follows Jozsa, Miyake & Strelchuk [1] and
Projansky, Necaise & Whitfield [2].

## Setting

We use the convention $\langle A\rangle_\rho = \mathrm{Tr}[\rho A]$ throughout. The
Jordan–Wigner (JW) transformation maps $n$ qubits to $2n$ Hermitian Majorana operators
(0-indexed),

$$
c_{2k} = Z_0\cdots Z_{k-1}\,X_k, \qquad c_{2k+1} = Z_0\cdots Z_{k-1}\,Y_k,
$$

obeying the Clifford algebra $\{c_\mu, c_\nu\} = 2\delta_{\mu\nu}$. The computation of the expectation values
goes by four stages: initial covariance, single-particle transition matrix, evolved covariance, and
Pfaffian readout.

## 1. The initial covariance matrix $\Lambda_0$ for a computational-basis state

The initial state is a computational-basis state $|x\rangle = |x_1 \cdots x_n\rangle$.
Set $z_k = (-1)^{x_k}$ and recall $c_{2k} = Z_{<k} X_k$, $c_{2k+1} = Z_{<k} Y_k$. The
element is $(\Lambda_0)_{\mu\nu} = i \langle x | c_\mu c_\nu | x \rangle$, and the
expectation $\langle x | P | x \rangle$ is nonzero only when the Pauli string $P$ is
diagonal, i.e. composed of $Z$ and $I$.

In the diagonal ($\mu = \nu$), the strings $Z_{<k}$ square to identity and
$X_k^2 = Y_k^2 = I$, so $c_\mu c_\mu = I$ and $i \langle I \rangle = i$. The definition
removes this term, hence $(\Lambda_0)_{\mu\mu} = 0$.

In the upper/lower diagonal ($\{\mu, \nu\} = \{2k, 2k+1\}$), the strings $Z_{<k}$
cancel and $c_{2k} c_{2k+1} = X_k Y_k = i Z_k$, which is diagonal. Then
$(\Lambda_0)_{2k, 2k+1} = i \langle i Z_k \rangle = -z_k$ and
$(\Lambda_0)_{2k+1, 2k} = +z_k$.

Otherwise, the product retains an unpaired $X$ or $Y$, which is off-diagonal, so the
expectation vanishes and $(\Lambda_0)_{\mu\nu} = 0$.

Only the indices $2k$ and $2k+1$ are coupled, so $\Lambda_0$ is block-diagonal:

$$
\Lambda_0 = \bigoplus_{k=0}^{n-1} (-z_k) \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}.
$$

The overall sign follows the convention $c_{2k} c_{2k+1} = i Z_k$.

## 2. The single-particle transition matrix from $V$

A matchgate circuit $V$ acts by conjugation as a rotation of the Majorana operators into
one another. We fix the convention

$$
V\,c_\mu\,V^\dagger = \sum_\nu R_{\mu\nu}\,c_\nu,
\qquad\text{equivalently}\qquad
V^\dagger c_\mu V = \sum_\nu R_{\nu\mu}\,c_\nu,
\qquad R \in O(2n),
$$

where $R$ is the **single-particle transition matrix (SPTM)**. Each elementary matchgate
contributes a local orthogonal rotation on a small Majorana subspace, and the global $R_G$ is
their ordered product, assembled in $\mathcal{O}(\mathrm{poly}\,n)$ time. This single
matrix captures the entire effect of the circuit, which is a free-fermionic evolution.
MatchCake tracks $R_G$ explicitly, so it is always available in
$\mathcal{O}(n^2)$.

## 3. The evolved covariance matrix $\Lambda$

The evolved state is $|\psi\rangle = V|\psi_0\rangle$ which can be describe by the evolved covariance matrix defined as

$$
\Lambda_{\mu\nu} = i\,\langle \psi_0|V^\dagger c_\mu c_\nu V|\psi_0\rangle, \quad \mu\neq\nu,
$$

or defined by commutators
$\Lambda_{\mu\nu} = \tfrac{i}{2}\langle \psi_0|V^\dagger[c_\mu, c_\nu]V|\psi_0\rangle$.

This covariance matrix can be computed via the SPTM

$$
\Lambda_{\mu\nu} = i\,\langle \psi_0|V^\dagger c_\mu V V^\dagger c_\nu V|\psi_0\rangle \\
= i\,\left\langle \psi_0\left|\left(\sum_j R_{j\mu}\,c_j\right)\left(\sum_\ell R_{\ell\nu}\,c_\ell\right)\right|\psi_0\right\rangle \\
= i\sum_{j,\ell} R_{j\mu} R_{\ell\nu} \langle \psi_0|c_j c_\ell|\psi_0\rangle \\
= i\sum_{j,\ell} R_{j\mu} \langle \psi_0|c_j c_\ell|\psi_0\rangle R_{\ell\nu} \\
= \sum_{j,\ell} R_{j\mu} (\Lambda_0)_{j\ell} R_{\ell\nu}
$$

so that

$$
\boxed{\;\Lambda = R^\top \Lambda_0\, R\;}
$$

Note that the placement of the transposes is fixed by the SPTM convention of
Section 2; the mirrored index convention $V^\dagger c_\mu V = \sum_\nu R_{\mu\nu}c_\nu$
produces $\Lambda = R\,\Lambda_0\,R^\top$ instead, the two are equivalent provided $R$.

## 4. Expectation values from $\Lambda$, Pfaffian, and Jordan–Wigner

A Pauli expectation $\langle P\rangle$ is obtained in three steps.

**Majorana decomposition.** Every Pauli word is a single ordered Majorana product,
$P = \alpha\,c_{\mu_1}\cdots c_{\mu_m}$ with $\mu_1 < \cdots < \mu_m$ and a known
unit-modulus phase $\alpha$, obtained by applying the $\{X_k, Y_k, Z_k\}\to c$ dictionary,
sorting the indices, and accumulating $(-1)^{\#\text{transpositions}}$.

**Wick's theorem in Pfaffian form.** For the Gaussian state, with
$S = (\mu_1,\ldots,\mu_m)$ and $\Lambda|_S$ the $m\times m$ principal submatrix,

$$
\langle c_{\mu_1}\cdots c_{\mu_m}\rangle =
\begin{cases}
i^{-m/2}\,\mathrm{Pf}(\Lambda|_S) & m \text{ even}, \\
0 & m \text{ odd}.
\end{cases}
$$

Odd-$m$ products vanish because computational-basis states preserve fermion parity.

**Assembly.** Therefore $\langle P\rangle = \alpha\,i^{-m/2}\,\mathrm{Pf}(\Lambda|_S)$ for
even $m$, and a Hamiltonian is evaluated term by term,
$\langle\mathcal{H}\rangle = \sum_j \beta_j\langle P_j\rangle$.

## 5. Arbitrary product state

We now repeat Stages 1–4 for a generic initialization
$|\psi_\text{prod}\rangle = \bigotimes_k (a_k|0\rangle + b_k|1\rangle)$, with per-qubit
Bloch components

$$
x_k = \langle X_k\rangle = 2\,\mathrm{Re}(a_k^* b_k), \quad
y_k = \langle Y_k\rangle = 2\,\mathrm{Im}(a_k^* b_k), \quad
z_k = \langle Z_k\rangle = |a_k|^2 - |b_k|^2.
$$

### 5.1 Initial covariance, and why $\Lambda_0$ alone is insufficient

The two-point matrix is defined exactly as before, but transverse components now propagate
across qubits through the JW $Z$-string, so $\Lambda_0$ is no longer block-diagonal:

$$
(\Lambda_0)_{2k,2k+1} = -z_k, \qquad
(\Lambda_0)_{2j,2k} = y_j\,p_{jk}\,x_k \;\;(j<k), \quad
p_{jk} = \prod_{j<l<k} z_l,
$$

with analogous expressions for the mixed index pairs. Crucially, a generic product state is
*not* fermionic Gaussian: it carries nonzero single-Majorana expectations
$\langle c_\mu\rangle \neq 0$, and a real antisymmetric matrix has no slot in which to store
them. The diagnosis is immediate on $|+\rangle = \tfrac{1}{\sqrt 2}(|0\rangle + |1\rangle)$:
here $\langle c_0\rangle = \langle X\rangle = 1$ yet $\Lambda_0 = 0$, so the Pfaffian formula
returns $\langle X\rangle = 0$, which is wrong. The missing data is collected in the
**displacement vector**

$$
d_\mu = \langle\psi_\text{prod}|c_\mu|\psi_\text{prod}\rangle, \qquad
d_{2k} = x_k\!\prod_{l<k} z_l, \qquad
d_{2k+1} = y_k\!\prod_{l<k} z_l.
$$

### 5.2 Restoring Gaussianity: the extended covariance matrix

The fix is to extend the algebra by one mode. With the total-parity operator
$c_\text{par} = \prod_\mu c_\mu$, define the primed Majoranas $c'_\mu = i\,c_\mu c_\text{par}$
and $c'_{2n} = c_\text{par}$; these $2n+1$ operators are Hermitian and satisfy
$\{c'_\mu, c'_\nu\} = 2\delta_{\mu\nu}$. In this enlarged algebra every product state is
Gaussian, because the parity-breaking linear data becomes a genuine two-point correlator
with the parity mode. The **extended covariance matrix** is the
$(2n+1)\times(2n+1)$ real antisymmetric matrix

$$
\boxed{\;
\widetilde\Lambda = \begin{pmatrix} \Lambda & d \\ -d^\top & 0 \end{pmatrix},
\qquad d_\mu = \langle c_\mu\rangle,
\;}
$$

where the index $2n$ labels $c'_\text{par}$. The upper-left block is the ordinary covariance
matrix and the border is the displacement.

### 5.3 SPTM lift

A matchgate has no linear (single-Majorana) generators, so it leaves the parity mode
invariant, $V^\dagger c_\text{par} V = c_\text{par}$. The SPTM therefore lifts
block-diagonally,

$$
\widetilde R = \begin{pmatrix} R & 0 \\ 0 & 1 \end{pmatrix} \in O(2n+1),
\qquad
\widetilde\Lambda = \widetilde R^\top \widetilde\Lambda_0\, \widetilde R.
$$

Expanding the product reproduces both evolution laws at once,

$$
\widetilde R^\top \widetilde\Lambda_0 \widetilde R =
\begin{pmatrix} R^\top \Lambda_0 R & R^\top d_0 \\ -(R^\top d_0)^\top & 0 \end{pmatrix} =
\begin{pmatrix} \Lambda & d \\ -d^\top & 0 \end{pmatrix},
$$

that is, $\Lambda = R^\top\Lambda_0 R$ exactly as in Stage 3, together with $d = R^\top d_0$:
the displacement rotates under the same SPTM. No additional circuit data is required, only
one fixed extra row and column.

### 5.4 Expectation values

The readout is identical to Stage 4, performed on $\widetilde\Lambda$, with one rule that
keeps every submatrix even-sized so the Pfaffian remains well-defined. For
$P = \alpha\,c_{\mu_1}\cdots c_{\mu_m}$:

- if $m$ is even, $\widetilde S = (\mu_1,\ldots,\mu_m)$ and $\widetilde\alpha = \alpha$;
- if $m$ is odd, $\widetilde S = (\mu_1,\ldots,\mu_m, 2n)$ and
  $\widetilde\alpha = \alpha\,i^m(-1)^{m(m-1)/2}$.

Then

$$
\langle P\rangle = \widetilde\alpha\,i^{-|\widetilde S|/2}\,\mathrm{Pf}(\widetilde\Lambda|_{\widetilde S}).
$$

The odd-$m$ terms that previously vanished now acquire the parity index $2n$, pad to even
length, and take a real value. This is precisely what allows us to work with $|+\rangle$: the expectation
$\langle X\rangle = \langle c_0\rangle$ is read from
$\mathrm{Pf}(\widetilde\Lambda|_{\{0,2n\}}) = \widetilde\Lambda_{0,2n} = d_0$ instead of
collapsing to zero.

## Computing Hamiltonian Energy

Let us now bring everything together to compute the energy of a system evolving under a free-fermionic evolution. The
energy,

$$
\varepsilon = \langle \mathcal{H} \rangle = \sum_j \beta_j \langle P_j \rangle,
$$

can then be evaluated as

$$
\boxed{\;
\varepsilon = \sum_j \beta_j \widetilde{\alpha}_j \, i^{-|\widetilde{S}_j|/2}
\, \mathrm{Pf}\!\left(
\widetilde{R}^{\top} \widetilde{\Lambda}_0 \widetilde{R}
\big|_{\widetilde{S}_j}
\right)
\;}.
$$

To summarize, the expectation value of a Hamiltonian can be computed in polynomial time by tracking the SPTMs associated
with each matchgate in the circuit, together with the initial covariance matrix. Once these quantities are available, we
compute the Jordan-Wigner transformation of the Pauli operators appearing in the Hamiltonian and evaluate a batch of
Pfaffians to obtain the final energy.

## References

[1] Jozsa, Richard, Akimasa Miyake, and Sergii Strelchuk. "Jordan-Wigner formalism for
arbitrary 2-input 2-output matchgates and their classical simulation." arXiv preprint
arXiv:1311.3046 (2013).

[2] Projansky, Andrew M., Jason Necaise, and James D. Whitfield. "Gaussianity and
simulability of Cliffords and matchgates." Journal of Physics A: Mathematical and
Theoretical 58, no. 19 (2025): 195302.

[3] Bravyi, Sergey. "Lagrangian representation for fermionic linear optics." arXiv preprint
quant-ph/0404180 (2004).
