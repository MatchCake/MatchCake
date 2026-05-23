# Expectation Values in MatchCake: Theory and Algorithm

This document explains how MatchCake computes expectation values of Pauli observables for
matchgate circuits with arbitrary qubit product state initialization. The algorithm is
implemented in `MPfaffianExpvalStrategy` and is based on the extended Majorana algebra
introduced in Jozsa, Miyake & Strelchuk [1] and developed further by Projansky, Necaise
& Whitfield [2]. The underlying Gaussian formalism is due to Bravyi [3].

## Setting

An $n$-qubit circuit consists of:

1. A product state initialization $|\psi_\text{prod}\rangle = \bigotimes_{k=0}^{n-1}(a_k|0\rangle + b_k|1\rangle)$,
2. A matchgate circuit $U$ acting on $|\psi_\text{prod}\rangle$,
3. A Hamiltonian $\mathcal{H} = \sum_j \alpha_j P_j$ where each $P_j$ is a Pauli word.

The goal is to evaluate $\langle\mathcal{H}\rangle = \langle\psi_\text{prod}|U^\dagger \mathcal{H} U|\psi_\text{prod}\rangle$ efficiently.

## Jordan-Wigner Majoranas

The Jordan-Wigner (JW) transformation maps $n$ qubits to $2n$ Majorana operators
$c_0, \ldots, c_{2n-1}$ defined (0-indexed) as

$$
c_{2k} = Z_0 \cdots Z_{k-1}\, X_k, \qquad c_{2k+1} = Z_0 \cdots Z_{k-1}\, Y_k.
$$

They satisfy the Clifford algebra $\{c_\mu, c_\nu\} = 2\delta_{\mu\nu}$ and are Hermitian.
Every Pauli word $P$ on $n$ qubits can be written as $P = \alpha\, c_{\mu_1}\cdots c_{\mu_m}$
for some ordered index set $\mu_1 < \cdots < \mu_m$ and unit-modulus phase $\alpha$.

## Covariance matrix and Wick's theorem

For a state $\rho$, the **Majorana covariance matrix** is the $2n \times 2n$ real
antisymmetric matrix

$$
\Lambda_{\mu\nu} = i\,\mathrm{Tr}[\rho\, c_\mu c_\nu], \quad \mu \neq \nu, \qquad \Lambda_{\mu\mu} = 0.
$$

For fermionic Gaussian states (including all computational-basis product states), Wick's
theorem gives

$$
\mathrm{Tr}[\rho\, c_{\mu_1}\cdots c_{\mu_m}] = i^{-m/2}\,\mathrm{Pf}(\Lambda|_S), \quad m \text{ even},
$$

where $S = (\mu_1, \ldots, \mu_m)$ and $\Lambda|_S$ is the $m \times m$ principal submatrix.
Odd-$m$ products vanish for Gaussian states because they change total parity.

**Matchgate evolution.** A matchgate circuit $U$ maps the Majorana operators as
$U^\dagger c_\mu U = \sum_\nu Q_{\mu\nu} c_\nu$, where $Q \in O(2n)$ is the
single-particle transition matrix (SPTM). The covariance matrix evolves as

$$
\Lambda(t) = Q^\top \Lambda_0\, Q.
$$

MatchCake tracks $Q$ explicitly for the full circuit, so $\Lambda(t)$ is always
available in $O(n^2)$ time.

## The problem with non-Gaussian product states

A generic product state $|\psi_\text{prod}\rangle$ is **not** fermionic Gaussian: it has
nonzero single-Majorana expectations $\langle c_\mu\rangle \neq 0$ that $\Lambda$ has no
room to store. For example, on $|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$:

$$
\langle c_0\rangle = \langle X\rangle = 1, \quad \text{yet} \quad \Lambda = 0,
$$

so the Pfaffian formula gives $\langle X\rangle = 0$, which is wrong.

## Extended Majorana algebra

The fix, due to Jozsa, Miyake & Strelchuk [1], is to extend the algebra by one operator.

**Definition.** Let $c_\text{par} = \prod_{i=0}^{2n-1} c_i$ be the total-parity operator
(equal to $(\pm i)^n Z_0 \cdots Z_{n-1}$ under JW). Define the **primed** Majoranas

$$
c'_\text{par} = c_\text{par}, \qquad c'_\mu = i\, c_\mu\, c_\text{par} \;\text{ for } \mu = 0,\ldots,2n-1.
$$

These $2n+1$ operators form a valid Majorana algebra: they are Hermitian and satisfy
$\{c'_\mu, c'_\nu\} = 2\delta_{\mu\nu}$.

**Key identity.** In the primed algebra, every product state is Gaussian. This is because
single-qubit rotations $R_X(\theta) = e^{-i\theta X/2}$ have generators linear in JW
Majoranas, and such linear terms can always be absorbed into the quadratic part of the
primed algebra (Theorem 3 of [1], Lemma 1 of [2]).

## Extended covariance matrix

The **extended covariance matrix** $\widetilde\Lambda$ is the $(2n+1) \times (2n+1)$
real antisymmetric matrix built from the primed algebra:

$$
\widetilde\Lambda_{\mu\nu} = i\,\mathrm{Tr}[\rho\, c'_\mu c'_\nu], \quad \mu \neq \nu.
$$

Letting index $2n$ label $c'_\text{par}$, its entries simplify to

$$
\boxed{
\widetilde\Lambda =
\begin{pmatrix}
\Lambda & d \\
-d^\top & 0
\end{pmatrix}
}, \qquad d_\mu = \langle c_\mu\rangle_\rho,
$$

where $\Lambda$ is the standard $2n \times 2n$ covariance matrix and $d \in \mathbb{R}^{2n}$
is the **displacement vector** whose $\mu$-th entry is the single-Majorana expectation
$\langle c_\mu\rangle$. For a product state, these are the Bloch-vector components of
each qubit mapped through the JW string.

**Vacuum.** For $|0\rangle^{\otimes n}$, every $\langle c_\mu\rangle = 0$, so
$d = 0$ and $\widetilde\Lambda = \Lambda \oplus 0$.

**General product state.** For $|\psi_\text{prod}\rangle = \bigotimes_k(a_k|0\rangle + b_k|1\rangle)$,
let $x_k = 2\,\mathrm{Re}(a_k^* b_k)$, $y_k = 2\,\mathrm{Im}(a_k^* b_k)$,
$z_k = |a_k|^2 - |b_k|^2$. Then:

- $\Lambda_{2k,\,2k+1} = -z_k$ (same-qubit block),
- $\Lambda_{2j,\,2k} = y_j \cdot p_{jk} \cdot x_k$ for $j < k$, and similarly for mixed pairs (where $p_{jk} = \prod_{j<l<k} z_l$ is the JW parity string),
- $d_{2k} = x_k \cdot \prod_{l < k} z_l$ and $d_{2k+1} = y_k \cdot \prod_{l < k} z_l$.

All entries are analytic in the amplitudes and computable in $O(n^2)$.

## SPTM lift

A matchgate circuit has no linear-Majorana generators, so it acts trivially on
$c'_\text{par}$. The SPTM lifts block-diagonally:

$$
\widetilde Q = \begin{pmatrix} Q & 0 \\ 0 & 1 \end{pmatrix} \in O(2n+1),
$$

and the extended covariance matrix evolves
as $\widetilde\Lambda(t) = \widetilde Q^\top \widetilde\Lambda_0\, \widetilde Q$.
MatchCake computes the standard $Q$ from the circuit
and embeds it in this $(2n+1) \times (2n+1)$ block — no extra overhead.

## Pfaffian formula for Pauli expectations

**Parity rule.** A Pauli $P = \alpha\, c_{\mu_1}\cdots c_{\mu_m}$ is parity-preserving
when $m$ is even and parity-breaking when $m$ is odd.

**Expectation formula.** Given the extended index set:

- If $m$ is even: $\widetilde S = (\mu_1, \ldots, \mu_m)$, $\widetilde\alpha = \alpha$.
- If $m$ is odd: $\widetilde S = (\mu_1, \ldots, \mu_m, 2n)$, $\widetilde\alpha = \alpha \cdot i^m \cdot (-1)^{m(m-1)/2}$.

Then

$$
\langle P\rangle = \widetilde\alpha \cdot i^{-|\widetilde S|/2} \cdot \mathrm{Pf}(\widetilde\Lambda|_{\widetilde S}).
$$

The submatrix $\widetilde\Lambda|_{\widetilde S}$ is always of even size ($m$ or $m+1$ depending
on parity), so the Pfaffian is always well-defined. The cost per Pauli term is $O(m^3)$
where $m$ is the JW Majorana rank of the term.

**Hamiltonian.** For $\mathcal{H} = \sum_j \alpha_j P_j$, the expectation value is
computed term by term:

$$
\langle\mathcal{H}\rangle = \sum_j \alpha_j \langle P_j\rangle.
$$

Each term requires one submatrix extraction and one Pfaffian computation.

## Pfaffian algorithm

The Pfaffian is computed via the Parlett-Reid skew-tridiagonalization [4] with partial
pivoting, which gives a signed Pfaffian in $O(m^3)$ time and is numerically stable even
when the matrix is singular (the Pfaffian is then zero).

## Complexity summary

| Quantity | Basis-state path | Arbitrary product-state path |
|---|---|---|
| Covariance matrix size | $2n \times 2n$ | $(2n+1) \times (2n+1)$ |
| SPTM size | $2n \times 2n$ | $(2n+1) \times (2n+1)$, block $Q \oplus 1$ |
| Per-Pauli cost | $O(m^3)$ | $O((m+1)^3)$ |
| Initial-state class | Computational basis | Arbitrary qubit product state |

Asymptotically, extending to arbitrary product states adds one dimension — negligible in
practice.

## What this does not solve

- **Non-product initial states.** GHZ states, magic states, or any entangled initial state
  are not covered. The extension lifts Gaussianity from $L_2$ (quadratic algebra) to
  $L_{1,2}$ (linear+quadratic), not to arbitrary algebras.
- **Mid-circuit non-Gaussian gates.** If the circuit contains non-matchgate operations,
  the SPTM machinery breaks down.
- **Sampling.** This algorithm is specific to expectation values. Sampling from the output
  distribution is a separate problem.

## References

[1] Jozsa, Richard, Akimasa Miyake, and Sergii Strelchuk. "Jordan-Wigner formalism for arbitrary 2-input 2-output
matchgates and their classical simulation." arXiv preprint arXiv:1311.3046 (2013).

[2] Projansky, Andrew M., Jason Necaise, and James D. Whitfield. "Gaussianity and simulability of Cliffords and
matchgates." Journal of Physics A: Mathematical and Theoretical 58, no. 19 (2025): 195302.

[3] Bravyi, Sergey. "Lagrangian representation for fermionic linear optics." arXiv preprint quant-ph/0404180 (2004).

[4] W. H. Press, S. A. Teukolsky, W. T. Vetterling & B. P. Flannery, *Numerical Recipes*,
3rd ed., Cambridge University Press (2007).
