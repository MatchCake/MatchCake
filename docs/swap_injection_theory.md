# Simulating matchgate circuits with a few SWAP gates: the branch-tensor formalism

A pure matchgate circuit is free-fermionic and classically simulable in polynomial time through
the Majorana covariance matrix (see the companion note on covariance-matrix expectation values).
A genuine qubit SWAP breaks that structure: matchgates together with SWAP are universal for
quantum computation [2], so no single covariance matrix can carry the state once a SWAP acts. This
note develops the formalism that restores tractability when the number $m$ of genuine SWAPs is
small. The state is represented as a superposition of at most $2^m$ fermionic Gaussian states, that
is, a *branch tensor* of covariance matrices together with a Hermitian weight matrix, and every
observable reduces to a weighted sum of Pfaffians over pairs of branches. The cost is polynomial
in the qubit number $N$ and exponential only in $m$.

We reuse the conventions of the companion note throughout, and write $\langle A\rangle_\rho =
\mathrm{Tr}[\rho A]$.

## 1. Setting: Majoranas, covariance, and the matchgate rotation

The Jordan–Wigner (JW) transformation maps $N$ qubits to $2N$ Hermitian Majorana operators
(0-indexed),

$$
c_{2k} = Z_0\cdots Z_{k-1}\,X_k, \qquad c_{2k+1} = Z_0\cdots Z_{k-1}\,Y_k,
$$

obeying $\{c_\mu, c_\nu\} = 2\delta_{\mu\nu}$ and $c_{2k}c_{2k+1} = iZ_k$. The number operator is
therefore

$$
n_k = \frac{1 - Z_k}{2} = \frac{1 + i\,c_{2k}c_{2k+1}}{2}.
$$

A pure state $|\phi\rangle$ is described, in the fermionic linear-optics formalism [1, 3, 4], by its
real antisymmetric covariance matrix

$$
\Lambda_{\mu\nu} = i\,\langle\phi|\,c_\mu c_\nu\,|\phi\rangle \quad (\mu\neq\nu), \qquad
\Lambda_{\mu\mu} = 0.
$$

For a computational-basis state $|y\rangle$, $y\in\{0,1\}^N$, only the on-site blocks survive,

$$
(\Lambda_y)_{2k,2k+1} = 2y_k - 1 = -(\Lambda_y)_{2k+1,2k},
\qquad
\mathrm{Pf}(\Lambda_y) = \prod_k (2y_k - 1).
$$

A matchgate unitary $U$ rotates the Majoranas among themselves, $U^\dagger c_\mu U = \sum_\nu
Q_{\mu\nu} c_\nu$ with $Q\in O(2N)$ the single-particle transition matrix (SPTM) [1]. The covariance
then evolves by the congruence

$$
\boxed{\;\Lambda \;\longrightarrow\; Q^\top \Lambda\, Q\;}
$$

A circuit built entirely of matchgates is thus a single orthogonal rotation of one covariance
matrix: free-fermionic, polynomial in $N$. Everything below concerns what happens when that
structure is interrupted by SWAPs.

**Running example.** We follow one small circuit through every section of this note. Take $N=3$
qubits, the input $|011\rangle$, and a single matchgate applied before any SWAP: the
number-conserving hopping (Givens rotation) $V_1 = \exp[-i\tfrac{\theta}{2}(X_0X_1 + Y_0Y_1)]$ on
qubits $0$ and $1$, with $\theta=\pi/3$. It acts on the input as
$V_1|011\rangle = \tfrac12|011\rangle - \tfrac{i\sqrt3}{2}|101\rangle$, moving three quarters of the
particle from qubit $1$ to qubit $0$, so that $\langle n_0\rangle=\tfrac34$,
$\langle n_1\rangle=\tfrac14$, and $\langle n_2\rangle=1$. With $\cos\theta=\tfrac12$ and
$\sin\theta=\tfrac{\sqrt3}{2}$, the congruence $\Lambda\to Q^\top\Lambda Q$ takes the input covariance
to

$$
\Lambda_{\mathrm{in}} =
\begin{pmatrix}
0 & \tfrac12 & -\tfrac{\sqrt3}{2} & 0 & 0 & 0\\
-\tfrac12 & 0 & 0 & -\tfrac{\sqrt3}{2} & 0 & 0\\
\tfrac{\sqrt3}{2} & 0 & 0 & -\tfrac12 & 0 & 0\\
0 & \tfrac{\sqrt3}{2} & \tfrac12 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 1\\
0 & 0 & 0 & 0 & -1 & 0
\end{pmatrix}.
$$

As long as the circuit stays matchgate-only, this single $6\times6$ covariance is the entire state.
In the next sections we insert a genuine SWAP, which breaks this single-covariance description.

## 2. A SWAP is not a matchgate

Write the qubit SWAP on the pair $(j,k)$ as

$$
\mathrm{SWAP}_{jk} = \mathrm{fSWAP}_{jk}\,\mathrm{CZ}_{jk}.
$$

The fermionic swap $\mathrm{fSWAP}_{jk}$ *is* a matchgate: it is the signed permutation exchanging
the Majorana pairs $(c_{2j},c_{2j+1})\leftrightarrow(c_{2k},c_{2k+1})$, with a well-defined SPTM.
The remaining controlled-$Z$ factor is the obstruction. Using $n_k = (1 + i\,c_{2k}c_{2k+1})/2$,

$$
\mathrm{CZ}_{jk} = 1 - 2\,n_j n_k.
$$

The product $n_j n_k$ is *quartic* in Majoranas, so conjugation by $\mathrm{CZ}$ sends a single
Majorana to a cubic monomial, e.g. $\mathrm{CZ}\,c_{2j}\,\mathrm{CZ} = c_{2j}Z_k = -i\,c_{2j}
c_{2k}c_{2k+1}$. No $2N\times 2N$ orthogonal matrix implements this, so no single covariance
matrix can survive a $\mathrm{CZ}$. This is the precise sense in which SWAP leaves the
free-fermion manifold.

The two-factor decomposition above holds for adjacent wires, $k = j + 1$. A qubit SWAP on a
non-adjacent pair also picks up the parity phase of every mode it crosses,

$$
\mathrm{SWAP}_{jk} = \mathrm{fSWAP}_{jk}\,\mathrm{CZ}_{jk} \prod_{j<l<k} \mathrm{CZ}_{jl}\,
\mathrm{CZ}_{kl},
$$

where $\mathrm{fSWAP}_{jk}$ is the (Gaussian) fermionic mode swap, the sign
$(-1)^{n_j n_k + (n_j + n_k)\sum_{j<l<k} n_l}$ of reordering the crossed modes being supplied by
the $\mathrm{CZ}$ factors. A distance-$d$ SWAP therefore costs $2d - 1$ genuine $\mathrm{CZ}$
factors (each a branching of §7), while a non-adjacent $\mathrm{CZ}$ costs one, like an adjacent
one: $n_j n_k$ is string-free. Everything below is written for one $\mathrm{CZ}$ at a time and
covers both cases.

**Running example.** We now insert a genuine $\mathrm{SWAP}_{12}$ right after $V_1$, so the circuit
is $V_1$ followed by $\mathrm{SWAP}_{12} = \mathrm{fSWAP}_{12}\,(1 - 2 n_1 n_2)$. The
$\mathrm{fSWAP}_{12}$ factor is a matchgate and preserves the covariance description, but the
$\mathrm{CZ}_{12} = 1 - 2 n_1 n_2$ factor is quartic in the Majoranas. No $6\times6$ orthogonal $Q$
reproduces its action, so the single $\Lambda_{\mathrm{in}}$ of §1 can no longer carry the state once
the SWAP acts. We have to branch.

## 3. The sum-of-Gaussians (branch) decomposition

Substitute $\mathrm{CZ}_{jk} = 1 - 2 n_j n_k$ for every SWAP. Each $\mathrm{CZ}$ is a sum of two
*Gaussian operators*: the identity and $-2\,n_jn_k$. A product of Gaussian operators is Gaussian,
and a Gaussian operator maps a fermionic Gaussian state to another (unnormalized) Gaussian state.
A circuit with $m$ SWAPs interleaved among matchgate layers therefore expands into a sum of at
most $2^m$ operator strings acting on the Gaussian input, yielding a superposition of Gaussian
pure states [6, 7]

$$
\boxed{\;
|\psi\rangle = \sum_{\alpha=1}^{\chi} \lambda_\alpha\,|\phi_\alpha\rangle,
\qquad \chi \le 2^m,
\;}
$$

where each $|\phi_\alpha\rangle$ is a normalized fermionic Gaussian state (its norm absorbed into
$\lambda_\alpha$). Stacking the per-branch covariances $\Lambda_\alpha$ gives the **covariance
tensor** $\Lambda \in \mathbb{R}^{\chi\times 2N\times 2N}$.

The tensor alone does **not** determine $|\psi\rangle$. A covariance fixes its Gaussian state only
up to a global phase, and distinct branches interfere coherently; the relative phases are physical
data that $\Lambda$ cannot store. We therefore also carry the Hermitian **weight matrix**

$$
\boxed{\;
W_{\alpha\beta} = \lambda_\alpha^*\,\lambda_\beta\,\langle\phi_\alpha|\phi_\beta\rangle.
\;}
$$

$W$ is positive semidefinite, with diagonal $W_{\alpha\alpha} = |\lambda_\alpha|^2$ and total
$\sum_{\alpha\beta} W_{\alpha\beta} = \langle\psi|\psi\rangle = 1$. The complete state data is the
pair $(\Lambda, W)$; this is the covariance-plus-relative-phase representation of a Gaussian
superposition [6]. All observables below are quadratic in $|\psi\rangle$,

$$
\langle\psi|O|\psi\rangle = \sum_{\alpha\beta}
\lambda_\alpha^*\lambda_\beta\,\langle\phi_\alpha|O|\phi_\beta\rangle,
$$

so we need matrix elements of Majorana monomials between two *different* Gaussian states.

**Running example.** With a single SWAP the sum has at most $\chi=2^1=2$ branches. Splitting on
$\mathrm{CZ}_{12}=1-2n_1n_2$ and then applying $\mathrm{fSWAP}_{12}$ to each piece (the explicit steps
are worked out in §7) produces the covariance tensor $\Lambda=(\Lambda_0,\Lambda_1)$ with

$$
\Lambda_0 =
\begin{pmatrix}
0 & \tfrac12 & 0 & 0 & -\tfrac{\sqrt3}{2} & 0\\
-\tfrac12 & 0 & 0 & 0 & 0 & -\tfrac{\sqrt3}{2}\\
0 & 0 & 0 & 1 & 0 & 0\\
0 & 0 & -1 & 0 & 0 & 0\\
\tfrac{\sqrt3}{2} & 0 & 0 & 0 & 0 & -\tfrac12\\
0 & \tfrac{\sqrt3}{2} & 0 & 0 & \tfrac12 & 0
\end{pmatrix},
\qquad
\Lambda_1 =
\begin{pmatrix}
0 & -1 & 0 & 0 & 0 & 0\\
1 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 1 & 0 & 0\\
0 & 0 & -1 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 1\\
0 & 0 & 0 & 0 & -1 & 0
\end{pmatrix},
$$

and the Hermitian weight matrix

$$
W = \begin{pmatrix} 1 & -\tfrac12 \\[2pt] -\tfrac12 & 1 \end{pmatrix}.
$$

Branch $0$ is a genuine (non-basis) Gaussian, while branch $1$ happens to be the computational-basis
state $|011\rangle$. The diagonal entries $W_{00}=W_{11}=1$ are the branch norms $|\lambda_\alpha|^2$,
the off-diagonal $W_{01}=-\tfrac12$ holds the (here real) relative-phase data, and
$\sum_{\alpha\beta}W_{\alpha\beta}=1$. We take this pair $(\Lambda,W)$ as given in §4 through §6, and
rebuild it from scratch in §7.

## 4. The transition covariance and the two-state Wick theorem

Section 3 ended with a concrete need: every observable is a double sum over branch pairs, and each
term is a matrix element $\langle\phi_\alpha|c_{\mu_1}\cdots c_{\mu_{2t}}|\phi_\beta\rangle$ of a
Majorana monomial between two *distinct* Gaussian branches. Recall why the single-state case is
easy. A Gaussian state is the exponential of a Majorana quadratic, so all of its higher moments
factorize into pairwise contractions: Wick's theorem collapses any monomial to a Pfaffian of the
covariance restricted to the monomial's support. One two-point object, $\Lambda$, controls every
expectation value. Here both the bra and the ket are still exponentials of quadratics, only no
longer the *same* one, so the same collapse must occur; the only question is which two-point object
replaces $\Lambda$. The natural candidate is the matrix of transition two-point functions between
the two branches, the **transition covariance**

$$
(\Gamma_{\alpha\beta})_{\mu\nu} = i\,\frac{\langle\phi_\alpha|\,c_\mu c_\nu\,|\phi_\beta\rangle}
{\langle\phi_\alpha|\phi_\beta\rangle}.
$$

This is defined exactly like the ordinary covariance $\Lambda_{\mu\nu} = i\langle c_\mu c_\nu\rangle$,
but with the bra and ket drawn from different branches and the overlap divided out; on the diagonal
it collapses to the ordinary covariance, $\Gamma_{\alpha\alpha} = \Lambda_\alpha$. Because both
$\langle\phi_\alpha|$ and $|\phi_\beta\rangle$ are exponentials of Majorana quadratics, every higher
transition matrix element factorizes through this single contraction; that factorization is the
two-state Wick theorem stated at the end of this section. To use it we first need
$\Gamma_{\alpha\beta}$ as an explicit function of the two covariances $\Lambda_\alpha$ and
$\Lambda_\beta$.

The explicit form comes from the Grassmann (Lagrangian) representation of fermionic linear optics
[1], and it is worth seeing where each piece originates. In that representation every even operator
becomes a polynomial in $2N$ anticommuting variables $\theta_\mu$, one per Majorana, and a Gaussian
state becomes a Gaussian kernel whose quadratic form *is* its covariance,

$$
\omega_\alpha(\theta) = 2^{-N}\exp\!\bigl(\tfrac{i}{2}\,\theta^\top \Lambda_\alpha\,\theta\bigr),
$$

while matrix elements become Berezin integrals of products of kernels. The element
$\langle\phi_\alpha|c_\mu c_\nu|\phi_\beta\rangle$ therefore involves exactly two kernels, one for
the bra (form $\Lambda_\alpha$) and one for the ket (form $\Lambda_\beta$), glued by a Gaussian
pairing term. Integrating out the ket's variables against that pairing is itself a Gaussian
integral; by the purity relation $\Lambda_\beta^2 = -I$ it reproduces the form $\Lambda_\beta$ on
the surviving variables, so the two exponents simply *add* and a single Grassmann Gaussian remains,
with the combined quadratic form

$$
\tfrac{i}{2}\,\theta^\top(\Lambda_\alpha + \Lambda_\beta)\,\theta .
$$

This answers how two covariance matrices combine: by addition of their quadratic forms. Moments of
the combined Gaussian are then read off the *inverse* of that form, through the standard Gaussian
integral

$$
\int\! \mathcal D\theta\, e^{\frac12\theta^\top M\theta + \eta^\top\theta} \propto
\mathrm{Pf}(M)\,e^{-\frac12\eta^\top M^{-1}\eta}.
$$

Read it twice. With no insertions ($\eta = 0$) it produces the Pfaffian of the combined form; this
is exactly where the overlap-magnitude identity closing this section comes from. With a pair of
insertions $c_\mu c_\nu$, differentiating twice with respect to $\eta$ pulls down an entry of
$M^{-1}$; dividing the two-insertion integral by the zero-insertion one, which is what the overlap
denominator in the definition of $\Gamma_{\alpha\beta}$ does, cancels the Pfaffian and leaves the
entry of $(\Lambda_\alpha+\Lambda_\beta)^{-1}$. Keeping track of whether each insertion attaches to
the bra kernel (bringing down $\Lambda_\alpha$), to the ket kernel (bringing down $\Lambda_\beta$),
or to the pairing term assembles the remaining linear-in-$\Lambda$ factor, and the result [1] is,
with $\Lambda_\alpha+\Lambda_\beta$ invertible (the singular case is handled in §7 and §10),

$$
\boxed{\;
\Gamma_{\alpha\beta} =
\bigl(i(\Lambda_\alpha - \Lambda_\beta) - 2I\bigr)
\,(\Lambda_\alpha + \Lambda_\beta)^{-1}.
\;}
$$

The structure is easiest to read after splitting the numerator into its two parts,

$$
\Gamma_{\alpha\beta} = -2\,(\Lambda_\alpha + \Lambda_\beta)^{-1}
+ i\,(\Lambda_\alpha - \Lambda_\beta)(\Lambda_\alpha + \Lambda_\beta)^{-1}.
$$

The first term is the bare moment of the combined Gaussian, symmetric under exchanging the two
branches; the second flips sign under the exchange and records which side of the matrix element
each insertion sits on. Setting $\beta = \alpha$ is now an immediate check: the second term
vanishes, and the first reduces to $-2(2\Lambda_\alpha)^{-1} = -\Lambda_\alpha^{-1} =
\Lambda_\alpha$ by the purity relation $\Lambda_\alpha^2 = -I$, recovering
$\Gamma_{\alpha\alpha} = \Lambda_\alpha$ as required. $\Gamma_{\alpha\beta}$ is complex
antisymmetric (antisymmetry also rests on purity, through the anticommutator identity
$\{\Lambda_\alpha - \Lambda_\beta,\,\Lambda_\alpha + \Lambda_\beta\} = 2(\Lambda_\alpha^2 -
\Lambda_\beta^2) = 0$) and satisfies $\Gamma_{\beta\alpha} = \Gamma_{\alpha\beta}^{*}$.

The same object has a factored form in terms of the (non-Hermitian) projectors
$P_\alpha = (I + i\Lambda_\alpha)/2$ and $\bar P_\alpha = (I - i\Lambda_\alpha)/2$, which for a
pure covariance are complementary idempotents: $P_\alpha^2 = P_\alpha$, $\bar P_\alpha^2 = \bar
P_\alpha$, $P_\alpha\bar P_\alpha = 0$. Two substitutions, both direct consequences of the
definitions, do most of the work:

$$
\Lambda_\alpha + iI = 2i\,\bar P_\alpha,
\qquad
\Lambda_\alpha + \Lambda_\beta = 2i\,(\bar P_\alpha - P_\beta).
$$

Starting from the closed form and clearing the $i$'s,

$$
I - i\,\Gamma_{\alpha\beta}
= \bigl[(\Lambda_\alpha + \Lambda_\beta) + (\Lambda_\alpha - \Lambda_\beta) + 2iI\bigr]
(\Lambda_\alpha + \Lambda_\beta)^{-1}
= 2\,(\Lambda_\alpha + iI)(\Lambda_\alpha + \Lambda_\beta)^{-1}
= 2\,\bar P_\alpha\,(\bar P_\alpha - P_\beta)^{-1},
$$

and a last rearrangement with the idempotents (using $\bar P_\beta P_\beta = 0$; it holds whenever
$\Lambda_\alpha + \Lambda_\beta$ is invertible) symmetrizes the inverse:

$$
I - i\,\Gamma_{\alpha\beta} = 2\,\bar P_\alpha\,(\bar P_\alpha + P_\beta)^{-1}\,\bar P_\beta .
$$

The right-hand side is the standard transition (two-state) correlation matrix of fermionic linear
optics [1, 3]. Off the diagonal, $\Gamma_{\alpha\beta}$ therefore agrees entry by entry with
$i\cdot 2\bar P_\alpha(\bar P_\alpha + P_\beta)^{-1}\bar P_\beta$, which is the convention used on
the lifted path in §9.2, and the kernel $I - i\Gamma$ on the left is exactly the contraction kernel
that reappears in the Wick rule of §11.

With $\Gamma_{\alpha\beta}$ in hand, the collapse of arbitrary monomials is the two-state Wick
theorem, generalizing the Pfaffian form of Wick's theorem for a single Gaussian state [1]: for a
sorted index set $S = \{\mu_1 < \cdots < \mu_{2t}\}$,

$$
\boxed{\;
\frac{\langle\phi_\alpha|\,c_{\mu_1}\cdots c_{\mu_{2t}}\,|\phi_\beta\rangle}
{\langle\phi_\alpha|\phi_\beta\rangle}
= i^{-t}\,\mathrm{Pf}\!\left(\Gamma_{\alpha\beta}|_S\right),
\;}
$$

where $t = |S|/2$ is the number of Majorana pairs in the monomial and $\Gamma_{\alpha\beta}|_S$ is the
principal submatrix on the rows and columns in $S$. Odd-length
$S$ gives zero by fermion-parity superselection (all branches here carry equal, definite parity,
since matchgates and $n_jn_k$ are parity-even). The diagonal $\alpha=\beta$ is exactly the
single-state Pfaffian–Wick theorem.

The overlap that appears as the denominator is fixed in *magnitude* by the covariances, through the
zero-insertion Pfaffian noted above [1],

$$
\bigl|\langle\phi_\alpha|\phi_\beta\rangle\bigr|^2
= 2^{-N}\,\mathrm{Pf}(\Lambda_\alpha)\,\mathrm{Pf}(\Lambda_\alpha + \Lambda_\beta),
$$

but its *phase* is not. That phase is precisely the information held in the weight matrix $W$ of §3:
the covariances fix $|W_{\alpha\beta}|$ through this identity, while the phase of $W_{\alpha\beta}$
cannot be recovered from them, and is instead carried by $W$ as independent propagated data.

**Running example.** For the two branches of §3, the closed form gives the transition covariance

$$
\Gamma_{01} =
\begin{pmatrix}
0 & -1 & 0 & 0 & -\sqrt3 & i\sqrt3\\
1 & 0 & 0 & 0 & -i\sqrt3 & -\sqrt3\\
0 & 0 & 0 & 1 & 0 & 0\\
0 & 0 & -1 & 0 & 0 & 0\\
\sqrt3 & i\sqrt3 & 0 & 0 & 0 & 1\\
-i\sqrt3 & \sqrt3 & 0 & 0 & -1 & 0
\end{pmatrix},
$$

which is complex antisymmetric and satisfies $\Gamma_{10}=\Gamma_{01}^{*}$. Its $\{2,3\}$ block is
$\bigl(\begin{smallmatrix}0&1\\-1&0\end{smallmatrix}\bigr)$, the value both branches share on qubit
$1$ (both leave it occupied after the fSWAP), while the genuinely complex entries $\pm i\sqrt3$ encode
the interference between the non-basis branch $0$ and the basis branch $1$. The diagonal cases reduce
to the ordinary covariances, $\Gamma_{00}=\Lambda_0$ and $\Gamma_{11}=\Lambda_1$.

## 5. Hamiltonian expectation values

Let $\mathcal H = \sum_P h_P\,P$ be a Pauli sum. The JW dictionary [1] writes each Pauli word as one
ordered Majorana monomial up to a unit-modulus phase,

$$
P = \kappa_P\,c_{\mu_1}\cdots c_{\mu_{m_P}}, \qquad
S_P = \{\mu_1 < \cdots < \mu_{m_P}\}, \qquad \kappa_P \in \{1, i, -1, -i\}.
$$

Inserting this and the two-state Wick theorem into the quadratic form of §3, and recognizing the
prefactor $\lambda_\alpha^*\lambda_\beta\langle\phi_\alpha|\phi_\beta\rangle = W_{\alpha\beta}$,
gives

$$
\boxed{\;
\langle\mathcal H\rangle = \sum_{\alpha\beta} W_{\alpha\beta}
\sum_P h_P\,\kappa_P\,i^{-m_P/2}\,
\mathrm{Pf}\!\left(\Gamma_{\alpha\beta}|_{S_P}\right).
\;}
$$

Odd-weight terms ($m_P$ odd) drop out. The diagonal terms $\alpha=\beta$ reproduce the ordinary
single-state matchgate expectation value of the companion note, since
$\Gamma_{\alpha\alpha} = \Lambda_\alpha$; the off-diagonal terms are the genuinely new
inter-branch interference.

**Running example.** Take the data $(\Lambda,W)$ of §3 as given and measure the simplest observable,
$\mathcal H = Z_0$. Its Jordan-Wigner image is $Z_0 = -i\,c_0 c_1$, so $\kappa_P=-i$, $S_P=\{0,1\}$,
$m_P=2$, and $i^{-m_P/2}=i^{-1}=-i$. The prefactor is $\kappa_P\,i^{-1}=(-i)(-i)=-1$, and
$\mathrm{Pf}(\Gamma_{\alpha\beta}|_{\{0,1\}})=(\Gamma_{\alpha\beta})_{01}$. Reading the $01$ entry off
each branch pair ($\tfrac12$, $-1$, $-1$, $-1$ for $(0,0),(0,1),(1,0),(1,1)$) gives

$$
\langle Z_0\rangle
= -\sum_{\alpha\beta} W_{\alpha\beta}\,(\Gamma_{\alpha\beta})_{01}
= -\Bigl[\,
\underbrace{(1)(\tfrac12) + (1)(-1)}_{\text{diagonal}\,=\,-\frac12}
+ \underbrace{(-\tfrac12)(-1) + (-\tfrac12)(-1)}_{\text{interference}\,=\,+1}
\,\Bigr]
= -\tfrac12 .
$$

Keeping only the diagonal terms would give $-(-\tfrac12)=+\tfrac12$, the wrong sign; the inter-branch
interference corrects it to $\langle Z_0\rangle=-\tfrac12$, consistent with $\langle n_0\rangle=\tfrac34$
through $Z=1-2n$. The same evaluation gives $\langle Z_2\rangle=+\tfrac12$.

## 6. Basis-state outcome probabilities

The projector onto a full computational outcome $y$ is $P_y = |y\rangle\langle y| = \prod_k(n_k$
if $y_k=1$ else $1-n_k)$. Each factor is the single-mode Gaussian operator
$(1 + (2y_k-1)\,i\,c_{2k}c_{2k+1})/2$, whose covariance block is $(\Lambda_y)_{2k,2k+1} = 2y_k-1$.
The Gaussian overlap-with-a-reference-covariance identity [1] collapses the $2^N$-term sector sum into
a single Pfaffian,

$$
\frac{\langle\phi_\alpha|P_y|\phi_\beta\rangle}{\langle\phi_\alpha|\phi_\beta\rangle}
= 2^{-N}\,\mathrm{Pf}(\Lambda_y)\,\mathrm{Pf}\!\left(\Gamma_{\alpha\beta} + \Lambda_y\right),
$$

so that, summing over branches with the weights,

$$
\boxed{\;
p(y) = \langle\psi|P_y|\psi\rangle
= \frac{\mathrm{Pf}(\Lambda_y)}{2^{N}}
\sum_{\alpha\beta} W_{\alpha\beta}\,
\mathrm{Pf}\!\left(\Gamma_{\alpha\beta} + \Lambda_y\right).
\;}
$$

**Running example.** Still using the $(\Lambda,W)$ of §3, ask for the probability of the outcome
$y=110$. Here $\mathrm{Pf}(\Lambda_y)=\prod_k(2y_k-1)=(+1)(+1)(-1)=-1$ and $2^N=8$. Evaluating
$\mathrm{Pf}(\Gamma_{\alpha\beta}+\Lambda_y)$ on the four pairs, only the $(0,0)$ pair is nonzero (it
equals $-6$); the others vanish because branch $1$ is the state $|011\rangle$, which is orthogonal to
$|110\rangle$. Hence

$$
p(110) = \frac{-1}{8}\Bigl[(1)(-6) + (-\tfrac12)(0) + (-\tfrac12)(0) + (1)(0)\Bigr]
= \frac{6}{8} = \tfrac34 .
$$

The only other reachable outcome is $y=011$, for which all four pairs contribute and the interference
is essential:
$p(011)=\tfrac{-1}{8}\bigl[(1)(-2)+(-\tfrac12)(-8)+(-\tfrac12)(-8)+(1)(-8)\bigr]=\tfrac{-1}{8}(-2)=\tfrac14$.
The two probabilities sum to one, matching the exact final state
$\tfrac12|011\rangle - \tfrac{i\sqrt3}{2}|110\rangle$.

## 7. Updating the data through a SWAP

The branch data $(\Lambda, W)$ is propagated gate by gate rather than by expanding the full
$2^m$-term product symbolically. Matchgate layers are the easy case: they act by the SPTM
congruence of §1 on every branch covariance and leave $W$ untouched, since a common matchgate
changes neither the branch coefficients nor any pairwise overlap. A SWAP is processed through its
decomposition $\mathrm{SWAP}_{jk} = \mathrm{fSWAP}_{jk}\,(1 - 2 n_j n_k)$: first the $\mathrm{CZ}$
factor, which branches, then the $\mathrm{fSWAP}$ factor, which is a matchgate. Suppose the current
state has $\chi$ branches. One SWAP is then the following six steps, applied in this order; at each
step we say which of the two objects changes and which stays fixed.

**Step 1: branch doubling on the $\mathrm{CZ}$.** The $\mathrm{CZ}$ factor splits every branch in
two, $|\phi_\alpha\rangle \mapsto |\phi_\alpha\rangle - 2\,n_jn_k|\phi_\alpha\rangle$, so the branch
count doubles. The first piece is the untouched "type-0" branch. The second piece is, up to
normalization, the occupation-projected "type-1" branch $n_jn_k|\phi_\alpha\rangle/\sqrt{q_\alpha}$,
where $q_\alpha = \langle\phi_\alpha|n_jn_k|\phi_\alpha\rangle$ is the joint occupation (computed in
Step 3). The scalar coefficient $-2$ and the normalization $\sqrt{q_\alpha}$ are *not* stored in
any covariance; they belong to the weights and are booked in Step 4. After this step the covariance
tensor has $2\chi$ slots, the first $\chi$ holding the old covariances unchanged and the last
$\chi$ holding copies awaiting the projection of Step 2. $W$ is not yet touched.

**Step 2: conditioning the type-1 covariances.** Only the type-1 covariances change in this step.
Pinning modes $j,k$ to occupied is a Gaussian projection [1]: it replaces the local state of qubits
$j,k$ by $|1\rangle_j|1\rangle_k$ and propagates the back-action of that projection onto every
other mode through whatever entanglement was present. Group the $2N$ Majorana indices into the two
pinned pairs and the rest,

$$
S_4 = \{2j,\,2j+1,\,2k,\,2k+1\},
\qquad
R = \{0,\dots,2N-1\}\setminus S_4,
$$

so $R$ (the **rest**) is simply every Majorana mode that is *not* being pinned; on the lifted path
of §9 the two ancilla rows and columns ride along inside $R$. Split the branch covariance into the
corresponding blocks,

$$
\Lambda_\alpha =
\begin{pmatrix} A & B \\ -B^\top & C \end{pmatrix}
\ \text{on } (S_4, R),
\qquad
A = \Lambda_\alpha\big|_{S_4},\quad
C = \Lambda_\alpha\big|_{R},\quad
B = \Lambda_\alpha\big|_{S_4, R},
$$

so $A$ ($4\times4$) is the covariance of the pinned modes, $C$ that of the rest, and the cross-block
$B$ holds the entanglement between them. Let $\Lambda_{\mathrm{occ}}$ be the covariance of the
target local configuration $|1\rangle_j|1\rangle_k$; it is exactly the $S_4$ block of the basis
covariance $\Lambda_y$ of §1 with $y_j=y_k=1$, the block-diagonal $4\times4$ matrix

$$
\Lambda_{\mathrm{occ}} =
\begin{pmatrix} 0&1&0&0\\ -1&0&0&0\\ 0&0&0&1\\ 0&0&-1&0 \end{pmatrix},
\qquad
(\Lambda_{\mathrm{occ}})_{2j,2j+1} = (\Lambda_{\mathrm{occ}})_{2k,2k+1} = +1 .
$$

The projected covariance is then

$$
\boxed{\;
\Lambda'_\alpha\big|_{S_4} = \Lambda_{\mathrm{occ}}, \qquad
\Lambda'_\alpha\big|_{S_4, R} = 0, \qquad
\Lambda'_\alpha\big|_{R} =
\underbrace{C}_{\text{prior}}
+ \underbrace{B^\top (A + \Lambda_{\mathrm{occ}})^{-1} B}_{\text{projection back-action}} .
\;}
$$

The first two pieces are immediate: the pinned modes now carry the definite occupied configuration
$\Lambda_{\mathrm{occ}}$, and projecting them onto a product state severs their correlations with
everything else, so the $S_4$–$R$ cross block is zeroed. The third piece, the only nontrivial one,
is the fermionic Schur-complement update of Gaussian conditioning, and its origin is the same
Grassmann mechanism as in §4: writing $\Lambda_\alpha$ as a Grassmann Gaussian and inserting the
projector onto $|11\rangle_{jk}$ (itself a Gaussian whose kernel on $S_4$ is
$\Lambda_{\mathrm{occ}}$) produces a single combined quadratic form, and integrating out the pinned
Grassmann variables leaves the rest with covariance $C$ corrected by the Schur complement of the
combined pinned block $A+\Lambda_{\mathrm{occ}}$ through the coupling $B$ [1]. This is the exact
fermionic analogue of classical Gaussian conditioning, where conditioning a joint Gaussian on a
subset of variables corrects the remaining covariance by $-\Sigma_{RS}\Sigma_{SS}^{-1}\Sigma_{SR}$,
and the shift $A\to A+\Lambda_{\mathrm{occ}}$ is precisely what turns a marginalization into a
projection onto the specific occupied outcome. When $B=0$, meaning the pinned qubits are
unentangled from the rest, the back-action vanishes and $\Lambda'_\alpha|_R = C$: pinning
already-product modes leaves everything else untouched, as it must. The type-0 covariances and $W$
are not modified by this step; the running-example build below shows the Schur term in action.

**Step 3: cross occupations.** The weight update of Step 4 needs, for every pair of old branches,
the matrix element $\langle\phi_\alpha|n_jn_k|\phi_\beta\rangle$. Expanding
$n_jn_k = (1 + i\,c_{2j}c_{2j+1})(1 + i\,c_{2k}c_{2k+1})/4$ and applying the two-state Wick theorem
of §4 gives the **cross-occupation matrix**

$$
q_{\alpha\beta} =
\frac{\langle\phi_\alpha|n_jn_k|\phi_\beta\rangle}{\langle\phi_\alpha|\phi_\beta\rangle}
= \tfrac{1}{4}\Bigl[\,1 + (\Gamma_{\alpha\beta})_{2j,2j+1} + (\Gamma_{\alpha\beta})_{2k,2k+1} + \mathrm{Pf}\!\left(\Gamma_{\alpha\beta}|_{S_4}\right)\Bigr],
$$

with $q_\alpha = q_{\alpha\alpha}\in[0,1]$ real. Here $\Gamma_{\alpha\beta}$ is built from the
*pre-SWAP* covariances (the ones now held in the type-0 slots): $q$ is a property of the state
before the $\mathrm{CZ}$ acts. Its diagonal supplies the normalizations of Step 1, and the full
matrix feeds every block of Step 4. Nothing in $(\Lambda, W)$ is modified by this step; $q$ is
auxiliary data.

**Step 4: weight update.** Collect the new branch coefficients: a type-0 branch keeps
$\lambda_\alpha$, while a type-1 branch carries $\lambda'_\alpha = -2\sqrt{q_\alpha}\,
\lambda_\alpha$, the $-2$ coming from the $\mathrm{CZ}$ expansion and the $\sqrt{q_\alpha}$ from
the normalization of Step 1. Inserting these into the definition $W_{\alpha\beta} =
\lambda_\alpha^*\lambda_\beta\langle\phi_\alpha|\phi_\beta\rangle$, block by block, gives the
$2\chi\times 2\chi$ form

$$
\boxed{\;
W_{\mathrm{new}} =
\begin{pmatrix}
W & -2\,W\!\odot q \\[2pt]
-2\,W\!\odot q & 4\,W\!\odot q
\end{pmatrix},
\;}
$$

with $\odot$ the elementwise (Hadamard) product against the $\chi\times\chi$ matrix
$q_{\alpha\beta}$. The four blocks read as follows. In the top-left block (type-0 with type-0) both
coefficients and the overlap are unchanged, so the block is the old $W$. In the off-diagonal
blocks (type-0 with type-1) a single factor $-2$ appears, and the overlap
$\langle\phi_\alpha|n_jn_k|\phi_\beta\rangle = q_{\alpha\beta}\langle\phi_\alpha|\phi_\beta\rangle$
turns $W$ into $W\!\odot q$, the $\sqrt{q}$ of the coefficient cancelling against the normalization
of the type-1 state. In the bottom-right block (type-1 with type-1) the coefficients contribute
$(-2)^2 = 4$, and because $n_jn_k$ is a projector, $(n_jn_k)^2 = n_jn_k$, the double insertion
collapses to the same single $q_{\alpha\beta}$, giving $4\,W\!\odot q$. One checks
$\sum W_{\mathrm{new}} = 1$ is preserved. The covariance tensor is not touched by this step.

**Step 5: pruning.** Any branch with $W_{\mathrm{new}}^{\alpha\alpha} = 0$ contributes nothing to
any observable and is deleted, together with its covariance slot and its row and column of
$W_{\mathrm{new}}$. For instance, swapping in an ancilla currently in $|0\rangle$ gives
$q_\alpha = 0$, so the corresponding type-1 branch is born dead. Pruning is what keeps
$\chi < 2^m$ in practice; the worst case $2^m$ is reached only when every swapped pair is fully
entangled.

**Step 6: fSWAP congruence.** Finally apply the matchgate $\mathrm{fSWAP}_{jk}$ to every surviving
branch covariance by the SPTM congruence of §1; $W$ is unchanged by a matchgate.

After the six steps the data has the same form as before: a covariance tensor
$\Lambda \in \mathbb{R}^{\chi'\times 2N\times 2N}$ and a Hermitian weight matrix
$W \in \mathbb{C}^{\chi'\times\chi'}$, with $\chi' \le 2\chi$ surviving branches. The next circuit
layer, matchgate or SWAP, is applied identically.

**Running example: building $(\Lambda, W)$ step by step.** We now construct the data of §3 from the
input by applying the six steps in order.

*Start.* The input $|011\rangle$ is a single branch, $\chi=1$, with covariance the basis value
$\Lambda_{|011\rangle}$ (on-site blocks $-1,+1,+1$ for qubits $0,1,2$) and weight $W=(1)$.

*Matchgate $V_1$.* The hopping $V_1$ is a matchgate, so it acts by the congruence of §1 on the single
branch and leaves $W$ unchanged. The result is the $\Lambda_{\mathrm{in}}$ of §1, still with $\chi=1$
and $W=(1)$.

*Step 1 (branch doubling).* For $\mathrm{SWAP}_{12}$ we have $j=1$, $k=2$. The single branch splits
into a type-0 branch, which keeps the covariance $\Lambda_{\mathrm{in}}$, and a type-1 branch, whose
covariance is conditioned next.

*Step 2 (conditioning).* Here $S_4=\{2,3,4,5\}$ and $R=\{0,1\}$ (qubit $0$). The type-1 branch pins
qubits $1$ and $2$ to occupied. Splitting $\Lambda_{\mathrm{in}}$ on $(S_4, R)$,

$$
A = \begin{pmatrix}0&-\tfrac12&0&0\\ \tfrac12&0&0&0\\ 0&0&0&1\\ 0&0&-1&0\end{pmatrix},
\qquad
B = \begin{pmatrix}\tfrac{\sqrt3}{2}&0\\ 0&\tfrac{\sqrt3}{2}\\ 0&0\\ 0&0\end{pmatrix},
\qquad
C = \begin{pmatrix}0&\tfrac12\\ -\tfrac12&0\end{pmatrix},
$$

where the rest block $R$ is qubit $0$, entangled with qubit $1$ through $B$. The Schur term
$B^\top(A+\Lambda_{\mathrm{occ}})^{-1}B = \bigl(\begin{smallmatrix}0&-\tfrac32\\ \tfrac32&0\end{smallmatrix}\bigr)$
corrects $C$ to $\Lambda'|_R = \bigl(\begin{smallmatrix}0&-1\\ 1&0\end{smallmatrix}\bigr)$, i.e.
$(\Lambda')_{0,1}=-1$: forcing qubit $1$ back to occupied empties qubit $0$, exactly as
particle-number conservation of $V_1$ requires. The conditioned covariance is therefore the basis
state $|011\rangle$.

*Step 3 (cross occupations).* With $\chi=1$ the cross-occupation matrix is the scalar $q=q_{00}$,
read off $\Gamma_{00}=\Lambda_{\mathrm{in}}$:

$$
q = \tfrac14\Bigl[1 + (\Lambda_{\mathrm{in}})_{2,3} + (\Lambda_{\mathrm{in}})_{4,5}
+ \mathrm{Pf}(\Lambda_{\mathrm{in}}|_{S_4})\Bigr]
= \tfrac14\Bigl[1 + (-\tfrac12) + 1 + (-\tfrac12)\Bigr] = \tfrac14 .
$$

*Step 4 (weight update).* The block rule with $q=\tfrac14$ gives

$$
W_{\mathrm{new}} =
\begin{pmatrix} W & -2\,W q \\ -2\,W q & 4\,W q\end{pmatrix}
= \begin{pmatrix} 1 & -\tfrac12 \\ -\tfrac12 & 1\end{pmatrix} .
$$

*Step 5 (pruning).* Both diagonal entries are nonzero, so no branch is pruned and $\chi=2$.

*Step 6 (fSWAP congruence).* Finally apply the matchgate $\mathrm{fSWAP}_{12}$ to both branch
covariances by congruence, with $W$ unchanged. It sends the type-0 covariance
$\Lambda_{\mathrm{in}}$ to $\Lambda_0$ and the conditioned covariance to $\Lambda_1$ (the basis
state $|011\rangle$ is invariant under $\mathrm{fSWAP}_{12}$). The result is precisely the
$(\Lambda, W)$ used in §3 through §6.

## 8. Cost

Three evaluation paths coexist, and the dispatch of §11 selects among them at runtime; this
section collects their costs so the trade-offs are visible in one place. Throughout, $m$ counts
$\mathrm{CZ}$ branching events, so by the decomposition of §2 a distance-$d$ SWAP contributes
$2d - 1$ to $m$ rather than $1$.

**Overlap-normalized branch pairs (§4–§7).** Each observable costs $\chi^2 \le 4^m$ branch pairs,
with one $O(N^3)$ matrix inverse (for $\Gamma_{\alpha\beta}$) and one Pfaffian per pair; memory is
$O(\chi^2 + \chi N^2)$. Using $W = W^\dagger$ and $\Gamma_{\beta\alpha} = \Gamma_{\alpha\beta}^{*}$
only the upper triangle of pairs is needed. This is the default path; it runs whenever no pair is
masked.

**Full CZ-expansion engine (§11).** An expectation value costs $16^m$ string pairs per Pauli term,
a factor $4^m$ more than the branch pairs above; each string pair is one small Pfaffian of a
sandwich of at most $8m + |S_P|$ linear forms (up to four Majoranas per $\mathrm{CZ}$ event on
each side, plus the support of the word). A $k$-qubit marginal costs $2^k$ such term evaluations,
and a full-outcome probability costs $4^m$ strings per outcome through the amplitude relay. This
path runs when most pairs are masked.

**Hybrid per-pair evaluation (§11).** Healthy pairs keep the $\chi^2$ branch-pair cost. A masked
pair $(\alpha,\beta)$ instead costs $4^{s_\alpha + s_\beta}$ string pairs, where $s_\alpha$ is the
number of type-1 (projected) events in branch $\alpha$'s history, because the unnormalized branch
expands into $4^{s_\alpha}$ projector monomial strings (§11). The masked total is
$\sum_{(\alpha,\beta)\,\mathrm{masked}} 4^{s_\alpha + s_\beta}$. If every pair is masked this
becomes $\bigl(\sum_\alpha 4^{s_\alpha}\bigr)^2$, and summing $4^{s}$ over all $2^m$
type-0/type-1 histories gives $(1+4)^m$, so the worst case reaches $25^m$ and overtakes the
engine's $16^m$. The implementation therefore switches to the full engine exactly when the
masked-pair sum exceeds $16^m$. In the typical wire-sharing case only a few pairs are masked, and
the hybrid is several times cheaper than the full engine (about $3\times$ on a three-SWAP
benchmark).

Every path is polynomial in $N$ and exponential only in $m$. The exponential-in-$m$ floor is
unavoidable: matchgates plus SWAP are universal [2], so any exact classical method must pay
exponentially in some resource as $m\to N$.

## 9. Arbitrary product-state inputs

Everything so far assumed a computational-basis input. A generic product state
$|\psi_{\mathrm{in}}\rangle = \bigotimes_k(a_k|0\rangle + b_k|1\rangle)$, the generalized-input setting
of [5], is still fermionic Gaussian, but *displaced*: its first Majorana moments

$$
d_\mu = \langle\psi_{\mathrm{in}}|\,c_\mu\,|\psi_{\mathrm{in}}\rangle
$$

are nonzero (single-qubit superpositions break parity), and a real antisymmetric covariance has
no slot to store them. The diagnosis is the same as for the single-state formalism: on
$|+\rangle$, $\langle c_0\rangle = \langle X\rangle = 1$ while $\Lambda = 0$. The remedy is also
the same in spirit, namely to enlarge the algebra by one mode, but here we use an *even* extension so the
covariance remains orthogonal and the entire machinery of §3–§7 applies verbatim.

### 9.1 The even parity-purified lift

For any product state one has $\mathrm{rank}(I + \Lambda^2) = 2$ regardless of $N$, so a single
ancilla mode suffices: $d$ lies in the rank-2 range of $I+\Lambda^2$. Lift each branch to a
$(2N+2)$-dimensional Majorana space with

$$
\boxed{\;
M =
\begin{pmatrix}
\Lambda & b_0 & -d \\
-b_0^\top & 0 & -m \\
d^\top & m & 0
\end{pmatrix},
\qquad
m^2 = -\frac{d^\top \Lambda^2 d}{d^\top d}, \quad b_0 = \frac{\Lambda d}{m}.
\;}
$$

The lift is real orthogonal, $M^2 = -I$, so it is a genuine pure-state covariance in the enlarged
algebra. Its physical block is exactly $\Lambda$, and the last column (the **marker**, index
$2N+1$) holds $-d$; the ancilla mode at index $2N$ (the $b_0$ mode) exists only so that $M$ closes
to an orthogonal matrix and never appears in any observable support. When the displacement
vanishes the ancilla decouples, $M = \Lambda \oplus \begin{pmatrix}0&1\\-1&0\end{pmatrix}$, so the
same construction covers basis-state and product-state inputs with one code path. (As $|d|\to 0$
the formula $b_0 = \Lambda d/m$ divides by a vanishing $m$; one simply snaps to the
zero-displacement case below a small threshold.)

### 9.2 Propagating the lift, transition covariance, and a uniform Wick rule

The ancilla frame is a gauge: cross-branch matrix elements are correct only if *every* branch
shares one frame. The lift is therefore built **once** on the initial state and then propagated,
never rebuilt per branch from its own $(\Lambda, d)$. Under a matchgate the SPTM acts as
$Q \oplus I_2$ (the two ancilla modes ride along trivially), and the occupation projection of §7
conditions on the physical modes $\{2j,2j+1,2k,2k+1\}$ only, leaving the ancilla rows and columns
in the complement.

Because the lifted covariance is orthogonal and even-dimensional, the transition covariance of §4
is well-defined; in the factored form,

$$
\Gamma = i\,\bigl(2\,\bar P_\alpha(\bar P_\alpha + P_\beta)^{-1}\bar P_\beta\bigr),
\qquad P = \tfrac{I + iM}{2},\ \bar P = \tfrac{I - iM}{2},
$$

followed by rescaling the marker row and column (index $2N+1$) by $-i$. The leading $i$ puts the
result in the same convention as §4 ($\Gamma(M,M)=M$ off-diagonal), and the $-i$ marker rescaling
encodes $i\,d$ in the marker column, which is what makes the two parity sectors share a single
phase convention. A single rule then covers monomials of either parity: for support $S$,

$$
\boxed{\;
\frac{\langle\phi_\alpha|\,\textstyle\prod_{\mu\in S} c_\mu\,|\phi_\beta\rangle}
{\langle\phi_\alpha|\phi_\beta\rangle}
= i^{-\lceil |S|/2\rceil}\,\mathrm{Pf}\!\left(\Gamma|_{S'}\right),
\qquad
S' =
\begin{cases}
S & |S|\ \text{even},\\
S \cup \{2N+1\} & |S|\ \text{odd}.
\end{cases}
\;}
$$

Outcome probabilities use the parity-even projector and the physical block of $\Gamma$ (the marker
is never appended); Hamiltonian terms append the marker for odd-weight Pauli strings. The
expressions of §5 and §6 are otherwise unchanged, now evaluated in $2N+2$ dimensions. The lift
adds two Majorana modes in total, not per branch and not per SWAP, so the cost stays
$O(\chi^2 N^3)$ in time and $O(\chi N^2)$ in memory: displaced inputs are on the same complexity
footing as basis-state inputs.

## 10. Validity regime of the overlap-normalized evaluation

The overlap-normalized evaluation of §4–§7 is exact whenever the branches remain linearly
non-degenerate. This holds in particular for circuits with no SWAP (where it reduces to the
ordinary single-state matchgate simulation), for a single SWAP, for SWAPs acting on disjoint
wire pairs, and for generic inputs. These are precisely the cases in which distinct branches keep
nonzero mutual overlaps.

It becomes ill-defined when two branches turn **exactly orthogonal**,
$\langle\phi_\alpha|\phi_\beta\rangle = 0$, while a measurement still **connects** them,
$\langle\phi_\alpha|P_y|\phi_\beta\rangle \neq 0$. The probability contribution of such a pair is

$$
\langle\phi_\alpha|P_y|\phi_\beta\rangle
= W_{\alpha\beta}\,\Bigl[2^{-N}\mathrm{Pf}(\Lambda_y)\,
\mathrm{Pf}(\Gamma_{\alpha\beta} + \Lambda_y)\Bigr],
$$

and at orthogonality $W_{\alpha\beta} = 0$ multiplies a Pfaffian that diverges, because
$\Gamma_{\alpha\beta}$ carries $(\Lambda_\alpha + \Lambda_\beta)^{-1}$ which is singular there. The
finite true value is the indeterminate product $0\times\infty$ and is lost. This is intrinsic to
the (covariance, $W$) parametrization: orthogonality erases precisely the inter-branch relative
phase that $W$ is meant to store, so the representation can no longer reconstruct the matrix
element. Such exactly-degenerate pairs arise generically when two SWAPs share a wire.

The structural cure is to evaluate observables without dividing by the overlap. Because a common
matchgate preserves all pairwise overlaps and all branch coefficients, the relevant degrees of
freedom are the individual per-branch amplitudes, and an outcome probability is the manifestly finite

$$
p(y) = \Bigl|\sum_\alpha \lambda_\alpha\,\langle y|\phi_\alpha\rangle\Bigr|^2,
$$

which never forms a pairwise $0\times\infty$. The difficulty here is not the evaluation of any single
amplitude. Its magnitude is fixed by the covariance, $|\langle y|\phi_\alpha\rangle|^2 =
2^{-N}|\mathrm{Pf}(\Lambda_\alpha + \Lambda_y)|$, and a complex signed Pfaffian is available to
evaluate the amplitude itself once a branch is represented by its Bogoliubov data, so the signed
square root is not the obstruction. The genuine obstruction is the *relative phase between
branches*. A covariance fixes its Gaussian state only up to a global phase, so the covariances alone
do not say how the per-branch amplitudes should be phased relative to one another, which is exactly
the information the coherent sum requires, and exactly the information that the weight matrix $W$
supplies through the pairwise overlaps until those overlaps vanish. The cure is therefore to stop
representing branches as states at all, and instead to keep the non-Gaussian $\mathrm{CZ}$ factors as
*operators* whose phases stay explicit throughout the evaluation. Section 11 develops this
Heisenberg-picture construction and shows that it remains exact precisely where the
overlap-normalized representation of §4–§7 breaks down.

## 11. Exactly orthogonal branches: the Heisenberg CZ-expansion evaluation

Carrying the non-Gaussian factors as operators, as anticipated at the close of §10, means expanding
the circuit in the Heisenberg picture rather than branching the state. Every branch is generated
from one Gaussian input by an alternating string of matchgate layers and $\mathrm{CZ}$ factors, and
only the $\mathrm{CZ}$s are non-Gaussian, so it suffices to expand each $\mathrm{CZ}$ as an operator
identity in Majorana monomials,

$$
\boxed{\;
\mathrm{CZ}_{jk} = \tfrac12\bigl(1 + Z_j + Z_k - Z_j Z_k\bigr),
\qquad Z_q = -i\,c_{2q}c_{2q+1},
\;}
$$

a sum of four monomials of $0$, $2$, or $4$ Majorana factors with explicit coefficients
$\{\tfrac12, -\tfrac i2, -\tfrac i2, \tfrac12\}$. A matchgate conjugates a Majorana into a *linear
form* in Majoranas, $S\,c_\mu\,S^\dagger = \sum_\nu (Q_S)_{\mu\nu}\,c_\nu$ with $Q_S$ the SPTM of
$S$, so each inserted factor can be pushed through every matchgate layer that follows it: this is
the suffix-SPTM push-through that carries a $\mathrm{CZ}$ insertion forward as a rotated linear form
rather than as a branch split. Writing $Q_t$ for the SPTM of the layers *before* the $t$-th
$\mathrm{CZ}$ and $Q_{\mathrm{tot}}$ for the whole matchgate stream (the circuit with all
$\mathrm{CZ}$s deleted), the suffix rotation of event $t$ is $Q_t^\top Q_{\mathrm{tot}}$, and the
state becomes a sum of at most $4^m$ *strings*,

$$
\boxed{\;
|\psi\rangle = \sum_{\sigma} w_\sigma\, \ell^\sigma_1 \cdots \ell^\sigma_{r_\sigma}\, |G\rangle,
\qquad
|G\rangle = U_{\mathrm{tot}}\,|\Omega\rangle,
\qquad
\ell = \sum_\nu v_\nu\, c_\nu,
\;}
$$

where $|G\rangle$ is the single Gaussian output of the $\mathrm{CZ}$-free circuit, every $\ell$ is a
final-frame linear form (a row of a suffix SPTM), and every coefficient $w_\sigma$ is explicit. No
branch state is ever normalized here, so no relative-phase gauge is ever needed: the $0\times\infty$
diagnosed in §10 cannot form.

**Wick contraction rule.** All observables reduce to moments of ordered products of these linear
forms in a Gaussian (quasi-free) functional, which factorize through pairwise contractions,

$$
\langle \ell_1 \cdots \ell_{2t} \rangle = \mathrm{Pf}(A),
\qquad
A_{ab} = v_a^\top\,\bigl(I - i\,\Gamma\bigr)\,v_b \quad (a<b),
$$

where $\Gamma$ is the relevant (transition) covariance and the identity term carries the
anticommutator, so repeated and overlapping modes are handled automatically. Three observable
classes follow from this single rule, each with its own cost.

**Hamiltonian terms.** For a Pauli word $P = \kappa_P\, c_{S_P}$, the expectation value
$\langle\psi|P|\psi\rangle = \sum_{\sigma\tau} w_\sigma^* w_\tau\,\kappa_P\,
\langle G|\,\ell^{\sigma\dagger}\cdots c_{S_P} \cdots \ell^\tau\,|G\rangle$ is evaluated with
$\Gamma = \Lambda_G$ (marker-rescaled on the lift, with the marker appended to odd-weight supports
exactly as in §9.2), at a cost of $16^m$ string pairs per term, each a small Pfaffian.

**Full-outcome probabilities.** Rather than evaluate each amplitude $\langle y|G\rangle$ on its own,
relay it through one fixed reference: pick once a lifted basis state $y'$ greedily, mode by mode, so
that $p_G(y') \ge 2^{-(N+1)}$, which keeps $\Gamma_{y'G}$ finite and well-conditioned. Writing the
explicit flip monomial $\langle y| = \langle y'|\prod_{q\in F} X_q$, with $X_q = Z_0\cdots
Z_{q-1}\,c_{2q}$, then gives

$$
p(y) = p_G(y')\sum_{a}\Bigl|\sum_\sigma w_\sigma\,
\frac{\langle y'|\,X_F\,\ell^\sigma_1\cdots\ell^\sigma_{r_\sigma}\,|G\rangle}
{\langle y'|G\rangle}\Bigr|^2 ,
$$

where $a$ sums over the two ancilla values of the lift (only the parity-allowed one survives) and
the unknown phase of $\langle y'|G\rangle$ cancels in the modulus, at a cost of $4^m$ strings per
outcome.

**Marginals.** The projector on $k$ measured qubits expands into $2^k$ Pauli-$Z$ words,
$P_y = \prod_i \tfrac12\bigl(1 + (1-2y_i)\,Z_{q_i}\bigr)$, each an even monomial evaluated exactly
like a Hamiltonian term.

**Dispatch.** Pairwise branch overlaps satisfy

$$
|\langle\phi_\alpha|\phi_\beta\rangle|^2
= 2^{-D/2}\,|\mathrm{Pf}(\Lambda_\alpha + \Lambda_\beta)|,
$$

the absolute value of the overlap-magnitude identity of §4, with $D$ the Majorana dimension and
$|\mathrm{Pf}(\Lambda_\alpha)| = 1$. By the same matchgate-invariance argument used in §10, common
matchgate layers do not change these overlaps, so checking every pair once after each branching
event covers the whole circuit. When every pair stays above a small threshold, the
overlap-normalized evaluation of §4–§7 runs unchanged; it is the cheapest path by a factor $4^m$
on expectation values (§8).

When a pair falls below the threshold, the failure is pairwise and so is the cure. A pair is
*masked* when its overlap is (near-)zero, or when its weight entry was propagated through such a
pair at an earlier branching. The second clause is a tainted-lineage rule, and it is needed
because the weight update of §7 (Step 4) writes the parents' cross occupation $q_{\alpha\beta}$
into the children's weight entries: $q_{\alpha\beta}$ is computed through $\Gamma_{\alpha\beta}$,
so if the parents were (near-)orthogonal it was already unreliable, and the children's entry stays
unreliable even if the children themselves are no longer orthogonal.

A masked pair is evaluated overlap-free, by the expansion of this section specialized to one pair.
Every branch is, by construction, the explicit operator string of its type-0/type-1 history,

$$
\lambda_\alpha|\phi_\alpha\rangle
= \prod_t \bigl[\,1 \ \text{or}\ -2\,n_{j_t} n_{k_t}\,\bigr]\,|G\rangle,
$$

and each type-1 factor expands into four monomials,

$$
-2\,n_j n_k = \tfrac12\bigl(-1 + Z_j + Z_k - Z_j Z_k\bigr),
$$

so a branch with $s_\alpha$ type-1 events becomes a sum of $4^{s_\alpha}$ strings, and the masked
pair becomes a string-pair Wick sum against $\Lambda_G$, exactly as for the Hamiltonian terms
above. No overlap is divided out, so the $0\times\infty$ of §10 cannot form.

The healthy pairs keep the overlap-normalized sum, with the masked entries of $W$ zeroed and the
masked transition covariances replaced by a benign placeholder, so that neither the value nor the
gradient ever touches the $0\times\infty$. When most pairs are masked, the per-pair route loses
its advantage; the implementation counts the masked string pairs and, once that count exceeds the
cost of the full expansion above, switches to the full engine (§8 quantifies the crossover).
Full-outcome probabilities need no dispatch at all: they always use the per-branch amplitude relay
above, which never forms a pair overlap in the first place. Every path is exact for arbitrary
$\mathrm{SWAP}$ placement, including the wire-sharing case that §10 excluded.

## References

[1] S. Bravyi. "Lagrangian representation for fermionic linear optics." *Quantum Information and
Computation* 5, 216 (2005). arXiv:quant-ph/0404180.

[2] R. Jozsa and A. Miyake. "Matchgates and classical simulation of quantum circuits."
*Proceedings of the Royal Society A* 464, 3089 (2008). arXiv:0804.4050.

[3] B. M. Terhal and D. P. DiVincenzo. "Classical simulation of noninteracting-fermion quantum
circuits." *Physical Review A* 65, 032325 (2002). arXiv:quant-ph/0108010.

[4] E. Knill. "Fermionic linear optics and matchgates." (2001). arXiv:quant-ph/0108033.

[5] D. J. Brod. "Efficient classical simulation of matchgate circuits with generalized inputs and
measurements." *Physical Review A* 93, 062332 (2016). arXiv:1602.03539.

[6] M. Dias and R. König. "Classical simulation of non-Gaussian fermionic circuits." *Quantum* 8,
1350 (2024). arXiv:2307.12912.

[7] J. Cudby and S. Strelchuk. "Gaussian decomposition of magic states for matchgate
computations." (2023). arXiv:2307.12654.
