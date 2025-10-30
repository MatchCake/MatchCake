# Naming conventions

This file contains the naming conventions used in the project.

## Matchgates

A matchgate is a matrix of the form

$$
M =\begin{pmatrix}
    a & 0 & 0 & b \\
    0 & w & x & 0 \\
    0 & y & z & 0 \\
    c & 0 & 0 & d
\end{pmatrix}
$$
where $a, b, c, d, w, x, y, z \in \mathbb{C}$. The matrix M can be decomposed as

$$
A = \begin{pmatrix}
    a & b \\
    c & d
\end{pmatrix}
$$
and
$$
W = \begin{pmatrix}
    w & x \\
    y & z
\end{pmatrix}
$$
So, the matchgate is often named as $M(A, W)$ or what we called a fermionic composition of the gates $A$ and $W$. From that point of view, we call `Comp{A}{W}` the operations that come from the fermionic composition of $A$ and $W$. Else, we name `{A}{W}` operations that come from the tensor product of $A$ and $W$ ($A \otimes W$) like other quantum computation packages like PennyLane usually do.


## Single Particle Transition Matrices (SPTMs)

The single particle transition matrices are a special form of matchgates
$$
\begin{align}
    R_{\mu\nu} = \frac{1}{4} \text{Tr}{\left(M c_\mu M^\dagger\right)c_\nu},
\end{align}
$$
which is related to the free-fermionic hamiltonian of the particule. Then we call those operations as `Sptm{M}` where `M` is the matchgate nomenclature explain in the last section. 


## Fermionic Operators

We call `fermionic_{*}` operations that are special fermionic operations that are usually made of multiple matchgates.



# Others
- https://pennylane.ai/codebook/circuits-with-many-qubits/multi-qubit-systems

