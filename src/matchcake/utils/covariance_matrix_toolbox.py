import itertools
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch

from ..typing import TensorLike
from ._pfaffian import sector_pfaffian_features
from .block_diagonalize_covariance import BlockDiagonalizeCovariance
from .majorana import MajoranaGetter
from .math import convert_like_and_cast_to
from .torch_utils import infer_complex_dtype, infer_real_dtype, torch_dtype_name


class CovarianceMatrixToolbox:
    r"""
    Static toolbox for the Majorana covariance matrices of fermionic Gaussian states.

    The covariance matrix of a state :math:`\rho` on :math:`m` modes is the real antisymmetric
    :math:`2m \times 2m` matrix

    .. math::
        \Lambda_{\mu\nu} = i\, \mathrm{Tr}(\rho\, c_\mu c_\nu), \quad \mu \neq \nu,

    with the Jordan-Wigner Majorana operators :math:`c_{2k} = Z_{<k} X_k` and
    :math:`c_{2k+1} = Z_{<k} Y_k`. It describes a fermionic Gaussian state without displacement
    completely. The state is pure when :math:`\Lambda^2 = -\mathbb{1}` and mixed otherwise.

    Every method takes tensors with arbitrary leading batch dimensions and returns its result in the
    backend of its input.
    """

    @staticmethod
    def displacement_vector(
        product_state_amplitudes: TensorLike,
        wires: Sequence[int],
    ) -> torch.Tensor:
        r"""
        Displacement vector :math:`d_\mu = \langle c_\mu \rangle` of a product state.

        The product state is :math:`\bigotimes_k (a_k |0\rangle + b_k |1\rangle)`, with
        ``product_state_amplitudes[..., k, :]`` holding :math:`(a_k, b_k)`. With the Jordan-Wigner
        convention,

        .. math::
            \langle c_{2k} \rangle = \langle X_k \rangle \prod_{l<k} \langle Z_l \rangle,
            \qquad
            \langle c_{2k+1} \rangle = \langle Y_k \rangle \prod_{l<k} \langle Z_l \rangle.

        :param product_state_amplitudes: Complex tensor of shape ``(..., n, 2)``.
        :type product_state_amplitudes: TensorLike
        :param wires: Wire labels (unused for the computation; here for API consistency).
        :type wires: Sequence[int]
        :return: Real tensor of shape ``(..., 2n)``.
        :rtype: torch.Tensor
        """
        # Preserve the precision of the input amplitudes (complex64 -> float32) instead
        # of forcing complex128/float64.
        c_dtype = infer_complex_dtype(product_state_amplitudes)
        r_dtype = infer_real_dtype(product_state_amplitudes)
        psi = torch.as_tensor(
            np.array(product_state_amplitudes)
            if not isinstance(product_state_amplitudes, torch.Tensor)
            else product_state_amplitudes,
            dtype=c_dtype,
        )
        alpha = psi[..., 0]  # (..., n)
        beta = psi[..., 1]  # (..., n)

        x = 2.0 * torch.real(torch.conj(alpha) * beta)
        y = 2.0 * torch.imag(torch.conj(alpha) * beta)
        z = torch.abs(alpha) ** 2 - torch.abs(beta) ** 2  # (..., n), real

        n = psi.shape[-2]
        batch_shape = psi.shape[:-2]

        # Exclusive prefix product: z_prod[k] = z[0] * ... * z[k-1], z_prod[0] = 1.
        if n > 1:
            z_prod = torch.cat(
                [
                    torch.ones(batch_shape + (1,), dtype=r_dtype, device=psi.device),
                    torch.cumprod(z[..., :-1], dim=-1),
                ],
                dim=-1,
            )  # (..., n)
        else:
            z_prod = torch.ones(batch_shape + (n,), dtype=r_dtype, device=psi.device)

        # Interleave without in-place writes so that gradients flow through the amplitudes.
        return torch.stack([z_prod * x, z_prod * y], dim=-1).reshape(batch_shape + (2 * n,))

    @staticmethod
    def majorana_indices(wire_indices: Union[Sequence[int], np.ndarray]) -> np.ndarray:
        r"""
        Return the Majorana indices :math:`(2w, 2w+1)` of the given wire positions, in the given order.

        :param wire_indices: Positions of the wires in the register, shape ``(k,)``.
        :type wire_indices: Union[Sequence[int], np.ndarray]
        :return: Integer array of shape ``(2k,)``.
        :rtype: np.ndarray
        """
        positions = np.asarray(wire_indices, dtype=int).reshape(-1)
        return np.stack([2 * positions, 2 * positions + 1], axis=1).reshape(-1)

    @staticmethod
    def are_wires_contiguous(wire_indices: Union[Sequence[int], np.ndarray]) -> bool:
        r"""
        Tell whether wire positions form a contiguous block listed in increasing order.

        For such a block, the principal submatrix of a covariance matrix on the Majorana indices of the block
        is the covariance matrix of the qubit reduced state. Otherwise the Jordan-Wigner strings running
        through the removed wires make the fermionic and the qubit reduced states differ.

        :param wire_indices: Positions of the wires in the register.
        :type wire_indices: Union[Sequence[int], np.ndarray]
        :return: ``True`` for positions of the form ``[p, p + 1, ..., p + k - 1]``.
        :rtype: bool
        """
        positions = np.asarray(wire_indices, dtype=int).reshape(-1)
        if positions.size == 0:
            return False
        return bool(np.array_equal(positions, np.arange(positions[0], positions[0] + positions.size)))

    @staticmethod
    def validate_reduced_wire_indices(wire_indices: Union[Sequence[int], np.ndarray], fermionic: bool = False) -> None:
        r"""
        Check that the reduced state on the given wire positions is the one the caller expects.

        :param wire_indices: Positions of the kept wires in the register.
        :type wire_indices: Union[Sequence[int], np.ndarray]
        :param fermionic: Whether the caller asked for the fermionic reduced state, which is defined for any
            set of distinct positions.
        :type fermionic: bool
        :raises ValueError: If no wire is kept, if a wire is repeated, or if the positions are not a contiguous
            block in increasing order while ``fermionic`` is ``False``.
        """
        positions = np.asarray(wire_indices, dtype=int).reshape(-1)
        if positions.size == 0:
            raise ValueError("At least one wire must be kept.")
        if len(set(positions.tolist())) != positions.size:
            raise ValueError(f"The kept wires must be distinct, got the positions {positions.tolist()}.")
        if not fermionic and not CovarianceMatrixToolbox.are_wires_contiguous(positions):
            raise ValueError(
                f"The kept wires (register positions {positions.tolist()}) are not a contiguous block listed in "
                f"register order. The Jordan-Wigner strings of their Majorana operators then run through the "
                f"removed wires, so the covariance matrix restricted to them describes the reduced state of "
                f"the fermionic modes, which differs from the reduced state of the qubits and from what a "
                f"qubit simulator returns. Pass fermionic=True to get the fermionic reduced state."
            )

    @staticmethod
    def reduce(covariance_matrix: TensorLike, wire_indices: Union[Sequence[int], np.ndarray]) -> TensorLike:
        r"""
        Return the covariance matrix of the fermionic reduced state on the given wire positions.

        The reduced state of a fermionic Gaussian state on a subset of its modes is Gaussian, and its
        covariance matrix is the principal submatrix on the Majorana indices of the kept modes. It equals
        the qubit reduced state when the kept positions are contiguous, see :meth:`are_wires_contiguous`.

        :param covariance_matrix: Covariance matrix of shape ``(..., 2n, 2n)``.
        :type covariance_matrix: TensorLike
        :param wire_indices: Positions of the kept wires, shape ``(k,)``.
        :type wire_indices: Union[Sequence[int], np.ndarray]
        :return: Principal submatrix of shape ``(..., 2k, 2k)``, in the backend and dtype of the input.
        :rtype: TensorLike
        """
        indices = CovarianceMatrixToolbox.majorana_indices(wire_indices)
        return covariance_matrix[..., indices[:, None], indices[None, :]]  # type: ignore[index]

    @staticmethod
    def mode_polarizations(covariance_matrix: TensorLike) -> TensorLike:
        r"""
        Return the magnitudes :math:`|\lambda_j|` of the normal-mode polarizations of a covariance matrix.

        They are the non-negative eigenvalues of the Hermitian matrix :math:`i\Lambda`, in ascending order.
        A mode with :math:`|\lambda_j| = 1` is pure and a mode with :math:`\lambda_j = 0` is maximally mixed.
        Only eigenvalues are used, so the result is differentiable with the native autograd of the backend.

        :param covariance_matrix: Real antisymmetric matrix of shape ``(..., 2m, 2m)``.
        :type covariance_matrix: TensorLike
        :return: Polarization magnitudes of shape ``(..., m)``, in the backend of the input and in the real
            floating dtype of its precision.
        :rtype: TensorLike
        """
        covariance_t = CovarianceMatrixToolbox._as_real_torch(covariance_matrix)
        covariance_t = 0.5 * (covariance_t - covariance_t.mT)
        n_modes = covariance_t.shape[-1] // 2
        hermitian = 1j * covariance_t.to(infer_complex_dtype(covariance_t))
        eigenvalues = torch.linalg.eigvalsh(hermitian)  # ascending: (-lambda_j, ..., +lambda_j)
        return CovarianceMatrixToolbox._convert_real_like(eigenvalues[..., n_modes:], covariance_matrix, covariance_t)

    @staticmethod
    def von_neumann_entropy(covariance_matrix: TensorLike, base: Optional[float] = None) -> TensorLike:
        r"""
        Von Neumann entropy of the fermionic Gaussian state with the given covariance matrix.

        With :math:`p_j = (1 + |\lambda_j|) / 2` the occupation probabilities of the normal modes,

        .. math::
            S = -\sum_j \left[ p_j \ln p_j + (1 - p_j) \ln (1 - p_j) \right].

        The terms with a vanishing probability are dropped, so the value and the gradient stay finite for
        pure modes.

        :param covariance_matrix: Real antisymmetric matrix of shape ``(..., 2m, 2m)``.
        :type covariance_matrix: TensorLike
        :param base: Base of the logarithm. Defaults to the natural logarithm.
        :type base: Optional[float]
        :return: Entropy of shape ``(...)``, in the backend of the input and in the real floating dtype of
            its precision.
        :rtype: TensorLike
        """
        covariance_t = CovarianceMatrixToolbox._as_real_torch(covariance_matrix)
        polarizations = CovarianceMatrixToolbox._as_real_torch(CovarianceMatrixToolbox.mode_polarizations(covariance_t))
        probabilities = 0.5 * (1.0 + polarizations)
        entropy = -torch.sum(
            CovarianceMatrixToolbox._entropy_term(probabilities)
            + CovarianceMatrixToolbox._entropy_term(1.0 - probabilities),
            dim=-1,
        )
        if base is not None:
            entropy = entropy / np.log(base)
        return CovarianceMatrixToolbox._convert_real_like(entropy, covariance_matrix, covariance_t)

    @staticmethod
    def purity(covariance_matrix: TensorLike) -> TensorLike:
        r"""
        Purity :math:`\mathrm{Tr}\,\rho^2` of the fermionic Gaussian state with the given covariance matrix.

        In terms of the normal-mode polarizations, the purity is :math:`\prod_j (1 + \lambda_j^2) / 2`. It is
        computed without any eigendecomposition as :math:`2^{-m} \sqrt{\det(\mathbb{1} - \Lambda^2)}`, which is
        smooth everywhere.

        :param covariance_matrix: Real antisymmetric matrix of shape ``(..., 2m, 2m)``.
        :type covariance_matrix: TensorLike
        :return: Purity of shape ``(...)``, in the backend of the input and in the real floating dtype of
            its precision.
        :rtype: TensorLike
        """
        covariance_t = CovarianceMatrixToolbox._as_real_torch(covariance_matrix)
        covariance_t = 0.5 * (covariance_t - covariance_t.mT)
        size = covariance_t.shape[-1]
        identity = torch.eye(size, dtype=covariance_t.dtype, device=covariance_t.device)
        determinant = torch.linalg.det(identity - covariance_t @ covariance_t)
        result = torch.sqrt(determinant) / 2 ** (size // 2)
        return CovarianceMatrixToolbox._convert_real_like(result, covariance_matrix, covariance_t)

    @staticmethod
    def to_density_matrix(covariance_matrix: TensorLike) -> TensorLike:
        r"""
        Dense density matrix of the fermionic Gaussian state with the given covariance matrix.

        The state has no displacement, so only even Majorana products contribute to the Wick expansion

        .. math::
            \rho = \frac{1}{2^m} \sum_{|S| \,\mathrm{even}} i^{|S|/2}\, \mathrm{Pf}(\Lambda|_S)\, c_S,

        where :math:`c_S` is the ordered product of the Majorana operators of the index set :math:`S`. The
        sum has :math:`2^{2m-1}` terms of size :math:`2^m \times 2^m`: this is meant for small systems.

        :param covariance_matrix: Real antisymmetric matrix of shape ``(..., 2m, 2m)``.
        :type covariance_matrix: TensorLike
        :return: Density matrix of shape ``(..., 2^m, 2^m)``, in the backend of the input and in the complex
            dtype of its precision.
        :rtype: TensorLike
        """
        covariance_t = CovarianceMatrixToolbox._as_real_torch(covariance_matrix)
        n_modes = covariance_t.shape[-1] // 2
        dimension = 2**n_modes
        complex_dtype = infer_complex_dtype(covariance_t)
        majoranas = np.asarray(MajoranaGetter(n_modes).majorana_tensor, dtype=complex)  # (2m, 2^m, 2^m)
        batch_shape = covariance_t.shape[:-2]
        density = torch.eye(dimension, dtype=complex_dtype, device=covariance_t.device).expand(
            *batch_shape, dimension, dimension
        )
        for subset_size in range(2, 2 * n_modes + 1, 2):
            index_sets = np.array(list(itertools.combinations(range(2 * n_modes), subset_size)), dtype=int)
            operators = np.stack([np.linalg.multi_dot([majoranas[index] for index in subset]) for subset in index_sets])
            operators_t = torch.as_tensor(operators, dtype=complex_dtype, device=covariance_t.device)
            pfaffians = sector_pfaffian_features(covariance_t, index_sets).to(complex_dtype)  # (..., n_terms)
            density = density + (1j ** (subset_size // 2)) * torch.einsum("...s,sij->...ij", pfaffians, operators_t)
        density = density / dimension
        return convert_like_and_cast_to(density, covariance_matrix, dtype=torch_dtype_name(complex_dtype))

    @staticmethod
    def from_density_matrix(density_matrix: TensorLike) -> TensorLike:
        r"""
        Majorana covariance matrix of a dense density matrix.

        The entries are :math:`\Lambda_{\mu\nu} = i\, \mathrm{Tr}(\rho\, c_\mu c_\nu)` for
        :math:`\mu \neq \nu` and zero on the diagonal, with the Jordan-Wigner Majorana operators of the
        register. The covariance matrix describes the state completely only when the state is a fermionic
        Gaussian state without displacement.

        :param density_matrix: Density matrix of shape ``(..., 2^m, 2^m)``.
        :type density_matrix: TensorLike
        :return: Real antisymmetric matrix of shape ``(..., 2m, 2m)``, in the backend of the input and in the
            real floating dtype of its precision.
        :rtype: TensorLike
        """
        is_tensor = isinstance(density_matrix, torch.Tensor)
        density_t = torch.as_tensor(density_matrix if is_tensor else np.asarray(density_matrix))
        complex_dtype = infer_complex_dtype(density_t)
        density_t = density_t.to(complex_dtype)
        n_modes = int(np.round(np.log2(density_t.shape[-1])))
        majoranas = torch.as_tensor(
            np.asarray(MajoranaGetter(n_modes).majorana_tensor, dtype=complex),
            dtype=complex_dtype,
            device=density_t.device,
        )  # (2m, 2^m, 2^m)
        products = torch.einsum("mab,nbc->mnac", majoranas, majoranas)  # (2m, 2m, 2^m, 2^m)
        traces = torch.einsum("...ca,mnac->...mn", density_t, products)
        covariance = torch.real(1j * traces)
        off_diagonal = 1.0 - torch.eye(2 * n_modes, dtype=covariance.dtype, device=covariance.device)
        covariance = covariance * off_diagonal
        return convert_like_and_cast_to(covariance, density_matrix, dtype=torch_dtype_name(covariance.dtype))

    @staticmethod
    def from_polarizations(polarizations: TensorLike) -> TensorLike:
        r"""
        Build the block-diagonal covariance matrix of a product of single-mode states.

        The matrix is :math:`\Lambda_D = \bigoplus_j (-\lambda_j) J` with
        :math:`J = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}`, so that
        :math:`(\Lambda_D)_{2j, 2j+1} = -\lambda_j`. A computational-basis state :math:`|x\rangle` has
        :math:`\lambda_j = (-1)^{x_j}`, and a single-mode mixture :math:`p_j |0\rangle\langle 0| + (1 - p_j)
        |1\rangle\langle 1|` has :math:`\lambda_j = 2 p_j - 1`.

        :param polarizations: Polarizations :math:`\lambda_j = \langle Z_j \rangle` of shape ``(..., m)``.
        :type polarizations: TensorLike
        :return: Real antisymmetric matrix of shape ``(..., 2m, 2m)``, in the backend of the input and in the real
            floating dtype of its precision (``float64`` for an integer input).
        :rtype: TensorLike
        """
        polarizations_t = CovarianceMatrixToolbox._as_real_torch(polarizations)
        zeros = torch.zeros_like(polarizations_t)
        # Each mode contributes the block [[0, -lambda], [lambda, 0]]. Built out-of-place to keep gradients.
        blocks = torch.stack(
            [torch.stack([zeros, -polarizations_t], dim=-1), torch.stack([polarizations_t, zeros], dim=-1)], dim=-2
        )  # (..., m, 2, 2)
        n_modes = polarizations_t.shape[-1]
        mode_identity = torch.eye(n_modes, dtype=polarizations_t.dtype, device=polarizations_t.device)
        covariance = torch.einsum("jk,...jab->...jakb", mode_identity, blocks).reshape(
            *polarizations_t.shape[:-1], 2 * n_modes, 2 * n_modes
        )
        return CovarianceMatrixToolbox._convert_real_like(covariance, polarizations, polarizations_t)

    @staticmethod
    def block_diagonalize(covariance_matrix: TensorLike) -> Tuple[TensorLike, TensorLike]:
        r"""
        Block-diagonalize a Majorana covariance matrix with a single-particle transition matrix.

        Any real antisymmetric matrix :math:`\Lambda` of even size can be written as

        .. math::
            \Lambda = R^\top \Lambda_D R,
            \qquad
            \Lambda_D = \bigoplus_{j=0}^{m-1} (-\lambda_j) \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix},

        with :math:`R \in SO(2m)` a single-particle transition matrix and :math:`\lambda_j` real. When
        :math:`\Lambda` is the covariance matrix of a fermionic Gaussian state, the :math:`\lambda_j` lie in
        :math:`[-1, 1]` and are the polarizations :math:`\langle Z_j \rangle` of the normal modes: the state
        is the product of single-mode mixtures :math:`\tfrac{1 + \lambda_j}{2} |0\rangle\langle 0| +
        \tfrac{1 - \lambda_j}{2} |1\rangle\langle 1|` evolved by the Gaussian unitary whose transition
        matrix is :math:`R`. A mode with :math:`|\lambda_j| = 1` is pure and a mode with :math:`\lambda_j = 0`
        is maximally mixed.

        The decomposition is not unique: rotations inside a mode, and rotations mixing modes with equal or
        opposite polarizations, leave :math:`\Lambda` unchanged. The function is differentiable with respect to
        ``covariance_matrix`` for any downstream quantity that only depends on :math:`\Lambda` through the
        reconstructed state, with the gradient along these gauge directions projected out.

        :param covariance_matrix: Real antisymmetric matrix of shape ``(..., 2m, 2m)``.
        :type covariance_matrix: TensorLike
        :return: The transition matrix ``R`` of shape ``(..., 2m, 2m)`` and the polarizations of shape
            ``(..., m)``, both in the backend of the input and in the real floating dtype of its precision
            (``float64`` for an integer input, the real counterpart for a complex input).
        :rtype: Tuple[TensorLike, TensorLike]
        """
        covariance_t = CovarianceMatrixToolbox._as_real_torch(covariance_matrix)
        sptm, polarizations = BlockDiagonalizeCovariance.apply(covariance_t)
        return (
            CovarianceMatrixToolbox._convert_real_like(sptm, covariance_matrix, covariance_t),
            CovarianceMatrixToolbox._convert_real_like(polarizations, covariance_matrix, covariance_t),
        )

    @staticmethod
    def _as_real_torch(tensor: TensorLike) -> torch.Tensor:
        r"""
        Convert a tensor-like to a real floating ``torch.Tensor`` of the same precision.

        Torch inputs keep their device and their autograd graph. Integer inputs are promoted to ``float64``
        and complex inputs are reduced to their real part.

        :param tensor: The tensor to convert.
        :type tensor: TensorLike
        :return: A real floating tensor.
        :rtype: torch.Tensor
        """
        tensor_t = tensor if isinstance(tensor, torch.Tensor) else torch.as_tensor(np.asarray(tensor))
        if torch.is_complex(tensor_t):
            tensor_t = torch.real(tensor_t)
        return tensor_t.to(infer_real_dtype(tensor_t))

    @staticmethod
    def _convert_real_like(result: torch.Tensor, like: TensorLike, real_reference: torch.Tensor) -> TensorLike:
        r"""
        Convert a real result to the backend of ``like`` with the real dtype of ``real_reference``.

        :param result: The tensor to convert.
        :type result: torch.Tensor
        :param like: The tensor whose backend is used.
        :type like: TensorLike
        :param real_reference: The real tensor whose dtype is used.
        :type real_reference: torch.Tensor
        :return: The converted tensor.
        :rtype: TensorLike
        """
        return convert_like_and_cast_to(result, like, dtype=torch_dtype_name(real_reference.dtype))

    @staticmethod
    def _entropy_term(probability: torch.Tensor) -> torch.Tensor:
        r"""
        Return :math:`p \ln p` with the convention :math:`0 \ln 0 = 0` and a finite gradient at zero.

        :param probability: Probabilities of any shape.
        :type probability: torch.Tensor
        :return: The entropy terms, of the same shape.
        :rtype: torch.Tensor
        """
        is_positive = probability > 0
        safe_probability = torch.where(is_positive, probability, torch.ones_like(probability))
        return torch.where(is_positive, probability * torch.log(safe_probability), torch.zeros_like(probability))
