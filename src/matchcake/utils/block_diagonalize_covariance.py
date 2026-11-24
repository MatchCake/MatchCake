from typing import Tuple

import numpy as np
import torch

from .torch_utils import infer_complex_dtype

DEGENERACY_ATOL = 1e-8


class BlockDiagonalizeCovariance(torch.autograd.Function):
    r"""
    Autograd function behind
    :meth:`~matchcake.utils.covariance_matrix_toolbox.CovarianceMatrixToolbox.block_diagonalize`.

    The forward pass block-diagonalizes a real antisymmetric matrix
    :math:`\Lambda = R^\top \Lambda_D R` where :math:`R \in SO(2m)` and
    :math:`\Lambda_D = \bigoplus_j (-\lambda_j) J` with
    :math:`J = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}`.
    The eigenvectors of the Hermitian matrix :math:`i\Lambda` with non-negative eigenvalues give the
    real vectors :math:`u_j = \sqrt{2}\,\mathrm{Re}(v_j)` and :math:`w_j = \sqrt{2}\,\mathrm{Im}(v_j)` which
    form the rows of :math:`R`. Zero eigenvalues are degenerate between the positive and the negative
    sector, so their vectors are replaced by an orthonormal basis of the kernel of :math:`\Lambda` through
    a QR factorization. The polarizations are read from the actual product :math:`R \Lambda R^\top`, which
    makes the forward pass insensitive to the signs and the pairing chosen by the linear algebra kernels.

    The backward pass maps the gradients with respect to :math:`R` and :math:`\lambda` back to a gradient
    with respect to :math:`\Lambda`. A perturbation :math:`dR = \Omega R` with :math:`\Omega` antisymmetric
    gives :math:`R\, d\Lambda\, R^\top = [\Lambda_D, \Omega] + d\Lambda_D`, which is inverted block by block.
    Directions that leave :math:`\Lambda` unchanged (rotations inside a mode, and rotations between two modes
    with equal polarizations or with opposite polarizations) are gauge freedoms of the decomposition. Their
    inverse is not defined, so they are projected out: a loss that only depends on :math:`\Lambda` has no
    gradient component along them.
    """

    @staticmethod
    def forward(ctx, covariance_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Block-diagonalize a batch of real antisymmetric matrices.

        :param ctx: Autograd context.
        :param covariance_matrix: Real matrices of shape ``(..., 2m, 2m)``.
        :type covariance_matrix: torch.Tensor
        :return: The transition matrices ``(..., 2m, 2m)`` and the polarizations ``(..., m)``.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        covariance_matrix = 0.5 * (covariance_matrix - covariance_matrix.mT)
        n_modes = covariance_matrix.shape[-1] // 2
        complex_dtype = infer_complex_dtype(covariance_matrix)
        hermitian = 1j * covariance_matrix.to(complex_dtype)
        _, eigenvectors = torch.linalg.eigh(hermitian)  # eigenvalues ascending: (-lambda_j, ..., +lambda_j)
        # The last n_modes columns hold the non-negative eigenvalues. Reverse them so that the modes with the
        # largest eigenvalues come first and the (possibly degenerate) zero modes are last.
        positive_vectors = torch.flip(eigenvectors[..., :, n_modes:], dims=[-1])  # (..., 2m, m)
        real_columns = np.sqrt(2.0) * torch.real(positive_vectors)
        imag_columns = np.sqrt(2.0) * torch.imag(positive_vectors)
        interleaved = torch.stack([real_columns, imag_columns], dim=-1)  # (..., 2m, m, 2)
        raw_basis = interleaved.reshape(*covariance_matrix.shape[:-2], 2 * n_modes, 2 * n_modes)
        # A QR factorization re-orthonormalizes the columns. The columns of the modes with a positive
        # eigenvalue are already orthonormal and are kept up to a sign, while the columns of the zero
        # modes are completed to an orthonormal basis of the kernel of the covariance matrix.
        orthonormal_basis, _ = torch.linalg.qr(raw_basis, mode="complete")
        sptm = orthonormal_basis.mT  # rows are the orthonormal vectors
        # Enforce a unit determinant by flipping the sign of the second row when needed. The flip changes
        # the sign of the first polarization, which is read from the actual product below.
        determinant_sign = torch.sign(torch.linalg.det(sptm))
        row_signs = torch.ones_like(sptm[..., :, 0])
        row_signs[..., 1] = determinant_sign
        sptm = sptm * row_signs[..., :, None]
        block_diagonal = sptm @ covariance_matrix @ sptm.mT
        polarizations = -block_diagonal[..., 0::2, 1::2].diagonal(dim1=-2, dim2=-1)  # (..., m)
        ctx.save_for_backward(sptm, polarizations)
        return sptm, polarizations

    @staticmethod
    def backward(ctx, grad_sptm: torch.Tensor, grad_polarizations: torch.Tensor) -> torch.Tensor:
        r"""
        Map the gradients of the decomposition back to the covariance matrix.

        :param ctx: Autograd context.
        :param grad_sptm: Gradient with respect to the transition matrices, shape ``(..., 2m, 2m)``.
        :type grad_sptm: torch.Tensor
        :param grad_polarizations: Gradient with respect to the polarizations, shape ``(..., m)``.
        :type grad_polarizations: torch.Tensor
        :return: Antisymmetric gradient with respect to the covariance matrices, shape ``(..., 2m, 2m)``.
        :rtype: torch.Tensor
        """
        sptm, polarizations = ctx.saved_tensors
        n_modes = polarizations.shape[-1]
        batch_shape = polarizations.shape[:-1]
        batch_axes = tuple(range(len(batch_shape)))
        # <G_R, Omega R> = <G_R R^T, Omega>; only the antisymmetric part couples to Omega.
        rotation_grad = grad_sptm @ sptm.mT
        rotation_grad = 0.5 * (rotation_grad - rotation_grad.mT)
        blocks = rotation_grad.reshape(*batch_shape, n_modes, 2, n_modes, 2).permute(
            *batch_axes, -4, -2, -3, -1
        )  # (..., m, m, 2, 2)
        # Decompose each 2x2 block on the orthogonal basis {I, J, K, L}.
        identity_part = 0.5 * (blocks[..., 0, 0] + blocks[..., 1, 1])
        rotation_part = 0.5 * (blocks[..., 0, 1] - blocks[..., 1, 0])
        diagonal_part = 0.5 * (blocks[..., 0, 0] - blocks[..., 1, 1])
        symmetric_part = 0.5 * (blocks[..., 0, 1] + blocks[..., 1, 0])

        differences = polarizations[..., :, None] - polarizations[..., None, :]
        sums = polarizations[..., :, None] + polarizations[..., None, :]
        inverse_differences = torch.where(
            torch.abs(differences) > DEGENERACY_ATOL, 1.0 / differences, torch.zeros_like(differences)
        )
        inverse_sums = torch.where(torch.abs(sums) > DEGENERACY_ATOL, 1.0 / sums, torch.zeros_like(sums))

        identity_coefficient = rotation_part * inverse_differences
        rotation_coefficient = -identity_part * inverse_differences
        diagonal_coefficient = -symmetric_part * inverse_sums
        symmetric_coefficient = diagonal_part * inverse_sums

        adjoint_blocks = torch.zeros_like(blocks)
        adjoint_blocks[..., 0, 0] = identity_coefficient + diagonal_coefficient
        adjoint_blocks[..., 0, 1] = rotation_coefficient + symmetric_coefficient
        adjoint_blocks[..., 1, 0] = -rotation_coefficient + symmetric_coefficient
        adjoint_blocks[..., 1, 1] = identity_coefficient - diagonal_coefficient
        # Diagonal blocks carry the polarization gradients: N_jj = -(g_j / 2) J.
        mode_index = torch.arange(n_modes, device=sptm.device)
        adjoint_blocks[..., mode_index, mode_index, 0, 1] = -0.5 * grad_polarizations
        adjoint_blocks[..., mode_index, mode_index, 1, 0] = 0.5 * grad_polarizations

        adjoint = adjoint_blocks.permute(*batch_axes, -4, -2, -3, -1).reshape(*batch_shape, 2 * n_modes, 2 * n_modes)
        grad_covariance = sptm.mT @ adjoint @ sptm
        return 0.5 * (grad_covariance - grad_covariance.mT)
