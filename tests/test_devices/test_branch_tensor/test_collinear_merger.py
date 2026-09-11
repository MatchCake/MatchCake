import numpy as np
import pytest
import torch

from matchcake.devices.branch_tensor.collinear_merger import EXACT_COLLINEAR_TOL, CollinearMerger

from ...configs import (
    ATOL_MATRIX_COMPARISON,
    ATOL_SCALAR_COMPARISON,
    RTOL_MATRIX_COMPARISON,
    RTOL_SCALAR_COMPARISON,
)

# Overlap magnitude of a pair that is close to parallel without being parallel. Its Gram deficit,
# 1e-2, sits far above EXACT_COLLINEAR_TOL, so a merge at this overlap must be reported approximate.
APPROXIMATE_OVERLAP = 0.99

# Threshold loose enough to fuse a pair at APPROXIMATE_OVERLAP. Any value in (1e-2, 1) would do.
LOOSE_COLLINEAR_TH = 0.05

# Threshold and overlap chosen so that the Gram lands exactly on ``1 - collinear_th``: both 0.5 and
# 1.0 are exactly representable in binary, which 1 - 0.05 is not, so the boundary can be probed
# without the comparison turning on a round-off bit.
BOUNDARY_COLLINEAR_TH = 0.5

# Hilbert-space dimension of the explicit branch vectors backing the physics oracle. Only has to be
# large enough that independently drawn vectors are not accidentally close to parallel.
ORACLE_DIM = 6


class TestCollinearMerger:
    @staticmethod
    def weights_from_amplitudes(amplitudes, overlaps) -> torch.Tensor:
        """Weight matrix ``W_ab = conj(z_a) z_b <phi_a|phi_b>`` from explicit amplitudes and overlaps.

        :param amplitudes: Sequence of ``chi`` complex branch amplitudes.
        :param overlaps: ``(chi, chi)`` matrix of branch overlaps.
        :return: Complex ``(chi, chi)`` weight matrix.
        :rtype: torch.Tensor
        """
        amplitudes = np.asarray(amplitudes, dtype=complex)
        return torch.as_tensor(np.conj(amplitudes)[:, None] * amplitudes[None, :] * np.asarray(overlaps))

    @staticmethod
    def weights_from_vectors(amplitudes, vectors) -> torch.Tensor:
        """Weight matrix built from explicit branch vectors rather than from a table of overlaps.

        The vectors are deliberately left unnormalized: the Gram this module computes divides by
        ``sqrt(W_aa W_bb)`` and so must be blind to the branch norms.

        :param amplitudes: Sequence of ``chi`` complex branch amplitudes.
        :param vectors: ``(chi, dim)`` array of branch vectors.
        :return: Complex ``(chi, chi)`` weight matrix.
        :rtype: torch.Tensor
        """
        amplitudes = np.asarray(amplitudes, dtype=complex)
        vectors = np.asarray(vectors, dtype=complex)
        overlaps = np.conj(vectors) @ vectors.T
        return torch.as_tensor(np.conj(amplitudes)[:, None] * amplitudes[None, :] * overlaps)

    @staticmethod
    def fused_weights_from_vectors(amplitudes, vectors, groups) -> np.ndarray:
        """Independent oracle: rebuild the fused weight matrix from the merged branch vectors.

        This does not sum rows and columns of ``W``. It takes the merger's own claim at face value,
        that a group of collinear branches is one branch of amplitude
        ``sum_m z_m exp(i varphi_m)`` carrying the representative's vector, and rebuilds the weight
        matrix from the merged amplitudes. Agreement therefore pins the relative phase the merger
        never extracts explicitly.

        :param amplitudes: Sequence of ``chi`` complex branch amplitudes.
        :param vectors: ``(chi, dim)`` array of branch vectors, collinear within each group.
        :param groups: Partition of ``range(chi)`` into groups of collinear branches.
        :return: Complex ``(n_groups, n_groups)`` weight matrix.
        :rtype: np.ndarray
        """
        amplitudes = np.asarray(amplitudes, dtype=complex)
        vectors = np.asarray(vectors, dtype=complex)
        fused_amplitudes = []
        for group in groups:
            representative = vectors[group[0]]
            # phi_m = exp(i varphi_m) phi_r, so the phase is read off the ratio at the entry of the
            # representative with the largest magnitude, which is the numerically safest one.
            pivot = int(np.argmax(np.abs(representative)))
            total = 0.0 + 0.0j
            for member in group:
                phase = vectors[member][pivot] / representative[pivot]
                total = total + amplitudes[member] * phase
            fused_amplitudes.append(total)
        fused_amplitudes = np.asarray(fused_amplitudes)
        representatives = vectors[[group[0] for group in groups]]
        overlaps = np.conj(representatives) @ representatives.T
        return np.conj(fused_amplitudes)[:, None] * fused_amplitudes[None, :] * overlaps

    @staticmethod
    def fused_weights_by_summation(weights, groups) -> np.ndarray:
        """Structural oracle: fuse by explicit double summation instead of the contraction.

        :param weights: ``(chi, chi, ...)`` weight matrix.
        :param groups: Partition of ``range(chi)``.
        :return: ``(n_groups, n_groups, ...)`` fused weight matrix.
        :rtype: np.ndarray
        """
        weights = np.asarray(weights)
        batch_shape = weights.shape[2:]
        fused = np.zeros((len(groups), len(groups)) + batch_shape, dtype=weights.dtype)
        for row, group_row in enumerate(groups):
            for column, group_column in enumerate(groups):
                for member_row in group_row:
                    for member_column in group_column:
                        fused[row, column] += weights[int(member_row), int(member_column)]
        return fused

    @staticmethod
    def collinear_vectors(collinear_groups, seed: int = 0) -> np.ndarray:
        """Branch vectors that are exactly parallel within each declared group.

        :param collinear_groups: Sequence of groups of branch indices sharing a direction.
        :param seed: Seed of the generator drawing the directions, the phases and the norms.
        :return: ``(chi, ORACLE_DIM)`` complex array.
        :rtype: np.ndarray
        """
        generator = np.random.default_rng(seed)
        n_branches = sum(len(group) for group in collinear_groups)
        vectors = np.zeros((n_branches, ORACLE_DIM), dtype=complex)
        for group in collinear_groups:
            direction = generator.normal(size=ORACLE_DIM) + 1j * generator.normal(size=ORACLE_DIM)
            for member in group:
                phase = np.exp(1j * generator.uniform(0.0, 2.0 * np.pi))
                vectors[int(member)] = generator.uniform(0.5, 2.0) * phase * direction
        return vectors

    def test_gram_of_collinear_branches_is_one(self) -> None:
        phase = np.exp(1j * 0.7)
        overlaps = np.array([[1.0, phase], [np.conj(phase), 1.0]])
        weights = self.weights_from_amplitudes([0.6, 0.8], overlaps)
        np.testing.assert_allclose(
            CollinearMerger.gram_matrix(weights).numpy(),
            np.ones((2, 2)),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_gram_of_orthogonal_branches_is_zero_off_diagonal(self) -> None:
        overlaps = np.array([[1.0, 0.0], [0.0, 1.0]])
        gram = CollinearMerger.gram_matrix(self.weights_from_amplitudes([1.0, 1.0], overlaps)).numpy()
        np.testing.assert_allclose(gram, np.eye(2), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_gram_is_blind_to_the_branch_amplitudes_and_norms(self) -> None:
        """``|W_ab| / sqrt(W_aa W_bb)`` is scale invariant, so only the directions may matter."""
        vectors = self.collinear_vectors([[0], [1], [2]], seed=1)
        reference = CollinearMerger.gram_matrix(self.weights_from_vectors([1.0, 1.0, 1.0], vectors))
        rescaled = CollinearMerger.gram_matrix(
            self.weights_from_vectors([1e-6, 3.0 * np.exp(1j * 0.3), 42.0], 7.0 * vectors)
        )
        np.testing.assert_allclose(
            rescaled.numpy(),
            reference.numpy(),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_gram_is_symmetric_with_a_unit_diagonal_and_lies_in_the_unit_interval(self) -> None:
        vectors = self.collinear_vectors([[0, 1], [2], [3]], seed=2)
        gram = CollinearMerger.gram_matrix(self.weights_from_vectors([0.7, 1.3, 0.2, 2.0], vectors)).numpy()
        np.testing.assert_allclose(gram, gram.T, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(np.diag(gram), np.ones(4), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)
        assert gram.min() >= -ATOL_MATRIX_COMPARISON
        assert gram.max() <= 1.0 + ATOL_MATRIX_COMPARISON

    def test_gram_gives_a_zero_row_to_a_vanished_branch(self) -> None:
        weights = self.weights_from_amplitudes([1.0, 0.0], np.ones((2, 2)))
        gram = CollinearMerger.gram_matrix(weights).numpy()
        np.testing.assert_allclose(gram[1], np.zeros(2), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_a_vanished_branch_is_never_merged_even_at_a_loose_threshold(self) -> None:
        """Pruning is the budget's job, so the merger must leave a zero-population branch alone."""
        weights = self.weights_from_amplitudes([1.0, 0.0, 1.0], np.ones((3, 3)))
        assert CollinearMerger(collinear_th=LOOSE_COLLINEAR_TH).merge_groups(weights) == [[0, 2], [1]]

    def test_a_vanished_branch_gets_a_zero_row_even_with_a_surviving_off_diagonal(self) -> None:
        """The zero-population guard, on the only input that can tell it from a bare division.

        For a weight matrix that is internally consistent, ``W_aa = 0`` forces the whole row to
        vanish, and guarding or not guarding the division agree. They part company only when the
        population has underflowed to exactly zero while a differently rounded off-diagonal residual
        survives, which is what the guard exists for: without it the residual is divided by the
        substituted unit norm and the dead branch acquires a spurious Gram.
        """
        weights = torch.eye(3, dtype=torch.complex128)
        weights[1, 1] = 0.0
        weights[0, 1] = weights[1, 0] = 0.5
        gram = CollinearMerger.gram_matrix(weights).numpy()
        np.testing.assert_allclose(gram[1], np.zeros(3), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(gram[:, 1], np.zeros(3), atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON)

    def test_a_vanished_branch_with_a_residual_is_still_never_merged(self) -> None:
        """The consequence of the guard: pruning stays the budget's job however loose the threshold."""
        weights = torch.eye(3, dtype=torch.complex128)
        weights[1, 1] = 0.0
        weights[0, 1] = weights[1, 0] = 1.0
        assert CollinearMerger(collinear_th=LOOSE_COLLINEAR_TH).merge_groups(weights) == [[0], [1], [2]]

    @pytest.mark.parametrize(
        "groups",
        [
            [[0, 1], [1, 2]],
            [[0, 1]],
            [[0], [1]],
            [[0, 1], [2], []],
            [[0, 1], [2, 3]],
            [[0, 1], [-1]],
        ],
    )
    def test_groups_that_are_not_a_partition_are_rejected(self, groups) -> None:
        """A malformed partition is undetectable downstream, so it has to be refused here.

        Summing the named rows and columns counts a repeated branch twice and drops a branch named
        nowhere, either of which changes the norm of the state without raising. The all-singleton
        case is the reason a silent contract is not enough: it takes the fast path, so the same
        malformed input used to survive untouched there and lose a branch everywhere else.
        """
        weights = torch.eye(3, dtype=torch.complex128)
        covariances = torch.zeros(3, 2, 2)
        with pytest.raises(ValueError, match="groups must"):
            CollinearMerger.apply_groups(covariances, weights, groups)

    def test_a_partition_given_out_of_order_is_accepted(self) -> None:
        """Only the coverage is checked: the groups and their members keep the caller's order."""
        weights = torch.eye(3, dtype=torch.complex128)
        covariances = torch.zeros(3, 2, 2)
        _, fused = CollinearMerger.apply_groups(covariances, weights, [[2, 0], [1]])
        assert tuple(fused.shape) == (2, 2)

    def test_count_approximate_merges_also_rejects_a_malformed_partition(self) -> None:
        weights = torch.eye(3, dtype=torch.complex128)
        with pytest.raises(ValueError, match="groups must"):
            CollinearMerger().count_approximate_merges(weights, [[0, 1], [1, 2]])

    @pytest.mark.parametrize(
        "covariances",
        [
            torch.zeros(2, 2, 2),
            torch.zeros(4, 2, 2),
            np.zeros((2, 2, 2)),
            torch.tensor(5.0),
            5.0,
        ],
    )
    def test_a_covariance_stack_that_is_not_one_entry_per_branch_is_rejected(self, covariances) -> None:
        """Including inputs with no leading axis at all, which must not surface as a ``TypeError``."""
        weights = torch.eye(3, dtype=torch.complex128)
        with pytest.raises(ValueError, match="covariances must carry one entry per branch"):
            CollinearMerger.apply_groups(covariances, weights, [[0, 1], [2]])

    @pytest.mark.parametrize("shape", [(0, 0), (2, 2, 0), (3, 3, 2, 0)])
    def test_an_empty_weight_matrix_is_rejected(self, shape: tuple) -> None:
        """Reducing over an empty axis has no answer, and ``reshape(0, -1)`` is ambiguous."""
        weights = torch.zeros(*shape, dtype=torch.complex128)
        with pytest.raises(ValueError, match="no entries"):
            CollinearMerger.gram_matrix(weights)
        with pytest.raises(ValueError, match="no entries"):
            CollinearMerger().merge_groups(weights)
        with pytest.raises(ValueError, match="no entries"):
            CollinearMerger.apply_groups(torch.zeros(shape[0], 2, 2), weights, [[0]])

    @pytest.mark.parametrize("shape", [(2, 3), (4,), (3, 2, 5)])
    def test_a_weight_matrix_that_is_not_square_in_its_branch_axes_is_rejected(self, shape: tuple) -> None:
        weights = torch.ones(*shape, dtype=torch.complex128)
        with pytest.raises(ValueError, match=r"shape \(chi, chi, \.\.\.\)"):
            CollinearMerger.gram_matrix(weights)

    def test_gram_reduces_a_batch_by_the_minimum(self) -> None:
        weights = torch.zeros(2, 2, 3, dtype=torch.complex128)
        weights[0, 0] = 1.0
        weights[1, 1] = 1.0
        weights[0, 1] = torch.as_tensor([1.0, 0.25, 0.5], dtype=torch.complex128)
        weights[1, 0] = torch.as_tensor([1.0, 0.25, 0.5], dtype=torch.complex128)
        np.testing.assert_allclose(
            CollinearMerger.gram_matrix(weights).numpy(),
            np.array([[1.0, 0.25], [0.25, 1.0]]),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_gram_reduces_several_batch_axes_by_the_minimum(self) -> None:
        weights = torch.ones(2, 2, 2, 3, dtype=torch.complex128)
        weights[0, 1, 1, 2] = 0.125
        weights[1, 0, 1, 2] = 0.125
        gram = CollinearMerger.gram_matrix(weights).numpy()
        np.testing.assert_allclose(
            gram,
            np.array([[1.0, 0.125], [0.125, 1.0]]),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_batched_weights_merge_only_when_collinear_everywhere(self) -> None:
        weights = torch.zeros(2, 2, 2, dtype=torch.complex128)
        weights[0, 0] = 1.0
        weights[1, 1] = 1.0
        weights[0, 1] = torch.as_tensor([1.0, 0.0], dtype=torch.complex128)
        weights[1, 0] = torch.as_tensor([1.0, 0.0], dtype=torch.complex128)
        assert CollinearMerger().merge_groups(weights) == [[0], [1]]

    def test_gram_does_not_leak_the_autograd_graph(self) -> None:
        amplitudes = torch.tensor([1.0, 1.0], dtype=torch.float64, requires_grad=True)
        weights = (amplitudes[:, None] * amplitudes[None, :]).to(torch.complex128)
        assert not CollinearMerger.gram_matrix(weights).requires_grad

    def test_merging_collinear_branches_preserves_the_total_weight(self) -> None:
        phase = np.exp(1j * 0.7)
        overlaps = np.array(
            [
                [1.0, phase, 0.3],
                [np.conj(phase), 1.0, 0.3 * np.conj(phase)],
                [0.3, 0.3 * phase, 1.0],
            ]
        )
        weights = self.weights_from_amplitudes([0.6, 0.8, 0.5], overlaps)
        covariances = torch.arange(3 * 4 * 4, dtype=torch.float64).reshape(3, 4, 4)
        merger = CollinearMerger()
        groups = merger.merge_groups(weights)
        assert groups == [[0, 1], [2]]
        merged_covariances, merged_weights = merger.apply_groups(covariances, weights, groups)
        assert merged_covariances.shape == (2, 4, 4)
        np.testing.assert_allclose(
            complex(merged_weights.sum()),
            complex(weights.sum()),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )
        np.testing.assert_allclose(
            complex(merged_weights[0, 0]),
            complex(weights[0, 0] + weights[1, 1] + weights[0, 1] + weights[1, 0]),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )
        np.testing.assert_allclose(
            complex(merged_weights[0, 1]),
            complex(weights[0, 2] + weights[1, 2]),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    @pytest.mark.parametrize(
        "collinear_groups",
        [
            [[0, 1], [2]],
            [[0, 2], [1, 3]],
            [[0, 1, 2]],
            [[0, 3], [1], [2, 4, 5]],
        ],
    )
    def test_the_fused_weights_match_the_merged_branch_vectors(self, collinear_groups) -> None:
        """Physics oracle: the fused matrix is the one built from ``z_a + exp(i varphi) z_b``."""
        vectors = self.collinear_vectors(collinear_groups, seed=3)
        n_branches = vectors.shape[0]
        generator = np.random.default_rng(4)
        amplitudes = generator.normal(size=n_branches) + 1j * generator.normal(size=n_branches)
        weights = self.weights_from_vectors(amplitudes, vectors)
        covariances = torch.arange(n_branches * 4 * 4, dtype=torch.float64).reshape(n_branches, 4, 4)

        groups = [[int(member) for member in group] for group in collinear_groups]
        _, merged_weights = CollinearMerger.apply_groups(covariances, weights, groups)
        np.testing.assert_allclose(
            merged_weights.numpy(),
            self.fused_weights_from_vectors(amplitudes, vectors, groups),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize(
        "groups",
        [
            [[0, 1], [2], [3]],
            [[1, 3], [0, 2]],
            [[3], [0, 1, 2]],
            [[0], [1], [2], [3]],
        ],
    )
    def test_the_fused_weights_are_the_group_block_sums(self, groups) -> None:
        """Structural oracle: the contraction must agree with an explicit double summation.

        The groups here are arbitrary partitions rather than collinear ones, which is exactly the
        point: the contraction is pure bookkeeping and must not depend on the branches agreeing.
        """
        generator = np.random.default_rng(5)
        weights = torch.as_tensor(generator.normal(size=(4, 4)) + 1j * generator.normal(size=(4, 4)))
        covariances = torch.arange(4 * 2 * 2, dtype=torch.float64).reshape(4, 2, 2)
        _, merged_weights = CollinearMerger.apply_groups(covariances, weights, groups)
        np.testing.assert_allclose(
            merged_weights.numpy(),
            self.fused_weights_by_summation(weights.numpy(), groups),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_the_fused_weights_keep_the_batch_axes(self) -> None:
        generator = np.random.default_rng(6)
        weights = torch.as_tensor(generator.normal(size=(4, 4, 3)) + 1j * generator.normal(size=(4, 4, 3)))
        covariances = torch.arange(4 * 2 * 2, dtype=torch.float64).reshape(4, 2, 2)
        groups = [[0, 2], [1], [3]]
        _, merged_weights = CollinearMerger.apply_groups(covariances, weights, groups)
        assert tuple(merged_weights.shape) == (3, 3, 3)
        np.testing.assert_allclose(
            merged_weights.numpy(),
            self.fused_weights_by_summation(weights.numpy(), groups),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_merging_keeps_the_representative_covariance(self) -> None:
        weights = self.weights_from_amplitudes([1.0, 1.0], np.ones((2, 2)))
        covariances = torch.stack([torch.full((4, 4), 1.0), torch.full((4, 4), 2.0)])
        merger = CollinearMerger()
        merged_covariances, _ = merger.apply_groups(covariances, weights, merger.merge_groups(weights))
        np.testing.assert_allclose(
            merged_covariances.numpy(),
            covariances[:1].numpy(),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_merging_keeps_the_representative_of_each_group_in_group_order(self) -> None:
        """The kept covariance is ``group[0]``, and the groups keep the order they were given in."""
        covariances = torch.arange(5 * 2 * 2, dtype=torch.float64).reshape(5, 2, 2)
        weights = torch.eye(5, dtype=torch.complex128)
        groups = [[3, 0], [4], [1, 2]]
        merged_covariances, _ = CollinearMerger.apply_groups(covariances, weights, groups)
        np.testing.assert_allclose(
            merged_covariances.numpy(),
            covariances[[3, 4, 1]].numpy(),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_merging_keeps_batched_covariances(self) -> None:
        covariances = torch.arange(3 * 2 * 4 * 4, dtype=torch.float64).reshape(3, 2, 4, 4)
        weights = torch.eye(3, dtype=torch.complex128)
        merged_covariances, _ = CollinearMerger.apply_groups(covariances, weights, [[0, 1], [2]])
        assert tuple(merged_covariances.shape) == (2, 2, 4, 4)
        np.testing.assert_allclose(
            merged_covariances.numpy(),
            covariances[[0, 2]].numpy(),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_apply_groups_is_a_no_op_when_nothing_merges(self) -> None:
        weights = self.weights_from_amplitudes([1.0, 1.0], np.eye(2))
        covariances = torch.zeros(2, 4, 4)
        merged_covariances, merged_weights = CollinearMerger.apply_groups(covariances, weights, [[0], [1]])
        assert merged_covariances is covariances
        assert merged_weights is weights

    def test_apply_groups_returns_torch_tensors_for_numpy_inputs(self) -> None:
        """Documents the current backend behaviour of the merging path, which is torch only."""
        covariances = np.arange(2 * 4 * 4, dtype=float).reshape(2, 4, 4)
        weights = self.weights_from_amplitudes([1.0, 1.0], np.ones((2, 2))).numpy()
        merged_covariances, merged_weights = CollinearMerger.apply_groups(covariances, weights, [[0, 1]])
        assert isinstance(merged_weights, torch.Tensor)
        assert isinstance(merged_covariances, np.ndarray)

    def test_a_single_branch_state_is_left_alone(self) -> None:
        weights = self.weights_from_amplitudes([1.0], np.ones((1, 1)))
        merger = CollinearMerger()
        assert merger.merge_groups(weights) == [[0]]
        assert merger.count_approximate_merges(weights, [[0]]) == 0

    def test_merge_groups_compares_against_the_representative_not_the_whole_group(self) -> None:
        """Collinearity is only transitive up to the threshold, and the greedy pass is single sweep.

        Branch 2 is close to branch 1 but not to branch 0. Branch 1 has already joined branch 0's
        group, and the comparison is against the representative, so branch 2 starts its own group.
        """
        weights = torch.eye(3, dtype=torch.complex128)
        weights[0, 1] = weights[1, 0] = APPROXIMATE_OVERLAP
        weights[1, 2] = weights[2, 1] = APPROXIMATE_OVERLAP
        weights[0, 2] = weights[2, 0] = 0.5
        assert CollinearMerger(collinear_th=LOOSE_COLLINEAR_TH).merge_groups(weights) == [[0, 1], [2]]

    def test_a_gram_exactly_at_the_threshold_merges(self) -> None:
        weights = torch.eye(2, dtype=torch.complex128)
        weights[0, 1] = weights[1, 0] = 1.0 - BOUNDARY_COLLINEAR_TH
        assert CollinearMerger(collinear_th=BOUNDARY_COLLINEAR_TH).merge_groups(weights) == [[0, 1]]

    def test_a_gram_just_below_the_threshold_does_not_merge(self) -> None:
        weights = torch.eye(2, dtype=torch.complex128)
        weights[0, 1] = weights[1, 0] = 0.25
        assert CollinearMerger(collinear_th=BOUNDARY_COLLINEAR_TH).merge_groups(weights) == [[0], [1]]

    def test_raising_the_threshold_turns_the_merge_approximate(self) -> None:
        overlaps = np.array([[1.0, APPROXIMATE_OVERLAP], [APPROXIMATE_OVERLAP, 1.0]])
        weights = self.weights_from_amplitudes([1.0, 1.0], overlaps)
        exact_merger = CollinearMerger()
        assert exact_merger.merge_groups(weights) == [[0], [1]]
        loose_merger = CollinearMerger(collinear_th=LOOSE_COLLINEAR_TH)
        groups = loose_merger.merge_groups(weights)
        assert groups == [[0, 1]]
        assert loose_merger.count_approximate_merges(weights, groups) == 1

    def test_an_exact_merge_is_not_counted_as_approximate(self) -> None:
        weights = self.weights_from_amplitudes([1.0, 1.0], np.ones((2, 2)))
        merger = CollinearMerger()
        groups = merger.merge_groups(weights)
        assert merger.count_approximate_merges(weights, groups) == 0

    def test_only_merges_inside_a_group_are_counted(self) -> None:
        """Each fused branch is judged against its own representative, not against every group.

        Two independent pairs are fused, one approximately and one exactly. Counting a fused branch
        against a representative it was never merged into would report every cross pair as well.
        """
        weights = torch.eye(4, dtype=torch.complex128)
        weights[0, 1] = weights[1, 0] = APPROXIMATE_OVERLAP
        weights[2, 3] = weights[3, 2] = 1.0
        merger = CollinearMerger(collinear_th=LOOSE_COLLINEAR_TH)
        groups = merger.merge_groups(weights)
        assert groups == [[0, 1], [2, 3]]
        assert merger.count_approximate_merges(weights, groups) == 1

    def test_every_approximate_member_of_a_group_is_counted(self) -> None:
        weights = torch.eye(3, dtype=torch.complex128)
        weights[0, 1] = weights[1, 0] = APPROXIMATE_OVERLAP
        weights[0, 2] = weights[2, 0] = APPROXIMATE_OVERLAP
        weights[1, 2] = weights[2, 1] = APPROXIMATE_OVERLAP
        merger = CollinearMerger(collinear_th=LOOSE_COLLINEAR_TH)
        groups = merger.merge_groups(weights)
        assert groups == [[0, 1, 2]]
        assert merger.count_approximate_merges(weights, groups) == 2

    def test_nothing_is_counted_when_nothing_was_merged(self) -> None:
        weights = torch.eye(3, dtype=torch.complex128)
        merger = CollinearMerger(collinear_th=LOOSE_COLLINEAR_TH)
        assert merger.count_approximate_merges(weights, [[0], [1], [2]]) == 0

    def test_an_exact_group_next_to_an_approximate_one_is_not_counted(self) -> None:
        """Mixed groups: only the approximately fused member may raise the count."""
        weights = torch.eye(4, dtype=torch.complex128)
        weights[0, 1] = weights[1, 0] = 1.0
        weights[2, 3] = weights[3, 2] = APPROXIMATE_OVERLAP
        merger = CollinearMerger(collinear_th=LOOSE_COLLINEAR_TH)
        groups = merger.merge_groups(weights)
        assert groups == [[0, 1], [2, 3]]
        assert merger.count_approximate_merges(weights, groups) == 1

    def test_rejects_a_negative_threshold_and_reports_it_in_repr(self) -> None:
        with pytest.raises(ValueError, match="collinear_th must be non-negative"):
            CollinearMerger(collinear_th=-1.0)
        representation = repr(CollinearMerger())
        assert "CollinearMerger" in representation
        assert str(EXACT_COLLINEAR_TOL) in representation

    @pytest.mark.parametrize("collinear_th", [1.0, 1.5, 2.0])
    def test_a_threshold_at_or_above_one_is_rejected(self, collinear_th: float) -> None:
        """At and above 1 the threshold is non-positive, which every Gram entry clears."""
        with pytest.raises(ValueError, match="collinear_th must be non-negative and below 1"):
            CollinearMerger(collinear_th=collinear_th)

    def test_the_default_threshold_is_the_exact_one(self) -> None:
        assert CollinearMerger().collinear_th == EXACT_COLLINEAR_TOL

    def test_the_threshold_is_stored_as_a_float(self) -> None:
        assert isinstance(CollinearMerger(collinear_th=0).collinear_th, float)
