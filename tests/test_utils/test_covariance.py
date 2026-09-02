import numpy as np
import pennylane as qml
import pytest
import torch

from matchcake.devices.expval_strategies.m_pfaffian._extended_covariance import displacement_vector
from matchcake.operations.state_preparation import ProductState
from matchcake.utils import covariance as covariance_module
from matchcake.utils import signed_pfaffian_complex
from matchcake.utils.covariance import (
    DEGENERATE_OVERLAP_TOL,
    DISPLACEMENT_TOL,
    basis_state_covariance_block,
    condition_occupied,
    degenerate_overlap_tol,
    lift_from_product_state,
    lift_sptm,
    transition_cov,
)

from ..configs import (
    ATOL_MATRIX_COMPARISON,
    ATOL_SCALAR_COMPARISON,
    RTOL_MATRIX_COMPARISON,
)

OCCUPIED_BLOCK = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]], dtype=float)

# Singular-value cutoff for the product-state rank check, matching the one the lift uses.
RANK_TOL = 1e-9

# The degeneracy threshold is a closed-form function of the working epsilon, so equality with
# the calibrated value is exact up to floating-point round-off, not up to a physics tolerance.
RTOL_EXACT_ARITHMETIC = 1e-12

# Calibration bounds for _TOL_EPS_EXPONENT. The conditioning law err * overlap = eps was
# measured over 375 branch pairs spanning squared overlaps 5e-3 to 0.8: median ratio 0.98,
# max 2.68. The sweep below re-measures it over a coarser grid, so the bound carries margin.
CALIBRATION_OVERLAP_RANGE = (1e-2, 0.9)
MAX_CONDITIONING_RATIO = 10.0
MAX_ERROR_AT_THRESHOLD = 1e-3

# A Pfaffian whose imaginary part survives is O(1); this only has to exclude an exact zero.
MIN_NONZERO_IMAGINARY_PART = 1e-3


class TestCovariance:
    @staticmethod
    def orthogonal_covariance(dim, seed):
        """Covariance of a pure Gaussian state: ``M @ M = -I`` and ``M = -M^T``."""
        generator = torch.Generator().manual_seed(seed)
        antisymmetric = torch.randn(dim, dim, dtype=torch.float64, generator=generator)
        antisymmetric = antisymmetric - antisymmetric.T
        rotation = torch.matrix_exp(antisymmetric)
        base = torch.zeros(dim, dim, dtype=torch.float64)
        for qubit in range(dim // 2):
            base[2 * qubit, 2 * qubit + 1] = 1.0
            base[2 * qubit + 1, 2 * qubit] = -1.0
        return rotation.T @ base @ rotation

    @staticmethod
    def random_orthogonal(dim, seed):
        generator = torch.Generator().manual_seed(seed)
        return torch.linalg.qr(torch.randn(dim, dim, dtype=torch.float64, generator=generator))[0]

    @staticmethod
    def product_state_data(amplitudes):
        """Physical covariance and displacement of a ``ProductState``, as torch float64."""
        amplitudes = np.asarray(amplitudes, dtype=complex)
        n_wires = amplitudes.shape[0]
        preparation = ProductState(amplitudes, wires=range(n_wires))
        covariance = torch.as_tensor(np.asarray(preparation.covariance_matrix), dtype=torch.float64)
        displacement = torch.as_tensor(
            np.asarray(
                displacement_vector(
                    torch.as_tensor(amplitudes, dtype=torch.complex128), qml.wires.Wires(range(n_wires))
                )
            ),
            dtype=torch.float64,
        )
        return covariance, displacement

    @staticmethod
    def overlap_squared(cov_a, cov_b):
        """``|<phi_a|phi_b>|^2 = 2^{-D/2} |Pf(Lambda_a + Lambda_b)|``."""
        dim = cov_a.shape[-1]
        pfaffian = signed_pfaffian_complex((cov_a + cov_b).to(torch.complex128))
        return float(2.0 ** (-dim / 2) * abs(complex(pfaffian)))

    @pytest.mark.parametrize("dim", [2, 4, 6])
    def test_lift_sptm_is_the_direct_sum_with_the_two_by_two_identity(self, dim):
        sptm = self.random_orthogonal(dim, seed=dim)
        lifted = lift_sptm(sptm)
        assert tuple(lifted.shape) == (dim + 2, dim + 2)
        np.testing.assert_allclose(
            qml.math.toarray(lifted[:dim, :dim]), qml.math.toarray(sptm), atol=ATOL_MATRIX_COMPARISON, rtol=0
        )
        np.testing.assert_allclose(qml.math.toarray(lifted[dim:, dim:]), np.eye(2), atol=ATOL_MATRIX_COMPARISON, rtol=0)
        np.testing.assert_allclose(qml.math.toarray(lifted[:dim, dim:]), 0.0, atol=ATOL_MATRIX_COMPARISON, rtol=0)
        np.testing.assert_allclose(qml.math.toarray(lifted[dim:, :dim]), 0.0, atol=ATOL_MATRIX_COMPARISON, rtol=0)

    @pytest.mark.parametrize("dim", [2, 4, 6])
    def test_lift_sptm_preserves_orthogonality(self, dim):
        """The ancilla block must not break the property every downstream sandwich relies on."""
        lifted = lift_sptm(self.random_orthogonal(dim, seed=dim + 50))
        np.testing.assert_allclose(
            qml.math.toarray(lifted @ lifted.T), np.eye(dim + 2), atol=ATOL_MATRIX_COMPARISON, rtol=0
        )

    @pytest.mark.parametrize("batch_shape", [(3,), (2, 4)])
    def test_lift_sptm_supports_leading_batch_dimensions(self, batch_shape):
        sptm = torch.randn(*batch_shape, 4, 4, dtype=torch.float64)
        lifted = lift_sptm(sptm)
        assert tuple(lifted.shape) == batch_shape + (6, 6)
        for index in np.ndindex(*batch_shape):
            np.testing.assert_allclose(
                qml.math.toarray(lifted[index]),
                qml.math.toarray(lift_sptm(sptm[index])),
                atol=ATOL_MATRIX_COMPARISON,
                rtol=0,
            )

    def test_lift_sptm_returns_the_input_backend_and_dtype(self):
        sptm = self.random_orthogonal(4, seed=61)
        assert isinstance(lift_sptm(sptm), torch.Tensor)
        assert lift_sptm(sptm.to(torch.float32)).dtype == torch.float32
        assert isinstance(lift_sptm(sptm.numpy()), np.ndarray)

    @pytest.mark.parametrize(
        "amplitudes",
        [
            [[0.6, 0.8]],
            [[0.6, 0.8], [1.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
            [[0.6, 0.8], [1.0, 0.0], [0.3, np.sqrt(1 - 0.09)], [0.0, 1.0]],
        ],
    )
    def test_the_lift_of_a_product_state_squares_to_minus_the_identity(self, amplitudes):
        """``M_lift @ M_lift = -I`` is what makes the lifted frame a real Majorana covariance; if it
        failed, every Pfaffian downstream would be meaningless."""
        covariance, displacement = self.product_state_data(amplitudes)
        dim = covariance.shape[-1]
        lifted = torch.as_tensor(qml.math.toarray(lift_from_product_state(covariance, displacement)))
        assert tuple(lifted.shape) == (dim + 2, dim + 2)
        np.testing.assert_allclose(
            qml.math.toarray(lifted @ lifted), -np.eye(dim + 2), atol=ATOL_MATRIX_COMPARISON, rtol=0
        )
        np.testing.assert_allclose(qml.math.toarray(lifted + lifted.T), 0.0, atol=ATOL_MATRIX_COMPARISON, rtol=0)

    def test_the_lift_keeps_the_physical_block_and_marks_the_displacement(self):
        covariance, displacement = self.product_state_data([[0.6, 0.8], [1.0, 0.0]])
        dim = covariance.shape[-1]
        lifted = torch.as_tensor(qml.math.toarray(lift_from_product_state(covariance, displacement)))
        np.testing.assert_allclose(
            qml.math.toarray(lifted[:dim, :dim]), qml.math.toarray(covariance), atol=ATOL_MATRIX_COMPARISON, rtol=0
        )
        np.testing.assert_allclose(
            qml.math.toarray(lifted[:dim, dim + 1]),
            -qml.math.toarray(displacement),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=0,
        )

    @pytest.mark.parametrize("dim", [2, 4, 6])
    def test_a_zero_displacement_decouples_the_ancilla(self, dim):
        covariance = self.orthogonal_covariance(dim, seed=dim + 70)
        lifted = torch.as_tensor(
            qml.math.toarray(lift_from_product_state(covariance, torch.zeros(dim, dtype=torch.float64)))
        )
        np.testing.assert_allclose(
            qml.math.toarray(lifted[:dim, :dim]), qml.math.toarray(covariance), atol=ATOL_MATRIX_COMPARISON, rtol=0
        )
        np.testing.assert_allclose(qml.math.toarray(lifted[:dim, dim:]), 0.0, atol=ATOL_MATRIX_COMPARISON, rtol=0)
        np.testing.assert_allclose(
            qml.math.toarray(lifted[dim:, dim:]),
            np.array([[0.0, 1.0], [-1.0, 0.0]]),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=0,
        )

    def test_a_displacement_below_the_tolerance_takes_the_basis_state_path(self):
        """The tolerance exists because ``b0 = cov @ d / m`` is ill-conditioned as ``|d| -> 0``;
        just below it the ancilla must decouple exactly rather than blow up."""
        covariance = self.orthogonal_covariance(4, seed=81)
        tiny = torch.zeros(4, dtype=torch.float64)
        tiny[0] = DISPLACEMENT_TOL / 10.0
        lifted = torch.as_tensor(qml.math.toarray(lift_from_product_state(covariance, tiny)))
        np.testing.assert_allclose(qml.math.toarray(lifted[:4, 4:]), 0.0, atol=ATOL_MATRIX_COMPARISON, rtol=0)
        np.testing.assert_allclose(
            qml.math.toarray(lifted[4:, 4:]), np.array([[0.0, 1.0], [-1.0, 0.0]]), atol=ATOL_MATRIX_COMPARISON, rtol=0
        )

    def test_the_lift_rejects_a_batched_covariance(self):
        covariance = torch.stack([self.orthogonal_covariance(4, seed=seed) for seed in (91, 92)])
        with pytest.raises(NotImplementedError, match="unbatched"):
            lift_from_product_state(covariance, torch.zeros(4, dtype=torch.float64))

    @pytest.mark.parametrize("amplitudes", [[[0.6, 0.8]], [[0.6, 0.8], [0.6, 0.8]], [[0.6, 0.8], [1.0, 0.0]]])
    def test_every_product_state_needs_only_one_ancilla_mode(self, amplitudes):
        """The invariant the lift rests on: ``rank(I + cov^2) = 2`` however many qubits are
        displaced, which is why one ancilla mode is always enough."""
        covariance, _ = self.product_state_data(amplitudes)
        dim = covariance.shape[-1]
        rank = np.linalg.matrix_rank(qml.math.toarray(covariance @ covariance) + np.eye(dim), tol=RANK_TOL)
        assert int(rank) <= 2

    def test_the_lift_rejects_a_state_that_needs_more_than_one_ancilla_mode(self):
        """A partially mixed two-qubit covariance has ``M^2 = -c^2 I``, so ``rank(I + M^2) = 4``
        and a single ancilla mode cannot carry it."""
        purity = 0.6
        covariance = torch.zeros(4, 4, dtype=torch.float64)
        for qubit in range(2):
            covariance[2 * qubit, 2 * qubit + 1] = purity
            covariance[2 * qubit + 1, 2 * qubit] = -purity
        displacement = torch.zeros(4, dtype=torch.float64)
        displacement[0] = 0.5
        with pytest.raises(AssertionError, match=r"rank\(I \+ covariance\^2\)"):
            lift_from_product_state(covariance, displacement)

    def test_the_lift_returns_the_input_backend_and_dtype(self):
        covariance = self.orthogonal_covariance(4, seed=101)
        displacement = torch.zeros(4, dtype=torch.float64)
        assert isinstance(lift_from_product_state(covariance, displacement), torch.Tensor)
        assert isinstance(lift_from_product_state(covariance.numpy(), displacement.numpy()), np.ndarray)
        assert lift_from_product_state(covariance.to(torch.float32), displacement.to(torch.float32)).dtype == (
            torch.float32
        )

    @pytest.mark.parametrize("dim", [4, 6, 8])
    def test_the_transition_covariance_of_a_branch_with_itself_is_that_branch(self, dim):
        covariance = self.orthogonal_covariance(dim, seed=dim)
        gamma = transition_cov(covariance, covariance)
        np.testing.assert_allclose(
            qml.math.toarray(gamma),
            qml.math.toarray(covariance.to(torch.complex128)),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=0,
        )

    @pytest.mark.parametrize("dim", [4, 6, 8])
    def test_the_transition_covariance_is_antisymmetric_with_a_zero_diagonal(self, dim):
        """Pfaffian routines are only defined on antisymmetric input, and the raw projector form
        has a nonzero imaginary diagonal that the antisymmetrization is there to remove."""
        gamma = transition_cov(self.orthogonal_covariance(dim, seed=dim), self.orthogonal_covariance(dim, seed=dim + 5))
        gamma = torch.as_tensor(qml.math.toarray(gamma))
        np.testing.assert_allclose(
            qml.math.toarray(gamma + gamma.transpose(-1, -2)), 0.0, atol=ATOL_MATRIX_COMPARISON, rtol=0
        )
        np.testing.assert_allclose(qml.math.toarray(torch.diagonal(gamma)), 0.0, atol=ATOL_MATRIX_COMPARISON, rtol=0)

    def test_the_transition_covariance_broadcasts_into_a_pair_grid(self):
        """One call must give the same grid as the per-pair loop it replaces."""
        covariances = torch.stack([self.orthogonal_covariance(6, seed=seed) for seed in (1, 2, 3)])
        grid = torch.as_tensor(qml.math.toarray(transition_cov(covariances[:, None], covariances[None, :])))
        assert tuple(grid.shape) == (3, 3, 6, 6)
        for row in range(3):
            for column in range(3):
                np.testing.assert_allclose(
                    qml.math.toarray(grid[row, column]),
                    qml.math.toarray(transition_cov(covariances[row], covariances[column])),
                    atol=ATOL_MATRIX_COMPARISON,
                    rtol=0,
                )

    def test_the_marker_rescale_matches_an_explicit_rescale(self):
        """The marker row and column are multiplied by ``-i`` before antisymmetrization, so the
        marker entry itself picks up ``-i`` once and the crossing entry twice."""
        cov_a = self.orthogonal_covariance(6, seed=21)
        cov_b = self.orthogonal_covariance(6, seed=22)
        marker = 5
        plain = torch.as_tensor(qml.math.toarray(transition_cov(cov_a, cov_b)))
        marked = torch.as_tensor(qml.math.toarray(transition_cov(cov_a, cov_b, marker=marker)))
        expected = plain.clone()
        expected[marker, :] = expected[marker, :] * (-1j)
        expected[:, marker] = expected[:, marker] * (-1j)
        expected = 0.5 * (expected - expected.transpose(-1, -2))
        np.testing.assert_allclose(
            qml.math.toarray(marked), qml.math.toarray(expected), atol=ATOL_MATRIX_COMPARISON, rtol=0
        )

    def test_the_transition_covariance_of_an_orthogonal_pair_stays_finite(self):
        """Orthogonal branches make the projector sum singular. The pseudo-inverse must return a
        finite (meaningless, later reweighted to zero) result rather than raise or produce NaN."""
        cov_a = basis_state_covariance_block(np.array([0, 0]), 4, torch.float64, torch.device("cpu"))
        cov_b = basis_state_covariance_block(np.array([1, 1]), 4, torch.float64, torch.device("cpu"))
        assert self.overlap_squared(cov_a, cov_b) == pytest.approx(0.0, abs=ATOL_SCALAR_COMPARISON)
        gamma = torch.as_tensor(qml.math.toarray(transition_cov(cov_a, cov_b)))
        assert torch.isfinite(gamma.real).all() and torch.isfinite(gamma.imag).all()

    def test_the_transition_covariance_returns_the_input_backend(self):
        cov_a, cov_b = self.orthogonal_covariance(4, seed=33), self.orthogonal_covariance(4, seed=34)
        assert isinstance(transition_cov(cov_a, cov_b), torch.Tensor)
        from_numpy = transition_cov(cov_a.numpy(), cov_b.numpy())
        assert isinstance(from_numpy, np.ndarray)
        np.testing.assert_allclose(
            from_numpy, qml.math.toarray(transition_cov(cov_a, cov_b)), atol=ATOL_MATRIX_COMPARISON, rtol=0
        )

    def test_the_transition_covariance_preserves_the_working_precision(self):
        cov_a = self.orthogonal_covariance(4, seed=31).to(torch.complex64)
        cov_b = self.orthogonal_covariance(4, seed=32).to(torch.complex64)
        assert transition_cov(cov_a, cov_b).dtype == torch.complex64
        assert transition_cov(cov_a.to(torch.complex128), cov_b.to(torch.complex128)).dtype == torch.complex128

    @pytest.mark.parametrize("dim, j, k", [(4, 0, 1), (6, 0, 2), (8, 1, 3), (8, 0, 3)])
    def test_conditioning_pins_the_projected_modes_and_zeroes_the_cross_block(self, dim, j, k):
        conditioned = torch.as_tensor(
            qml.math.toarray(condition_occupied(self.orthogonal_covariance(dim, seed=dim + j), j, k))
        )
        occupied = [2 * j, 2 * j + 1, 2 * k, 2 * k + 1]
        rest = [mode for mode in range(dim) if mode not in occupied]
        np.testing.assert_allclose(
            qml.math.toarray(conditioned[np.ix_(occupied, occupied)]),
            OCCUPIED_BLOCK,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=0,
        )
        np.testing.assert_allclose(
            qml.math.toarray(conditioned[np.ix_(occupied, rest)]), 0.0, atol=ATOL_MATRIX_COMPARISON, rtol=0
        )
        np.testing.assert_allclose(
            qml.math.toarray(conditioned[np.ix_(rest, occupied)]), 0.0, atol=ATOL_MATRIX_COMPARISON, rtol=0
        )

    @pytest.mark.parametrize("dim, j, k", [(6, 0, 2), (8, 1, 3)])
    def test_the_conditioned_rest_block_is_the_schur_complement(self, dim, j, k):
        """The back-action term ``C + B^T (A + Lambda_occ)^{-1} B`` recomputed independently."""
        covariance = self.orthogonal_covariance(dim, seed=dim * 7 + j)
        conditioned = torch.as_tensor(qml.math.toarray(condition_occupied(covariance, j, k)))
        occupied = [2 * j, 2 * j + 1, 2 * k, 2 * k + 1]
        rest = [mode for mode in range(dim) if mode not in occupied]
        block_a = covariance.numpy()[np.ix_(occupied, occupied)]
        block_b = covariance.numpy()[np.ix_(occupied, rest)]
        block_c = covariance.numpy()[np.ix_(rest, rest)]
        expected = block_c + block_b.T @ np.linalg.inv(block_a + OCCUPIED_BLOCK) @ block_b
        np.testing.assert_allclose(
            qml.math.toarray(conditioned[np.ix_(rest, rest)]),
            expected,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    @pytest.mark.parametrize("dim, j, k", [(4, 0, 1), (6, 0, 2)])
    def test_the_conditioned_covariance_is_antisymmetric(self, dim, j, k):
        conditioned = torch.as_tensor(
            qml.math.toarray(condition_occupied(self.orthogonal_covariance(dim, seed=dim + 200), j, k))
        )
        np.testing.assert_allclose(
            qml.math.toarray(conditioned + conditioned.transpose(-1, -2)),
            0.0,
            atol=ATOL_MATRIX_COMPARISON,
            rtol=0,
        )

    def test_conditioning_an_already_occupied_basis_state_is_idempotent(self):
        """Projecting ``|1 1>`` onto occupied must leave it exactly where it is."""
        covariance = basis_state_covariance_block(np.array([1, 1]), 4, torch.float64, torch.device("cpu"))
        conditioned = condition_occupied(covariance, 0, 1)
        np.testing.assert_allclose(
            qml.math.toarray(conditioned), qml.math.toarray(covariance), atol=ATOL_MATRIX_COMPARISON, rtol=0
        )

    def test_conditioning_a_vanishing_branch_stays_finite(self):
        """``|0 0>`` has zero weight in the occupied sector, which makes ``A + Lambda_occ``
        singular. The pseudo-inverse must give a finite result for the branch that is pruned next,
        rather than raising or producing NaN."""
        covariance = basis_state_covariance_block(np.array([0, 0, 1]), 6, torch.float64, torch.device("cpu"))
        conditioned = torch.as_tensor(qml.math.toarray(condition_occupied(covariance, 0, 1)))
        assert torch.isfinite(conditioned).all()
        np.testing.assert_allclose(
            qml.math.toarray(conditioned[:4, :4]), OCCUPIED_BLOCK, atol=ATOL_MATRIX_COMPARISON, rtol=0
        )

    def test_conditioning_supports_leading_batch_dimensions(self):
        covariances = torch.stack([self.orthogonal_covariance(6, seed=seed) for seed in (41, 42, 43)])
        batched = torch.as_tensor(qml.math.toarray(condition_occupied(covariances, 0, 2)))
        assert tuple(batched.shape) == (3, 6, 6)
        for index in range(3):
            np.testing.assert_allclose(
                qml.math.toarray(batched[index]),
                qml.math.toarray(condition_occupied(covariances[index], 0, 2)),
                atol=ATOL_MATRIX_COMPARISON,
                rtol=0,
            )

    def test_conditioning_returns_the_input_backend(self):
        covariance = self.orthogonal_covariance(4, seed=51)
        assert isinstance(condition_occupied(covariance, 0, 1), torch.Tensor)
        assert isinstance(condition_occupied(covariance.numpy(), 0, 1), np.ndarray)

    def test_conditioning_promotes_an_integer_covariance_instead_of_truncating_it(self):
        """The Schur back-action is generally non-integral. Casting the result back to an integer
        input dtype would floor it to zero, so the promoted precision has to be the one returned.

        The fixture is chosen so the back-action is genuinely fractional (2/3): a basis-state
        covariance would have an integral one and would pass either way.
        """
        covariance = torch.tensor(
            [
                [0, 2, 1, 0, 0, 0],
                [-2, 0, 0, 0, 1, 0],
                [-1, 0, 0, 3, 0, 0],
                [0, 0, -3, 0, 0, 1],
                [0, -1, 0, 0, 0, 2],
                [0, 0, 0, -1, -2, 0],
            ],
            dtype=torch.int64,
        )
        conditioned = condition_occupied(covariance, 0, 2)
        reference = condition_occupied(covariance.to(torch.float64), 0, 2)
        assert conditioned.dtype == torch.float64
        np.testing.assert_allclose(
            qml.math.toarray(conditioned), qml.math.toarray(reference), atol=ATOL_MATRIX_COMPARISON, rtol=0
        )
        rest = [2, 3]
        assert np.abs(qml.math.toarray(reference)[np.ix_(rest, rest)]).max() > 0.1

    def test_conditioning_promotes_an_integer_numpy_covariance_on_its_own_backend(self):
        covariance = np.zeros((4, 4), dtype=np.int64)
        covariance[0, 1], covariance[1, 0] = 1, -1
        covariance[2, 3], covariance[3, 2] = 1, -1
        conditioned = condition_occupied(covariance, 0, 1)
        assert isinstance(conditioned, np.ndarray)
        assert conditioned.dtype == np.float64

    @pytest.mark.parametrize("j, k", [(0, 3), (-1, 1), (0, -2), (3, 0)])
    def test_conditioning_refuses_qubits_outside_the_covariance(self, j, k):
        """A negative index would wrap around and an oversized one would fail deep inside
        ``index_select``; both should be named at the boundary instead."""
        covariance = self.orthogonal_covariance(6, seed=57)
        with pytest.raises(ValueError, match="out of range"):
            condition_occupied(covariance, j, k)

    def test_conditioning_refuses_a_complex_covariance(self):
        """The Schur complement below is real, so a complex input would lose its imaginary part
        silently. That is the same class of defect the torchpfaffian floor bump exists to fix."""
        covariance = self.orthogonal_covariance(4, seed=53).to(torch.complex128)
        with pytest.raises(ValueError, match="expects a real covariance"):
            condition_occupied(covariance, 0, 1)

    def test_the_degeneracy_threshold_is_unchanged_at_complex128(self):
        """The published certificate is a complex128 certificate; the dtype-aware form must not
        move the value it was calibrated at."""
        assert degenerate_overlap_tol(torch.complex128) == pytest.approx(
            DEGENERATE_OVERLAP_TOL, rel=RTOL_EXACT_ARITHMETIC
        )
        assert isinstance(DEGENERATE_OVERLAP_TOL, float)

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128, torch.float32, torch.float64])
    def test_the_degeneracy_threshold_is_a_probability(self, dtype):
        assert 0.0 < degenerate_overlap_tol(dtype) < 1.0

    def test_the_degeneracy_threshold_loosens_as_precision_drops(self):
        assert degenerate_overlap_tol(torch.complex64) > degenerate_overlap_tol(torch.complex128)

    @pytest.mark.parametrize(
        "tensor_dtype, expected_dtype",
        [(torch.complex64, torch.complex64), (torch.complex128, torch.complex128), (torch.float32, torch.complex64)],
    )
    def test_the_degeneracy_threshold_accepts_a_tensor_reference(self, tensor_dtype, expected_dtype):
        reference = torch.zeros(2, 2, dtype=tensor_dtype)
        assert degenerate_overlap_tol(reference) == pytest.approx(
            degenerate_overlap_tol(expected_dtype), rel=RTOL_EXACT_ARITHMETIC
        )

    def test_the_degeneracy_threshold_refuses_a_precision_with_no_sound_value(self, monkeypatch):
        """No dtype torch ships reaches this, but the guard is what keeps the certificate from
        silently becoming vacuous if the scaling is ever retuned."""
        monkeypatch.setattr(covariance_module, "_TOL_EPS_EXPONENT", 1.0)
        with pytest.raises(ValueError, match="no sound degeneracy threshold"):
            degenerate_overlap_tol(torch.complex64)

    def test_the_degeneracy_threshold_bounds_the_conditioning_error(self):
        """Calibration pin for ``_TOL_EPS_EXPONENT``.

        The overlap-normalized path loses accuracy like ``eps / overlap``, so the threshold and the
        worst-case error at the threshold are two ends of one budget. This measures the law at
        complex64 against a complex128 reference and checks that the threshold the module derives
        leaves the error where the docstring claims it does.
        """
        dim = 8
        generator = torch.Generator().manual_seed(7)
        rotation_generator = torch.randn(dim, dim, dtype=torch.float64, generator=generator)
        rotation_generator = rotation_generator - rotation_generator.T
        cov_a = basis_state_covariance_block(np.zeros(dim // 2, dtype=int), dim, torch.float64, torch.device("cpu"))

        ratios = []
        for angle in np.linspace(0.0, np.pi / 2, 60):
            rotation = torch.matrix_exp(angle * rotation_generator)
            cov_b = rotation.T @ cov_a @ rotation
            overlap = self.overlap_squared(cov_a, cov_b)
            if not CALIBRATION_OVERLAP_RANGE[0] < overlap < CALIBRATION_OVERLAP_RANGE[1]:
                continue
            reference = transition_cov(cov_a.to(torch.complex128), cov_b.to(torch.complex128))
            low = transition_cov(cov_a.to(torch.complex64), cov_b.to(torch.complex64))
            error = float(torch.linalg.norm(low.to(torch.complex128) - reference) / torch.linalg.norm(reference))
            ratios.append(error * overlap / float(torch.finfo(torch.complex64).eps))

        assert len(ratios) >= 10
        assert max(ratios) < MAX_CONDITIONING_RATIO
        tolerance = degenerate_overlap_tol(torch.complex64)
        implied_error = max(ratios) * float(torch.finfo(torch.complex64).eps) / tolerance
        assert implied_error < MAX_ERROR_AT_THRESHOLD

    @pytest.mark.parametrize("bits", [[0], [0, 1], [1, 1, 0], [0, 1, 1, 0]])
    def test_the_basis_state_block_encodes_the_outcome_bits(self, bits):
        block = basis_state_covariance_block(np.asarray(bits), 2 * len(bits), torch.complex128, torch.device("cpu"))
        expected = np.zeros((2 * len(bits), 2 * len(bits)), dtype=complex)
        for qubit, bit in enumerate(bits):
            expected[2 * qubit, 2 * qubit + 1] = 2 * bit - 1
            expected[2 * qubit + 1, 2 * qubit] = -(2 * bit - 1)
        np.testing.assert_allclose(qml.math.toarray(block), expected, atol=ATOL_MATRIX_COMPARISON, rtol=0)

    @pytest.mark.parametrize("bits", [[0, 1], [1, 1, 0]])
    def test_the_basis_state_block_is_antisymmetric(self, bits):
        block = basis_state_covariance_block(np.asarray(bits), 2 * len(bits))
        np.testing.assert_allclose(qml.math.toarray(block + block.T), 0.0, atol=ATOL_MATRIX_COMPARISON, rtol=0)

    @pytest.mark.parametrize("bits", [[0, 1], [1, 1, 0]])
    def test_a_larger_dim_leaves_the_trailing_block_zero(self, bits):
        """A physical outcome block has to be addable to a lifted ``(2n+2)`` covariance."""
        physical = 2 * len(bits)
        block = basis_state_covariance_block(np.asarray(bits), physical + 2)
        assert tuple(block.shape) == (physical + 2, physical + 2)
        np.testing.assert_allclose(qml.math.toarray(block[physical:, :]), 0.0, atol=ATOL_MATRIX_COMPARISON, rtol=0)
        np.testing.assert_allclose(qml.math.toarray(block[:, physical:]), 0.0, atol=ATOL_MATRIX_COMPARISON, rtol=0)

    @pytest.mark.parametrize("bits", [[0], [0, 1], [1, 1, 0], [0, 1, 1, 0]])
    def test_the_pfaffian_of_the_basis_state_block_is_the_product_of_the_signs(self, bits):
        block = basis_state_covariance_block(np.asarray(bits), 2 * len(bits), torch.complex128, torch.device("cpu"))
        expected = float(np.prod([2 * bit - 1 for bit in bits]))
        np.testing.assert_allclose(
            complex(signed_pfaffian_complex(block)), expected + 0.0j, atol=ATOL_SCALAR_COMPARISON
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128])
    def test_the_basis_state_block_honours_the_requested_dtype(self, dtype):
        assert basis_state_covariance_block(np.array([0, 1]), 4, dtype).dtype == dtype

    def test_the_basis_state_block_defaults_to_float64_on_the_default_device(self):
        block = basis_state_covariance_block(np.array([0, 1]), 4)
        assert block.dtype == torch.float64
        assert block.device == torch.zeros(1).device

    def test_the_basis_state_block_accepts_a_torch_bit_tensor(self):
        from_numpy = basis_state_covariance_block(np.array([1, 0, 1]), 6)
        from_torch = basis_state_covariance_block(torch.tensor([1, 0, 1]), 6)
        np.testing.assert_allclose(
            qml.math.toarray(from_torch), qml.math.toarray(from_numpy), atol=ATOL_MATRIX_COMPARISON, rtol=0
        )

    def test_the_lift_rejects_a_collapsed_ancilla_scale_instead_of_returning_nan(self):
        """``rank(I + covariance^2) <= 2`` is not a complete validity check. An all-zero covariance
        with a nonzero displacement passes it (``rank(I) = 2``) but drives
        ``scale_sq = -(d^T Lambda^2 d) / (d^T d)`` to zero, so ``b0 = Lambda d / scale`` divides by
        zero and the lift comes back full of NaN. Genuine product states keep the scale well away
        from zero, so the exact condition ``scale_sq <= 0`` separates the two with no threshold.
        """
        covariance = torch.zeros(2, 2, dtype=torch.float64)
        displacement = torch.tensor([1.0, 0.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="ancilla scale"):
            lift_from_product_state(covariance, displacement)

    def test_the_lift_accepts_the_smallest_ancilla_scale_a_product_state_produces(self):
        """Guards the rejection above against being too eager: a genuine, badly conditioned product
        state must still lift. This one has an ancilla scale near 1e-3."""
        angle = 1e-3
        covariance = torch.zeros(2, 2, dtype=torch.float64)
        covariance[0, 1], covariance[1, 0] = angle, -angle
        displacement = torch.tensor([np.sqrt(1 - angle**2), 0.0], dtype=torch.float64)
        lifted = torch.as_tensor(qml.math.toarray(lift_from_product_state(covariance, displacement)))
        assert not torch.isnan(lifted).any()
        np.testing.assert_allclose(qml.math.toarray(lifted @ lifted), -np.eye(4), atol=ATOL_MATRIX_COMPARISON, rtol=0)

    def test_conditioning_the_same_qubit_against_itself_is_rejected(self):
        # Devil's-advocate finding: j and k are documented as two qubits being jointly projected to
        # occupied. Passing j == k builds occupied_modes = [2j, 2j+1, 2j, 2j+1], a duplicate-index
        # selection, and condition_occupied silently returns a structurally antisymmetric but
        # physically undefined covariance instead of rejecting the call.
        covariance = self.orthogonal_covariance(6, seed=301)
        with pytest.raises((ValueError, AssertionError)):
            condition_occupied(covariance, 1, 1)
