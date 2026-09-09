import itertools
from unittest import mock

import numpy as np
import pennylane as qml
import pytest
import torch

import matchcake.utils._pfaffian as pfaffian_module
from matchcake.devices.branch_tensor.wick_reduction import WickReduction
from matchcake.utils.covariance import basis_state_covariance_block, transition_cov
from matchcake.utils.majorana import MajoranaGetter

from ...configs import (
    ATOL_APPROX_COMPARISON,
    ATOL_MATRIX_COMPARISON,
    ATOL_SCALAR_COMPARISON,
    RTOL_APPROX_COMPARISON,
    RTOL_SCALAR_COMPARISON,
)

DIM = 8

# Qubit count of the dense oracles below. Three qubits give a 6-mode Majorana algebra, so every even
# support up to weight 6 is enumerable while the dense matrices stay 8x8.
ORACLE_N_QUBITS = 3
ORACLE_DIM = 2 * ORACLE_N_QUBITS

# Only has to exclude an exact zero; the values compared against it are O(1).
MIN_NONZERO_VALUE = 1e-3


class TestWickReduction:
    @staticmethod
    def transition_covariance(batch_shape: tuple = (), seed: int = 0, dim: int = DIM) -> torch.Tensor:
        """Random complex antisymmetric ``(*batch_shape, dim, dim)`` tensor."""
        generator = torch.Generator().manual_seed(seed)
        real = torch.randn(*batch_shape, dim, dim, dtype=torch.float64, generator=generator)
        imaginary = torch.randn(*batch_shape, dim, dim, dtype=torch.float64, generator=generator)
        matrix = (real + 1j * imaginary).to(torch.complex128)
        return 0.5 * (matrix - matrix.transpose(-1, -2))

    @staticmethod
    def pfaffian_by_expansion(matrix: np.ndarray) -> complex:
        """Independent oracle: the recursive first-row Pfaffian expansion.

        Deliberately shares no code with the Pfaffian backend ``element`` delegates to, so a change
        of sign convention or of pivoting strategy in that backend cannot pass unnoticed.
        """
        size = matrix.shape[0]
        if size == 0:
            return complex(1.0)
        total = 0.0 + 0.0j
        for column in range(1, size):
            keep = [index for index in range(1, size) if index != column]
            minor = matrix[np.ix_(keep, keep)]
            total += ((-1) ** (column + 1)) * matrix[0, column] * TestWickReduction.pfaffian_by_expansion(minor)
        return complex(total)

    @staticmethod
    def naive_sum(supports, coefficients, gamma: torch.Tensor) -> torch.Tensor:
        """Reference: one :meth:`WickReduction.element` call per monomial, summed in Python."""
        total = torch.zeros(tuple(gamma.shape[:-2]), dtype=gamma.dtype, device=gamma.device)
        for support, coefficient in zip(supports, coefficients):
            total = total + complex(coefficient) * WickReduction.element(support, gamma)
        return total

    @staticmethod
    def random_supports(count: int, dim: int = DIM, seed: int = 0):
        """``count`` sorted even-length supports drawn from ``range(dim)``, sizes mixed."""
        rng = np.random.default_rng(seed)
        supports = []
        for _ in range(count):
            size = int(rng.choice([0, 2, 4, 6]))
            supports.append(tuple(sorted(rng.choice(dim, size=size, replace=False).tolist())))
        return supports

    @staticmethod
    def even_supports(dim: int = ORACLE_DIM):
        """Every sorted, duplicate-free support of even size drawn from ``range(dim)``."""
        return [support for size in range(0, dim + 1, 2) for support in itertools.combinations(range(dim), size)]

    @staticmethod
    def dense_majorana_monomial(support, n_qubits: int = ORACLE_N_QUBITS) -> np.ndarray:
        """``c_{s1} c_{s2} ... c_{sw}`` in sorted order, as a dense ``(2**n, 2**n)`` matrix."""
        getter = MajoranaGetter(n_qubits)
        matrix = np.eye(2**n_qubits, dtype=complex)
        for mode in support:
            matrix = matrix @ np.asarray(qml.math.toarray(getter[mode]), dtype=complex)
        return matrix

    @staticmethod
    def covariance_from_statevector(vector: np.ndarray, n_qubits: int = ORACLE_N_QUBITS) -> torch.Tensor:
        r"""Majorana covariance ``Lambda_{mu nu} = (i/2) <[c_mu, c_nu]>`` of a pure state.

        This is the sign convention that reproduces
        :func:`~matchcake.utils.covariance.basis_state_covariance_block`, where
        ``Lambda_{2k, 2k+1} = 2 y_k - 1``. The opposite sign swaps the projector roles inside
        :func:`~matchcake.utils.covariance.transition_cov` and silently conjugates every Wick value,
        which is why it is pinned by ``test_the_covariance_oracle_matches_the_basis_state_block``.
        """
        getter = MajoranaGetter(n_qubits)
        dim = 2 * n_qubits
        majoranas = [np.asarray(qml.math.toarray(getter[mode]), dtype=complex) for mode in range(dim)]
        covariance = np.zeros((dim, dim), dtype=complex)
        for mu in range(dim):
            for nu in range(dim):
                commutator = majoranas[mu] @ majoranas[nu] - majoranas[nu] @ majoranas[mu]
                covariance[mu, nu] = 0.5j * (vector.conj() @ (commutator @ vector))
        return torch.as_tensor(covariance.real, dtype=torch.float64)

    @classmethod
    def gaussian_state(cls, seed: int, n_qubits: int = ORACLE_N_QUBITS):
        """Statevector and covariance of a matchgate circuit applied to ``|0...0>``.

        Only parity-preserving matchgates are used, so the result stays a pure Gaussian state with
        zero displacement (``M @ M = -I``) and the basis-path transition covariance applies. A
        single non-matchgate rotation here would leave the state outside the Gaussian manifold and
        the oracle would be meaningless.
        """
        rng = np.random.default_rng(seed)
        angles = rng.normal(size=6)
        device = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(device)
        def statevector():
            qml.IsingXX(angles[0], wires=[0, 1])
            qml.IsingXX(angles[1], wires=[1, 2])
            qml.IsingYY(angles[2], wires=[1, 2])
            qml.IsingXX(angles[3], wires=[0, 1])
            qml.RZ(angles[4], wires=2)
            qml.IsingYY(angles[5], wires=[0, 1])
            return qml.state()

        vector = np.asarray(statevector(), dtype=complex)
        return vector, cls.covariance_from_statevector(vector, n_qubits)

    def test_the_phase_follows_the_uniform_wick_rule(self) -> None:
        assert WickReduction.phase(0) == 1
        assert WickReduction.phase(2) == pytest.approx(-1j)
        assert WickReduction.phase(4) == pytest.approx(-1)
        assert WickReduction.phase(6) == pytest.approx(1j)

    def test_an_empty_support_broadcasts_the_constant_one(self) -> None:
        gamma = self.transition_covariance(batch_shape=(3, 2), seed=1)
        value = WickReduction.element((), gamma)
        assert tuple(value.shape) == (3, 2)
        np.testing.assert_allclose(value.numpy(), np.ones((3, 2)), atol=ATOL_SCALAR_COMPARISON, rtol=0)

    def test_a_weight_two_element_is_the_matrix_entry(self) -> None:
        """``Pf`` of a 2x2 antisymmetric block is its upper entry, so this pins the weight-two value
        against something with no Pfaffian machinery in it at all.

        The phase is written out as a literal rather than fetched from ``WickReduction.phase``: a
        test that builds its expectation from the module under test moves with any change to it and
        cannot catch a phase-convention error.
        """
        gamma = self.transition_covariance(seed=2)
        value = WickReduction.element((1, 4), gamma)
        np.testing.assert_allclose(complex(value), complex(-1j * gamma[1, 4]), atol=ATOL_SCALAR_COMPARISON, rtol=0)

    @pytest.mark.parametrize("size", [0, 2, 4, 6, 8])
    def test_an_element_matches_an_independent_pfaffian_expansion(self, size: int) -> None:
        """Pins every even weight, not just weight two, against a Pfaffian oracle that shares no
        code with the backend. Without this the only external check on ``element`` is the 2x2 case,
        and a sign error appearing first at weight four would pass."""
        gamma = self.transition_covariance(seed=20 + size)
        support = tuple(range(size))
        submatrix = gamma.numpy()[np.ix_(support, support)] if size else np.zeros((0, 0), dtype=complex)
        # ``(-1j) ** (w / 2)`` is ``i ** (-w / 2)`` written independently of the module, so the
        # expected value does not move when the module's phase convention does.
        expected = ((-1j) ** (size // 2)) * self.pfaffian_by_expansion(submatrix)
        np.testing.assert_allclose(
            complex(WickReduction.element(support, gamma)),
            expected,
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_the_covariance_oracle_matches_the_basis_state_block(self) -> None:
        """Guards the sign convention the two physics oracles below stand on. A flipped sign here
        would conjugate every expected value and still look self-consistent."""
        for bits in itertools.product([0, 1], repeat=ORACLE_N_QUBITS):
            vector = np.zeros(2**ORACLE_N_QUBITS, dtype=complex)
            vector[int("".join(str(bit) for bit in bits), 2)] = 1.0
            np.testing.assert_allclose(
                self.covariance_from_statevector(vector).numpy(),
                qml.math.toarray(basis_state_covariance_block(np.asarray(bits), ORACLE_DIM)),
                atol=ATOL_MATRIX_COMPARISON,
                rtol=0,
            )

    def test_a_diagonal_element_is_the_basis_state_expectation(self) -> None:
        r"""The physical contract in its simplest form: with ``Gamma = transition_cov(L_y, L_y)``,
        ``element(S, Gamma)`` is ``<y| c_S |y>`` computed from dense Majorana matrices.

        Every even support on three qubits is swept, so the phase convention is pinned at each
        weight against operators built independently of the Pfaffian path.
        """
        for bits in itertools.product([0, 1], repeat=ORACLE_N_QUBITS):
            lambda_y = basis_state_covariance_block(np.asarray(bits), ORACLE_DIM)
            gamma = torch.as_tensor(qml.math.toarray(transition_cov(lambda_y, lambda_y)), dtype=torch.complex128)
            vector = np.zeros(2**ORACLE_N_QUBITS, dtype=complex)
            vector[int("".join(str(bit) for bit in bits), 2)] = 1.0
            for support in self.even_supports():
                exact = complex(vector.conj() @ (self.dense_majorana_monomial(support) @ vector))
                np.testing.assert_allclose(
                    complex(WickReduction.element(support, gamma)),
                    exact,
                    atol=ATOL_SCALAR_COMPARISON,
                    rtol=0,
                    err_msg=f"bits={bits} support={support}",
                )

    def test_an_off_diagonal_element_is_the_overlap_normalized_matrix_element(self) -> None:
        r"""The contract the branch engine actually consumes:

        .. math::
            \mathrm{element}(S, \Gamma_{ab}) = \frac{\langle a| c_S |b\rangle}{\langle a|b\rangle},
            \qquad \Gamma_{ab} = \mathrm{transition\_cov}(\Lambda_a, \Lambda_b).

        Two distinct non-orthogonal Gaussian states, every even support. This is what fixes the
        argument order of ``transition_cov`` relative to the bra and the ket: swapping them
        conjugates the answer, which the diagonal test above cannot see.
        """
        vector_a, cov_a = self.gaussian_state(seed=1)
        vector_b, cov_b = self.gaussian_state(seed=2)
        np.testing.assert_allclose(
            qml.math.toarray(cov_a @ cov_a), -np.eye(ORACLE_DIM), atol=ATOL_MATRIX_COMPARISON, rtol=0
        )
        overlap = complex(vector_a.conj() @ vector_b)
        assert abs(overlap) > 0.1, "the two states must be comfortably non-orthogonal for 0/0 not to bite"

        gamma = torch.as_tensor(qml.math.toarray(transition_cov(cov_a, cov_b)), dtype=torch.complex128)
        for support in self.even_supports():
            exact = complex(vector_a.conj() @ (self.dense_majorana_monomial(support) @ vector_b)) / overlap
            np.testing.assert_allclose(
                complex(WickReduction.element(support, gamma)),
                exact,
                atol=ATOL_SCALAR_COMPARISON,
                rtol=RTOL_SCALAR_COMPARISON,
                err_msg=f"support={support}",
            )

    def test_a_summed_observable_matches_the_dense_matrix_element(self) -> None:
        """``sum_monomials`` on the same pair, so the batched path is checked against physics rather
        than only against its own single-term counterpart."""
        vector_a, cov_a = self.gaussian_state(seed=3)
        vector_b, cov_b = self.gaussian_state(seed=4)
        overlap = complex(vector_a.conj() @ vector_b)
        gamma = torch.as_tensor(qml.math.toarray(transition_cov(cov_a, cov_b)), dtype=torch.complex128)

        rng = np.random.default_rng(31)
        supports = [tuple(support) for support in self.even_supports() if len(support) in (0, 2, 4)]
        coefficients = (rng.normal(size=len(supports)) + 1j * rng.normal(size=len(supports))).tolist()

        dense = np.zeros((2**ORACLE_N_QUBITS, 2**ORACLE_N_QUBITS), dtype=complex)
        for support, coefficient in zip(supports, coefficients):
            dense = dense + complex(coefficient) * self.dense_majorana_monomial(support)
        exact = complex(vector_a.conj() @ (dense @ vector_b)) / overlap
        np.testing.assert_allclose(
            complex(WickReduction.sum_monomials(supports, coefficients, gamma)),
            exact,
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_an_odd_support_is_refused_by_both_entry_points(self) -> None:
        gamma = self.transition_covariance(seed=3)
        with pytest.raises(NotImplementedError, match="Odd Majorana support"):
            WickReduction.element((0, 1, 2), gamma)
        with pytest.raises(NotImplementedError, match="Odd Majorana support"):
            WickReduction.sum_monomials([(0, 1, 2)], [1.0], gamma)

    def test_a_negative_mode_index_is_refused_by_both_entry_points(self) -> None:
        """Torch advanced indexing wraps -1 to D-1, so without this guard ``sum_monomials`` would
        return a plausible wrong number where ``element`` raises."""
        gamma = self.transition_covariance(seed=13)
        with pytest.raises(ValueError, match="negative mode index"):
            WickReduction.sum_monomials([(0, -1)], [1.0], gamma)
        with pytest.raises(ValueError, match="negative mode index"):
            WickReduction.element((0, -1), gamma)

    def test_a_mode_index_past_the_covariance_is_not_silently_accepted(self) -> None:
        """Unlike a negative index there is no explicit guard for this one; the point is only that
        it fails loudly from the indexing layer rather than selecting the wrong modes."""
        gamma = self.transition_covariance(seed=14)
        with pytest.raises((IndexError, RuntimeError)):
            WickReduction.element((0, DIM), gamma)
        with pytest.raises((IndexError, RuntimeError)):
            WickReduction.sum_monomials([(0, DIM)], [1.0], gamma)

    def test_mismatched_lengths_are_refused(self) -> None:
        gamma = self.transition_covariance(seed=4)
        with pytest.raises(ValueError, match="index-aligned"):
            WickReduction.sum_monomials([(0, 1), (2, 3)], [1.0], gamma)

    @pytest.mark.parametrize("batch_shape", [(), (4,), (3, 3), (2, 2, 5)])
    def test_the_size_grouped_sum_agrees_with_a_naive_loop(self, batch_shape: tuple) -> None:
        """The size-grouped stacking builds ``(..., n_terms, w, w)`` submatrices by advanced
        indexing, which is the one place a silent transposition or mis-broadcast would hide and
        still produce plausible numbers."""
        gamma = self.transition_covariance(batch_shape=batch_shape, seed=5)
        supports = self.random_supports(12, seed=5)
        rng = np.random.default_rng(5)
        coefficients = (rng.normal(size=12) + 1j * rng.normal(size=12)).tolist()
        batched = WickReduction.sum_monomials(supports, coefficients, gamma)
        naive = self.naive_sum(supports, coefficients, gamma)
        assert tuple(batched.shape) == batch_shape
        np.testing.assert_allclose(batched.numpy(), naive.numpy(), atol=ATOL_MATRIX_COMPARISON, rtol=0)

    def test_every_even_support_size_is_covered(self) -> None:
        """Sweeps one support of each even size, so no size group is exercised only incidentally."""
        gamma = self.transition_covariance(batch_shape=(2,), seed=6)
        supports = [tuple(range(size)) for size in (0, 2, 4, 6, 8)]
        coefficients = [1.0 + 0.5j] * len(supports)
        np.testing.assert_allclose(
            WickReduction.sum_monomials(supports, coefficients, gamma).numpy(),
            self.naive_sum(supports, coefficients, gamma).numpy(),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=0,
        )

    def test_duplicate_supports_are_summed_not_merged(self) -> None:
        gamma = self.transition_covariance(seed=7)
        single = WickReduction.sum_monomials([(0, 1)], [1.0], gamma)
        doubled = WickReduction.sum_monomials([(0, 1), (0, 1)], [1.0, 1.0], gamma)
        np.testing.assert_allclose(complex(doubled), 2 * complex(single), atol=ATOL_SCALAR_COMPARISON, rtol=0)

    @pytest.mark.parametrize("chunk_size", [1, 2, 3, 100])
    def test_chunking_does_not_change_the_result(self, chunk_size: int) -> None:
        gamma = self.transition_covariance(batch_shape=(3, 3), seed=8)
        supports = self.random_supports(8, seed=8)
        coefficients = [1.0 + 0.0j] * 8
        unchunked = WickReduction.sum_monomials(supports, coefficients, gamma)
        chunked = WickReduction.sum_monomials(supports, coefficients, gamma, chunk_size=chunk_size)
        np.testing.assert_allclose(chunked.numpy(), unchunked.numpy(), atol=ATOL_MATRIX_COMPARISON, rtol=0)

    def test_an_empty_term_list_is_zero(self) -> None:
        gamma = self.transition_covariance(batch_shape=(2, 2), seed=9)
        value = WickReduction.sum_monomials([], [], gamma)
        np.testing.assert_allclose(value.numpy(), np.zeros((2, 2)), atol=ATOL_SCALAR_COMPARISON, rtol=0)

    def test_the_sum_is_linear_in_the_coefficients(self) -> None:
        gamma = self.transition_covariance(batch_shape=(3,), seed=10)
        supports = self.random_supports(6, seed=10)
        left = [1.0 + 0.0j] * 6
        right = [0.0 + 2.0j] * 6
        combined = [a + b for a, b in zip(left, right)]
        np.testing.assert_allclose(
            WickReduction.sum_monomials(supports, combined, gamma).numpy(),
            (
                WickReduction.sum_monomials(supports, left, gamma) + WickReduction.sum_monomials(supports, right, gamma)
            ).numpy(),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=0,
        )

    @pytest.mark.parametrize("entry_point", ["element", "sum_monomials"])
    @pytest.mark.parametrize("size", [0, 2, 4, 6])
    def test_the_gradient_is_correct_at_every_even_size(self, size: int, entry_point: str) -> None:
        """``gradcheck`` against finite differences, rather than only asserting the gradient is
        finite and nonzero. The branch-weight update runs through this sum, so a subtly wrong
        gradient here is worse than a missing one: it trains, it just trains on the wrong slope.

        The parameterization varies the strict upper triangle's real and imaginary parts
        independently, which keeps every perturbation on the complex skew-symmetric manifold.
        """
        upper_count = DIM * (DIM - 1) // 2
        theta = torch.randn(
            2, upper_count, dtype=torch.float64, generator=torch.Generator().manual_seed(size)
        ).requires_grad_()
        rows, cols = torch.triu_indices(DIM, DIM, offset=1)
        support = tuple(range(size))

        def reduction(flat):
            upper = torch.zeros(DIM, DIM, dtype=torch.complex128)
            upper = upper.clone()
            upper[rows, cols] = flat[0] + 1j * flat[1]
            gamma = upper - upper.transpose(-1, -2)
            # Both entry points are swept: sum_monomials does not delegate to element, so testing
            # only the former leaves element's gradient path entirely unpinned.
            if entry_point == "element":
                return WickReduction.element(support, gamma)
            return WickReduction.sum_monomials([support], [1.0 + 0.5j], gamma)

        assert torch.autograd.gradcheck(reduction, (theta,), atol=ATOL_APPROX_COMPARISON, rtol=RTOL_APPROX_COMPARISON)

    def test_the_gradient_flows_through_every_term_of_a_shared_sum(self) -> None:
        """Several same-size terms share one parameter tensor, so a bug that reused term zero's
        submatrix for every slot would still give a finite, nonzero, plausible gradient."""
        upper_count = DIM * (DIM - 1) // 2
        theta = torch.randn(
            2, upper_count, dtype=torch.float64, generator=torch.Generator().manual_seed(77)
        ).requires_grad_()
        rows, cols = torch.triu_indices(DIM, DIM, offset=1)
        supports = [(0, 1, 2, 3), (0, 1, 4, 5), (2, 3, 4, 5), (1, 2, 5, 6)]

        def reduction(flat, terms):
            upper = torch.zeros(DIM, DIM, dtype=torch.complex128)
            upper = upper.clone()
            upper[rows, cols] = flat[0] + 1j * flat[1]
            gamma = upper - upper.transpose(-1, -2)
            return WickReduction.sum_monomials(terms, [1.0] * len(terms), gamma)

        assert torch.autograd.gradcheck(
            lambda flat: reduction(flat, supports),
            (theta,),
            atol=ATOL_APPROX_COMPARISON,
            rtol=RTOL_APPROX_COMPARISON,
        )
        many = reduction(theta, supports).abs()
        many.backward()
        gradient_many = theta.grad.clone()
        theta.grad = None
        reduction(theta, supports[:1]).abs().backward()
        assert not torch.allclose(gradient_many, theta.grad), "the sum's gradient ignores all terms but the first"

    def test_chunk_size_actually_bounds_the_batched_pfaffian_call(self) -> None:
        """Value-equality alone cannot tell correct chunking from ``chunk_size`` being dropped on
        the floor: a mutant that never forwards it passes every other test in this file. This spies
        on the kernel so the forwarding itself is pinned."""
        gamma = self.transition_covariance(batch_shape=(3, 3), seed=8)
        supports = [support for support in self.random_supports(20, seed=8) if len(support) == 4]
        assert len(supports) > 3, "need enough same-size terms for chunking to split the group"
        batch_sizes = []
        original_kernel = pfaffian_module._pfaffian_kernel

        def spy(matrix_t, sign, epsilon):
            batch_sizes.append(matrix_t.reshape(-1, matrix_t.shape[-2], matrix_t.shape[-1]).shape[0])
            return original_kernel(matrix_t, sign, epsilon)

        with mock.patch.object(pfaffian_module, "_pfaffian_kernel", side_effect=spy):
            WickReduction.sum_monomials(supports, [1.0] * len(supports), gamma, chunk_size=2)
        assert batch_sizes, "no Pfaffian kernel call was recorded"
        assert max(batch_sizes) <= 2, f"chunk_size was not honoured: kernel saw batches {batch_sizes}"

    def test_an_unsorted_support_stays_antisymmetric(self) -> None:
        """The docstring asks for sorted supports, but the value is correctly antisymmetric under
        permutation and callers may rely on that. Pinned against the dense oracle, which multiplies
        the Majoranas in exactly the order given, so this is physical agreement and not merely
        internal self-consistency."""
        vector_a, cov_a = self.gaussian_state(seed=5)
        vector_b, cov_b = self.gaussian_state(seed=6)
        overlap = complex(vector_a.conj() @ vector_b)
        gamma = torch.as_tensor(qml.math.toarray(transition_cov(cov_a, cov_b)), dtype=torch.complex128)
        for support in [(4, 1), (3, 1, 5, 0), (5, 4, 1, 0)]:
            exact = complex(vector_a.conj() @ (self.dense_majorana_monomial(support) @ vector_b)) / overlap
            np.testing.assert_allclose(
                complex(WickReduction.element(support, gamma)),
                exact,
                atol=ATOL_SCALAR_COMPARISON,
                rtol=RTOL_SCALAR_COMPARISON,
                err_msg=f"support={support}",
            )

    def test_a_repeated_mode_in_one_support_is_the_callers_responsibility(self) -> None:
        """Documents a precondition, not a result. ``c_mu^2 = I``, so ``(1, 1, 2, 3)`` is physically
        ``(2, 3)``, but the submatrix has two identical rows and the Pfaffian is exactly zero. The
        module guards negative indices explicitly and does not guard this one; reducing repeats
        before calling is the caller's job. If that ever changes, this test should change with it.
        """
        gamma = self.transition_covariance(seed=18)
        np.testing.assert_allclose(
            complex(WickReduction.element((1, 1, 2, 3), gamma)), 0.0, atol=ATOL_SCALAR_COMPARISON
        )
        assert abs(complex(WickReduction.element((2, 3), gamma))) > MIN_NONZERO_VALUE

    def test_the_reduction_is_differentiable(self) -> None:
        """The branching weight update runs through this sum, so a detached path here would silently
        drop ``dW / d theta`` from every branch-path gradient."""
        generator = torch.Generator().manual_seed(11)
        real_part = torch.randn(DIM, DIM, dtype=torch.float64, generator=generator)
        imaginary_part = torch.randn(DIM, DIM, dtype=torch.float64, generator=generator)
        angle = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
        # Genuinely complex: a real gamma would make every Pfaffian real, the uniform i^{-w/2} phases
        # would then make the whole sum imaginary, and its real part would be identically zero.
        gamma = (angle * (real_part - real_part.T) + 1j * torch.sin(angle) * (imaginary_part - imaginary_part.T)).to(
            torch.complex128
        )
        value = WickReduction.sum_monomials([(0, 1), (2, 3, 4, 5)], [1.0, 0.5j], gamma)
        value.abs().backward()
        assert angle.grad is not None
        assert torch.isfinite(angle.grad)
        assert float(abs(angle.grad)) > 0.0

    def test_supports_may_be_any_sorted_subset(self) -> None:
        """Non-contiguous supports are the normal case once a Pauli word straddles distant qubits."""
        gamma = self.transition_covariance(seed=12)
        supports = [tuple(combination) for combination in itertools.combinations(range(DIM), 4)][:10]
        coefficients = [1.0 + 0.0j] * len(supports)
        np.testing.assert_allclose(
            WickReduction.sum_monomials(supports, coefficients, gamma).numpy(),
            self.naive_sum(supports, coefficients, gamma).numpy(),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=0,
        )

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    def test_the_reduction_preserves_the_working_precision(self, dtype: torch.dtype) -> None:
        """The engine chooses its working precision once and every reduction has to honour it, or a
        complex64 run silently pays complex128 memory."""
        gamma = self.transition_covariance(batch_shape=(2,), seed=15).to(dtype)
        supports = [(), (0, 1), (2, 3, 4, 5)]
        coefficients = [1.0, 0.5j, -2.0]
        assert WickReduction.element((0, 1), gamma).dtype == dtype
        assert WickReduction.sum_monomials(supports, coefficients, gamma).dtype == dtype

    def test_the_reduction_does_not_mutate_the_transition_covariance(self) -> None:
        gamma = self.transition_covariance(batch_shape=(2,), seed=16)
        before = gamma.clone()
        WickReduction.sum_monomials(self.random_supports(6, seed=16), [1.0 + 1j] * 6, gamma)
        np.testing.assert_array_equal(gamma.numpy(), before.numpy())

    @pytest.mark.parametrize("batch_shape", [(), (3,), (2, 2)])
    def test_the_result_shape_follows_the_leading_axes(self, batch_shape: tuple) -> None:
        """Passing the full ``(chi, chi, ..., D, D)`` branch-pair grid is the intended calling
        convention, so the leading axes must survive both entry points untouched."""
        gamma = self.transition_covariance(batch_shape=batch_shape, seed=17)
        assert tuple(WickReduction.element((0, 1, 2, 3), gamma).shape) == batch_shape
        assert tuple(WickReduction.sum_monomials([(0, 1)], [1.0], gamma).shape) == batch_shape
