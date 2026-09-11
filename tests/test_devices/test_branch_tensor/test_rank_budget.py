import numpy as np
import pytest
import torch

from matchcake.devices.branch_tensor.rank_budget import EXACT_DEAD_CUT_TOL, RankBudget

from ...configs import ATOL_SCALAR_COMPARISON, RTOL_SCALAR_COMPARISON

# A population far below any tolerance used here, standing in for a branch that is algebraically dead
# and carries only accumulated round-off.
DEAD_POPULATION = 1e-18

# A working tolerance well above DEAD_POPULATION and well below the O(1) live populations, so the two
# are never separated by an accident of the tolerance choice.
LIVE_WEIGHT_TOL = 1e-12


class TestRankBudget:
    @staticmethod
    def weights_with_populations(populations) -> torch.Tensor:
        """Diagonal weight matrix carrying the given per-branch populations.

        :param populations: Sequence of ``chi`` populations.
        :return: Complex ``(chi, chi)`` weight matrix.
        :rtype: torch.Tensor
        """
        return torch.diag(torch.as_tensor(populations, dtype=torch.complex128))

    @staticmethod
    def noisy_weights(populations, seed: int = 0) -> torch.Tensor:
        """Weight matrix with the given diagonal and large random off-diagonal entries.

        The off-diagonal block is deliberately much larger than the diagonal: every quantity this
        module reports is a function of the diagonal alone, so a wrong index would show up as a wildly
        wrong population rather than as a small perturbation.

        :param populations: Sequence of ``chi`` populations.
        :param seed: Seed of the generator drawing the off-diagonal entries.
        :return: Complex ``(chi, chi)`` weight matrix.
        :rtype: torch.Tensor
        """
        generator = torch.Generator().manual_seed(seed)
        n_branches = len(populations)
        real = torch.randn(n_branches, n_branches, dtype=torch.float64, generator=generator)
        imaginary = torch.randn(n_branches, n_branches, dtype=torch.float64, generator=generator)
        weights = 100.0 * (real + 1j * imaginary).to(torch.complex128)
        index = torch.arange(n_branches)
        weights[index, index] = torch.as_tensor(populations, dtype=torch.complex128)
        return weights

    def test_branch_populations_read_the_weight_diagonal(self) -> None:
        weights = self.weights_with_populations([0.5, 2.0, 0.0])
        np.testing.assert_allclose(
            RankBudget.branch_populations(weights).numpy(),
            np.array([0.5, 2.0, 0.0]),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_branch_populations_ignore_the_off_diagonal(self) -> None:
        populations = [0.5, 2.0, 0.25]
        np.testing.assert_allclose(
            RankBudget.branch_populations(self.noisy_weights(populations)).numpy(),
            np.array(populations),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_branch_populations_take_the_modulus_of_a_complex_diagonal(self) -> None:
        """A population is ``|W_aa|``, so a diagonal off the positive real axis keeps its magnitude."""
        weights = self.weights_with_populations([1j * 0.5, -2.0, (1.0 + 1.0j) / np.sqrt(2.0)])
        np.testing.assert_allclose(
            RankBudget.branch_populations(weights).numpy(),
            np.array([0.5, 2.0, 1.0]),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_branch_populations_reduce_a_batch_by_the_maximum(self) -> None:
        weights = torch.zeros(2, 2, 3, dtype=torch.complex128)
        weights[0, 0] = torch.as_tensor([0.0, 0.0, 1.5], dtype=torch.complex128)
        weights[1, 1] = torch.as_tensor([0.2, 0.1, 0.0], dtype=torch.complex128)
        np.testing.assert_allclose(
            RankBudget.branch_populations(weights).numpy(),
            np.array([1.5, 0.2]),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_branch_populations_reduce_several_batch_axes_by_the_maximum(self) -> None:
        weights = torch.zeros(2, 2, 3, 4, dtype=torch.complex128)
        weights[0, 0, 2, 1] = 1.5
        weights[1, 1, 0, 3] = 0.2
        np.testing.assert_allclose(
            RankBudget.branch_populations(weights).numpy(),
            np.array([1.5, 0.2]),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_a_branch_alive_in_a_single_batch_element_survives(self) -> None:
        """The maximum reduction is the point: one batch element needing a branch keeps it alive."""
        weights = torch.zeros(2, 2, 4, dtype=torch.complex128)
        weights[0, 0] = 1.0
        weights[1, 1] = torch.as_tensor([DEAD_POPULATION, DEAD_POPULATION, 0.7, DEAD_POPULATION])
        assert RankBudget(weight_tol=LIVE_WEIGHT_TOL).select(weights) == [0, 1]

    def test_vanished_branches_are_dropped(self) -> None:
        budget = RankBudget(weight_tol=1e-9)
        assert budget.select(self.weights_with_populations([1.0, 1e-15, 0.5])) == [0, 2]

    def test_a_population_exactly_at_the_tolerance_survives(self) -> None:
        """The cut is ``>= weight_tol``, so the boundary belongs to the live side."""
        budget = RankBudget(weight_tol=LIVE_WEIGHT_TOL)
        weights = self.weights_with_populations([1.0, LIVE_WEIGHT_TOL, LIVE_WEIGHT_TOL / 2.0])
        assert budget.alive_branches(weights) == [0, 1]

    def test_a_zero_tolerance_keeps_even_an_exactly_zero_population(self) -> None:
        budget = RankBudget(weight_tol=0.0)
        assert budget.alive_branches(self.weights_with_populations([1.0, 0.0])) == [0, 1]

    def test_chi_budget_keeps_the_strongest_branches_in_index_order(self) -> None:
        budget = RankBudget(weight_tol=0.0, chi_th=2)
        assert budget.select(self.weights_with_populations([0.1, 0.9, 0.5, 0.2])) == [1, 2]

    def test_chi_budget_keeps_exactly_chi_th_branches_when_populations_tie(self) -> None:
        budget = RankBudget(weight_tol=0.0, chi_th=2)
        selected = budget.select(self.weights_with_populations([1.0, 1.0, 1.0, 1.0]))
        assert len(selected) == 2
        assert selected == sorted(selected)

    def test_a_budget_wider_than_the_state_changes_nothing(self) -> None:
        weights = self.weights_with_populations([1.0, 0.5, 0.25])
        assert RankBudget(weight_tol=0.0, chi_th=10).select(weights) == [0, 1, 2]

    def test_at_least_one_branch_always_survives(self) -> None:
        budget = RankBudget(weight_tol=1.0)
        assert budget.select(self.weights_with_populations([1e-20, 1e-19])) == [1]

    def test_the_surviving_branch_of_a_dead_state_is_the_strongest_one(self) -> None:
        """The fallback is ``argmax``, not index zero: a rounding accident must not pick arbitrarily."""
        budget = RankBudget(weight_tol=1.0)
        assert budget.alive_branches(self.weights_with_populations([1e-20, 1e-16, 1e-18])) == [1]

    def test_the_exact_cut_never_empties_the_state(self) -> None:
        weights = self.weights_with_populations([DEAD_POPULATION, 1e-20])
        assert RankBudget(weight_tol=LIVE_WEIGHT_TOL).alive_branches(weights) == [0]

    def test_the_exact_cut_and_the_budget_cut_are_separable(self) -> None:
        """The branch count reported as pruned counts only the budget, so the cuts must read apart."""
        weights = self.weights_with_populations([1.0, DEAD_POPULATION, 0.5, 0.25])
        budget = RankBudget(weight_tol=LIVE_WEIGHT_TOL, chi_th=2)
        assert budget.alive_branches(weights) == [0, 2, 3]
        assert budget.select(weights) == [0, 2]

    def test_without_a_budget_the_two_cuts_agree(self) -> None:
        weights = self.weights_with_populations([1.0, DEAD_POPULATION, 0.5])
        budget = RankBudget(weight_tol=LIVE_WEIGHT_TOL)
        assert budget.alive_branches(weights) == budget.select(weights) == [0, 2]

    @pytest.mark.parametrize("chi_th", [None, 1, 2, 3, 5])
    @pytest.mark.parametrize("seed", list(range(4)))
    def test_the_selection_is_always_a_sorted_subset_of_the_live_branches(self, chi_th, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        populations = torch.rand(5, dtype=torch.float64, generator=generator) ** 8
        budget = RankBudget(weight_tol=1e-3, chi_th=chi_th)
        alive = budget.alive_branches(self.weights_with_populations(populations))
        selected = budget.select(self.weights_with_populations(populations))
        assert selected == sorted(selected)
        assert set(selected).issubset(set(alive))
        if chi_th is None:
            assert selected == alive
        else:
            assert len(selected) == min(len(alive), chi_th)

    def test_the_kept_branches_are_the_most_populated_ones(self) -> None:
        """No dropped live branch may be more populated than a kept one."""
        populations = [0.31, 0.87, 0.02, 0.64, 0.45]
        weights = self.weights_with_populations(populations)
        budget = RankBudget(weight_tol=0.0, chi_th=3)
        selected = budget.select(weights)
        dropped = [index for index in range(len(populations)) if index not in selected]
        assert max(populations[index] for index in dropped) <= min(populations[index] for index in selected)

    def test_populations_do_not_leak_the_autograd_graph(self) -> None:
        populations = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
        weights = torch.diag(populations).to(torch.complex128)
        assert not RankBudget.branch_populations(weights).requires_grad

    def test_a_grad_tracking_weight_matrix_can_be_selected_on(self) -> None:
        """Pruning is a discrete decision taken under a live graph, so it must not raise."""
        populations = torch.tensor([1.0, 1e-20, 0.5], dtype=torch.float64, requires_grad=True)
        weights = torch.diag(populations).to(torch.complex128)
        assert RankBudget(weight_tol=LIVE_WEIGHT_TOL, chi_th=2).select(weights) == [0, 2]

    def test_numpy_weights_are_accepted(self) -> None:
        weights = np.diag(np.array([1.0, DEAD_POPULATION, 0.5], dtype=complex))
        budget = RankBudget(weight_tol=LIVE_WEIGHT_TOL)
        np.testing.assert_allclose(
            RankBudget.branch_populations(weights).numpy(),
            np.array([1.0, DEAD_POPULATION, 0.5]),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )
        assert budget.select(weights) == [0, 2]

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.complex64, torch.complex128])
    def test_every_weight_dtype_is_accepted(self, dtype: torch.dtype) -> None:
        weights = torch.diag(torch.as_tensor([1.0, 0.25], dtype=dtype))
        np.testing.assert_allclose(
            RankBudget.branch_populations(weights).to(torch.float64).numpy(),
            np.array([1.0, 0.25]),
            atol=ATOL_SCALAR_COMPARISON,
            rtol=RTOL_SCALAR_COMPARISON,
        )

    def test_a_single_branch_state_survives_every_budget(self) -> None:
        weights = self.weights_with_populations([0.5])
        assert RankBudget(weight_tol=0.0, chi_th=1).select(weights) == [0]

    @pytest.mark.parametrize("shape", [(0, 0), (2, 2, 0), (3, 3, 2, 0)])
    def test_an_empty_weight_matrix_is_rejected(self, shape: tuple) -> None:
        """Reducing over an empty axis has no answer, and ``reshape(0, -1)`` is ambiguous."""
        weights = torch.zeros(*shape, dtype=torch.complex128)
        with pytest.raises(ValueError, match="no entries"):
            RankBudget.branch_populations(weights)
        with pytest.raises(ValueError, match="no entries"):
            RankBudget().select(weights)

    @pytest.mark.parametrize("shape", [(2, 3), (4,), (3, 2, 5)])
    def test_a_weight_matrix_that_is_not_square_in_its_branch_axes_is_rejected(self, shape: tuple) -> None:
        weights = torch.ones(*shape, dtype=torch.complex128)
        with pytest.raises(ValueError, match=r"shape \(chi, chi, \.\.\.\)"):
            RankBudget.branch_populations(weights)

    def test_rejects_invalid_settings(self) -> None:
        with pytest.raises(ValueError, match="weight_tol must be non-negative"):
            RankBudget(weight_tol=-1.0)
        with pytest.raises(ValueError, match="chi_th must be at least 1"):
            RankBudget(chi_th=0)

    def test_a_zero_tolerance_and_no_budget_are_both_accepted(self) -> None:
        budget = RankBudget(weight_tol=0.0, chi_th=None)
        assert budget.weight_tol == 0.0
        assert budget.chi_th is None

    def test_the_settings_are_stored_with_their_declared_types(self) -> None:
        budget = RankBudget(weight_tol=1, chi_th=3.0)
        assert isinstance(budget.weight_tol, float)
        assert isinstance(budget.chi_th, int)

    def test_repr_reports_both_settings(self) -> None:
        representation = repr(RankBudget(weight_tol=1e-9, chi_th=3))
        assert "RankBudget" in representation
        assert "chi_th=3" in representation
        assert "weight_tol=1e-09" in representation

    def test_the_exact_dead_cut_tolerance_is_a_round_off_level_bound(self) -> None:
        """``EXACT_DEAD_CUT_TOL`` certifies a cut as exact rather than deciding which cuts happen.

        It has to stay at the scale a genuinely dead population reaches through accumulated
        round-off, a small multiple of the double-precision epsilon, and well below the default
        ``weight_tol`` that actually performs the cut.
        """
        assert np.finfo(np.float64).eps < EXACT_DEAD_CUT_TOL < RankBudget().weight_tol
