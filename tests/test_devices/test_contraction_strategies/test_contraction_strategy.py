import numpy as np
import pytest
import tqdm

from matchcake.devices.contraction_strategies.forward_strategy import ForwardContractionStrategy
from matchcake.operations import SptmCompRxRx

from ...configs import TEST_SEED, set_seed


class TestContractionStrategy:
    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

    def _make_ops(self, n: int = 3):
        return [SptmCompRxRx(np.random.random(2), wires=[0, 1]) for _ in range(n)]

    def test_call_with_multiple_operations(self):
        strategy = ForwardContractionStrategy()
        ops = self._make_ops(3)
        result = strategy(ops)
        assert len(result) > 0

    def test_call_with_single_operation_returns_early(self):
        strategy = ForwardContractionStrategy()
        ops = self._make_ops(1)
        result = strategy(ops)
        assert result is ops

    def test_call_with_show_progress_true(self):
        strategy = ForwardContractionStrategy()
        ops = self._make_ops(3)
        result = strategy(ops, show_progress=True)
        assert len(result) > 0

    def test_call_with_external_p_bar(self):
        strategy = ForwardContractionStrategy()
        ops = self._make_ops(3)
        p_bar = tqdm.tqdm(total=len(ops), disable=True)
        result = strategy(ops, p_bar=p_bar)
        p_bar.close()
        assert len(result) > 0

    def test_p_bar_set_n_without_p_bar(self):
        strategy = ForwardContractionStrategy()
        strategy.p_bar_set_n(5)

    def test_p_bar_set_n_with_p_bar(self):
        strategy = ForwardContractionStrategy()
        strategy.p_bar = tqdm.tqdm(total=10, disable=True)
        strategy.p_bar_set_n(5)
        assert strategy.p_bar.n == 5
        strategy.p_bar.close()

    def test_p_bar_set_n_p1(self):
        strategy = ForwardContractionStrategy()
        strategy.p_bar = tqdm.tqdm(total=10, disable=True)
        strategy.p_bar_set_n_p1(4)
        assert strategy.p_bar.n == 5
        strategy.p_bar.close()

    def test_initialize_p_bar_creates_bar_when_show_progress(self):
        strategy = ForwardContractionStrategy(show_progress=True)
        bar = strategy.initialize_p_bar(total=5, disable=True)
        assert bar is not None
        bar.close()

    def test_close_p_bar_without_p_bar(self):
        strategy = ForwardContractionStrategy()
        strategy.close_p_bar()

    def test_close_p_bar_with_p_bar(self):
        strategy = ForwardContractionStrategy()
        strategy.p_bar = tqdm.tqdm(total=10, disable=True)
        strategy.close_p_bar()
