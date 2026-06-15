import pytest

from matchcake.devices.star_state_finding_strategies import (
    GreedyStrategy,
    get_star_state_finding_strategy,
)


class TestGetStarStateFindingStrategy:
    def test_get_greedy_strategy_by_name(self):
        strategy = get_star_state_finding_strategy("greedy")
        assert isinstance(strategy, GreedyStrategy)

    def test_get_strategy_from_instance(self):
        instance = GreedyStrategy()
        result = get_star_state_finding_strategy(instance)
        assert result is instance

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown star state finding strategy"):
            get_star_state_finding_strategy("unknown_strategy_xyz")
