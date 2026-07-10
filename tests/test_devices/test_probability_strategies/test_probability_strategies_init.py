import pytest

from matchcake.devices.probability_strategies import get_probability_strategy


class TestGetProbabilityStrategy:
    def test_lookup_table_strategy(self):
        from matchcake.devices.probability_strategies import LookupTableStrategy

        strategy = get_probability_strategy("LookupTable")
        assert isinstance(strategy, LookupTableStrategy)

    def test_case_insensitive(self):
        strategy = get_probability_strategy("lookuptable")
        assert strategy is not None

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown sampling strategy name"):
            get_probability_strategy("not_a_real_strategy")
