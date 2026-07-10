import pytest

from matchcake.devices.sampling_strategies import QubitByQubitSampling, get_sampling_strategy


class TestGetSamplingStrategy:
    def test_get_qubit_by_qubit_by_name(self):
        strategy = get_sampling_strategy("qubitbyqubitsampling")
        assert isinstance(strategy, QubitByQubitSampling)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown sampling strategy"):
            get_sampling_strategy("unknown_strategy_xyz")
