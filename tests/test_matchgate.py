import numpy as np
import pytest
from msim import Matchgate


@pytest.fixture
def matchgate_rn_init(*args, **kwargs) -> Matchgate:
    return Matchgate.random()


def test_matchgate_m_m_dagger_constraint(matchgate_rn_init):
    assert matchgate_rn_init.check_m_m_dagger_constraint()


def test_matchgate_m_dagger_m_constraint(matchgate_rn_init):
    assert matchgate_rn_init.check_m_dagger_m_constraint()


def test_matchgate_det_constraint(matchgate_rn_init):
    assert matchgate_rn_init.check_det_constraint()


@pytest.mark.parametrize(
    "params",
    [
        tuple(np.random.rand(6))
        for _ in range(100)
    ]
)
def test_matchgate_constructor_with_default_theta4(params):
    m = Matchgate(params)
    exp4 = np.exp(1j*np.mod(m.params.theta4, 2 * np.pi))
    exp2 = np.exp(1j*np.mod(m.params.theta2, 2 * np.pi))
    assert np.isclose(exp2 * exp4, 1.0+0j), f"{exp2 * exp4 = }, expected 1.0+0j"





