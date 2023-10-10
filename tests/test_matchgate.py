import pytest
from msim import Matchgate


@pytest.fixture
def matchgate_init(*args, **kwargs) -> Matchgate:
    return Matchgate.random()


def test_matchgate_m_m_dagger_constraint(matchgate_init):
    assert matchgate_init.check_m_m_dagger_constraint()


def test_matchgate_m_dagger_m_constraint(matchgate_init):
    assert matchgate_init.check_m_dagger_m_constraint()


def test_matchgate_det_constraint(matchgate_init):
    assert matchgate_init.check_det_constraint()







