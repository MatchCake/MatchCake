import pytest

from matchcake.constants import MatmulDirectionType


class TestMatmulDirectionType:
    def test_lr_value(self):
        assert MatmulDirectionType.LR.value == "lr"

    def test_rl_value(self):
        assert MatmulDirectionType.RL.value == "rl"

    def test_eq_enum_same(self):
        assert MatmulDirectionType.LR == MatmulDirectionType.LR

    def test_eq_string_lr(self):
        assert MatmulDirectionType.LR == "lr"

    def test_eq_string_rl(self):
        assert MatmulDirectionType.RL == "rl"

    def test_eq_other_type_returns_false(self):
        assert not (MatmulDirectionType.LR == 42)

    def test_place_ops_lr(self):
        ops = (1, 2, 3)
        result = list(MatmulDirectionType.place_ops(MatmulDirectionType.LR, *ops))
        assert result == list(reversed(ops))

    def test_place_ops_rl(self):
        ops = (1, 2, 3)
        result = list(MatmulDirectionType.place_ops(MatmulDirectionType.RL, *ops))
        assert result == list(ops)
