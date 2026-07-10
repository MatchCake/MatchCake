import numpy as np
import pennylane as qml
import pytest

from matchcake import MatchgateOperation
from matchcake.circuits.random_generator import RandomOperationsGenerator

from ..configs import TEST_SEED, set_seed


class TestRandomOperationsGenerator:
    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

    def test_init_with_int_wires(self):
        gen = RandomOperationsGenerator(wires=4, op_types=[MatchgateOperation])
        assert len(gen.wires) == 4

    def test_n_ops_default(self):
        gen = RandomOperationsGenerator(wires=4, op_types=[MatchgateOperation])
        assert gen.n_ops == 2 * 4 * 1

    def test_iter_yields_ops(self):
        gen = RandomOperationsGenerator(wires=4, n_ops=2, op_types=[MatchgateOperation], seed=TEST_SEED)
        ops = list(gen)
        assert len(ops) == 3

    def test_iter_zero_ops(self):
        gen = RandomOperationsGenerator(wires=4, n_ops=0, op_types=[MatchgateOperation])
        ops = list(gen)
        assert len(ops) == 0

    def test_get_ops(self):
        gen = RandomOperationsGenerator(wires=4, n_ops=2, op_types=[MatchgateOperation], seed=TEST_SEED)
        ops = gen.get_ops()
        assert isinstance(ops, list)
        assert len(ops) == 3

    def test_get_output_op_none(self):
        gen = RandomOperationsGenerator(wires=4, n_ops=1, op_types=[MatchgateOperation], output_type=None)
        result = gen.get_output_op()
        assert result is None

    def test_get_output_op_probs(self):
        gen = RandomOperationsGenerator(wires=4, n_ops=1, op_types=[MatchgateOperation], output_type="probs")
        result = gen.get_output_op()
        assert result is not None

    def test_get_output_op_expval(self):
        gen = RandomOperationsGenerator(
            wires=4, n_ops=1, op_types=[MatchgateOperation], output_type="expval", observable=qml.Z(0)
        )
        result = gen.get_output_op()
        assert result is not None

    def test_get_output_op_samples(self):
        gen = RandomOperationsGenerator(wires=4, n_ops=1, op_types=[MatchgateOperation], output_type="samples")
        result = gen.get_output_op()
        assert result is not None

    def test_get_output_op_invalid_raises(self):
        gen = RandomOperationsGenerator(wires=4, n_ops=1, op_types=[MatchgateOperation], output_type="invalid")
        with pytest.raises(ValueError, match="Invalid output_type"):
            gen.get_output_op()

    def test_get_initial_state_fixed(self):
        gen = RandomOperationsGenerator(
            wires=4,
            n_ops=1,
            op_types=[MatchgateOperation],
            initial_state=[0, 1, 0, 1],
        )
        rng = np.random.default_rng(TEST_SEED)
        state = gen.get_initial_state(rng)
        np.testing.assert_array_equal(state, [0, 1, 0, 1])

    def test_get_initial_state_wrong_length_raises(self):
        gen = RandomOperationsGenerator(
            wires=4,
            n_ops=1,
            op_types=[MatchgateOperation],
            initial_state=[0, 1],
        )
        rng = np.random.default_rng(TEST_SEED)
        with pytest.raises(ValueError, match="Initial state has"):
            gen.get_initial_state(rng)

    def test_repr(self):
        gen = RandomOperationsGenerator(wires=2, n_ops=1, op_types=[MatchgateOperation])
        r = repr(gen)
        assert "RandomOperationsGenerator" in r

    def test_n_qubits_property(self):
        gen = RandomOperationsGenerator(wires=4, op_types=[MatchgateOperation])
        assert gen.n_qubits == 4

    def test_n_wires_property(self):
        gen = RandomOperationsGenerator(wires=4, op_types=[MatchgateOperation])
        assert gen.n_wires == 4

    def test_output_kwargs_property(self):
        gen = RandomOperationsGenerator(wires=4, op_types=[MatchgateOperation], output_type="probs")
        kwargs = gen.output_kwargs
        assert kwargs["output_type"] == "probs"
