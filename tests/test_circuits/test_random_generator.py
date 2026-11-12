import numpy as np
import pennylane as qml
import pytest

from matchcake import MatchgateOperation
from matchcake.circuits.random_generator import RandomOperationsGenerator, neighbouring_wire_pairs

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

    def test_iter_stays_on_neighbouring_pairs_of_wires(self):
        gen = RandomOperationsGenerator(wires=[0, 1, 2, 7, 8], n_ops=40, op_types=[MatchgateOperation], seed=TEST_SEED)
        pairs = {tuple(op.wires.tolist()) for op in list(gen)[1:]}
        assert pairs == {(0, 1), (1, 2), (7, 8)}

    def test_iter_ops_of_same_type_have_different_params(self):
        gen = RandomOperationsGenerator(wires=3, n_ops=4, op_types=[MatchgateOperation], seed=TEST_SEED)
        matrices = [qml.math.to_numpy(op.matrix()) for op in list(gen)[1:]]
        for matrix in matrices[1:]:
            assert not np.allclose(matrix, matrices[0])

    @pytest.mark.parametrize(
        "wires, expected",
        [
            ([0, 1], [[0, 1]]),
            ([2, 0, 1], [[0, 1], [1, 2]]),
            ([0, 1, 3, 4, 6], [[0, 1], [3, 4]]),
            (np.arange(4), [[0, 1], [1, 2], [2, 3]]),
        ],
    )
    def test_neighbouring_wire_pairs(self, wires, expected):
        np.testing.assert_equal(neighbouring_wire_pairs(wires), expected)

    @pytest.mark.parametrize("wires", [[0], [0, 2], [1, 3, 5]])
    def test_neighbouring_wire_pairs_without_neighbours_raises(self, wires):
        with pytest.raises(ValueError, match="neighbouring wires"):
            neighbouring_wire_pairs(wires)

    def test_iter_use_cuda_moves_every_op(self, monkeypatch):
        # ``to_cuda`` itself is GPU-only; stub it so the use_cuda=True path runs on a CPU-only machine.
        moved = []

        def fake_to_cuda(self):
            moved.append(self)
            return self

        monkeypatch.setattr(MatchgateOperation, "to_cuda", fake_to_cuda, raising=False)
        gen = RandomOperationsGenerator(wires=3, n_ops=3, op_types=[MatchgateOperation], use_cuda=True, seed=TEST_SEED)
        ops = list(gen)

        assert moved == ops[1:]

    def test_len(self):
        gen = RandomOperationsGenerator(wires=3, n_ops=5, op_types=[MatchgateOperation], seed=TEST_SEED)

        assert len(gen) == 5

    def test_circuit_runs_as_qnode_body(self):
        gen = RandomOperationsGenerator(
            wires=3, n_ops=3, op_types=[MatchgateOperation], seed=TEST_SEED, output_type="probs"
        )
        probs = qml.QNode(gen.circuit, qml.device("default.qubit", wires=3))()

        np.testing.assert_allclose(float(qml.math.sum(probs)), 1.0, atol=1e-8)

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
