import numpy as np
import pytest

from matchcake import MatchgateOperation
from matchcake.devices.device_utils import circuit_or_fop_matmul
from matchcake.operations import SptmCompRxRx
from matchcake.operations.single_particle_transition_matrices import SingleParticleTransitionMatrixOperation

from ..configs import TEST_SEED, set_seed


class TestCircuitOrFopMatmul:
    @classmethod
    def setup_class(cls):
        set_seed(TEST_SEED)

    def test_matchgate_times_matchgate(self):
        op1 = MatchgateOperation.random(wires=[0, 1])
        op2 = MatchgateOperation.random(wires=[0, 1])
        result = circuit_or_fop_matmul(op1, op2)
        assert isinstance(result, (MatchgateOperation, SingleParticleTransitionMatrixOperation))

    def test_sptm_times_sptm(self):
        op1 = SptmCompRxRx(np.random.random(2), wires=[0, 1])
        op2 = SptmCompRxRx(np.random.random(2), wires=[0, 1])
        result = circuit_or_fop_matmul(op1, op2)
        assert isinstance(result, SingleParticleTransitionMatrixOperation)

    def test_matchgate_times_sptm(self):
        op1 = MatchgateOperation.random(wires=[0, 1])
        op2 = SptmCompRxRx(np.random.random(2), wires=[0, 1])
        result = circuit_or_fop_matmul(op1, op2)
        assert isinstance(result, SingleParticleTransitionMatrixOperation)

    def test_sptm_times_matchgate(self):
        op1 = SptmCompRxRx(np.random.random(2), wires=[0, 1])
        op2 = MatchgateOperation.random(wires=[0, 1])
        result = circuit_or_fop_matmul(op1, op2)
        assert isinstance(result, SingleParticleTransitionMatrixOperation)

    def test_invalid_second_type_raises_value_error(self):
        op1 = MatchgateOperation.random(wires=[0, 1])
        with pytest.raises(ValueError):
            circuit_or_fop_matmul(op1, "not_an_operation")

    def test_invalid_first_type_raises_value_error(self):
        op2 = MatchgateOperation.random(wires=[0, 1])
        with pytest.raises(ValueError):
            circuit_or_fop_matmul("not_an_operation", op2)
