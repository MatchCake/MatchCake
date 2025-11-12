import numpy as np
import pytest
import torch

from matchcake.operations import SingleParticleTransitionMatrixOperation
from matchcake.operations.matchgate_operation import MatchgateOperation
from matchcake.utils import PAULI_X, PAULI_Z


class TestMatchgateOperation:
    @pytest.fixture
    def fswap01(self):
        return MatchgateOperation.from_sub_matrices(PAULI_Z, PAULI_X, wires=[0, 1])

    @pytest.fixture
    def comp_hh01(self):
        _inv_sqrt_2 = 1 / np.sqrt(2)
        return MatchgateOperation.from_std_params(
            a=_inv_sqrt_2,
            b=_inv_sqrt_2,
            c=_inv_sqrt_2,
            d=-_inv_sqrt_2,
            w=_inv_sqrt_2,
            x=_inv_sqrt_2,
            y=_inv_sqrt_2,
            z=-_inv_sqrt_2,
            wires=[0, 1],
        )

    @pytest.mark.parametrize(
        "input_matrix",
        [
            np.array(
                [
                    [1.0, 0, 0, 0.0],
                    [0, 1.0, 0.0, 0],
                    [0, 0.0, 1.0, 0],
                    [0.0, 0, 0, 1.0],
                ]
            ),
            np.array(
                [
                    [1.0, 0, 0, 0.0],
                    [0, 0.0, 1.0, 0],
                    [0, -1.0, 0.0, 0],
                    [0.0, 0, 0, 1.0],
                ]
            ),
            np.array(
                [
                    [1.0, 0, 0, 0.0],
                    [0, 0.0, 1.0, 0],
                    [0, 1.0, 0.0, 0],
                    [0.0, 0, 0, -1.0],
                ]
            ),
        ],
    )
    def test_init(self, input_matrix):
        mgo = MatchgateOperation(input_matrix, wires=[0, 1])
        assert mgo.shape == input_matrix.shape
        assert isinstance(mgo.matrix(), torch.Tensor)

    @pytest.mark.parametrize(
        "input_matrix",
        [
            np.array(
                [
                    [10.0, 0, 0, 0.0],
                    [0, 1.0, 0.0, 0],
                    [0, 0.0, 1.0, 0],
                    [0.0, 0, 0, 1.0],
                ]
            ),
            np.array(
                [
                    [1.0, 0, 0, 0.0],
                    [0, 0.0, 1.0, 0],
                    [1.0, -1.0, 0.0, 0],
                    [0.0, 0, 0, 1.0],
                ]
            ),
            np.array(
                [
                    [1.0, 0, 0, 0.0],
                    [0, 0.0, 1.0, 0],
                    [0, 1.0, 0.0, 0],
                    [0.0, 0, 0, 1.0],
                ]
            ),
        ],
    )
    def test_init_from_not_matchgate(self, input_matrix):
        with pytest.raises(ValueError):
            MatchgateOperation(input_matrix, wires=[0, 1])

    def test_matmul_mg_mg(self, fswap01, comp_hh01):
        new_op = fswap01 @ comp_hh01
        assert isinstance(new_op, MatchgateOperation)
        assert new_op.matrix().shape == fswap01.shape
        matmul_value = torch.matmul(fswap01.matrix(), comp_hh01.matrix())
        torch.testing.assert_close(new_op.matrix(), matmul_value)

    def test_matmul_mg_sptm(self, fswap01, comp_hh01):
        new_op = fswap01 @ comp_hh01.to_sptm_operation()
        assert isinstance(new_op, SingleParticleTransitionMatrixOperation)

    def test_matmul_mg_mg_not_same_wires(self, fswap01, comp_hh01):
        comp_hh12 = MatchgateOperation(comp_hh01.matrix(), wires=[1, 2])
        new_op = fswap01 @ comp_hh12
        assert isinstance(new_op, SingleParticleTransitionMatrixOperation)
        assert new_op.wires.tolist() == [0, 1, 2]

    def test_to_sptm_operation(self, comp_hh01):
        sptm = comp_hh01.to_sptm_operation()
        assert isinstance(sptm, SingleParticleTransitionMatrixOperation)
        assert sptm.wires == comp_hh01.wires

    @pytest.mark.parametrize(
        "input_matrix",
        [
            np.array(
                [
                    [1.0j, 0, 0, 0.0],
                    [0, 1.0j, 0.0, 0],
                    [0, 0.0, 1.0j, 0],
                    [0.0, 0, 0, 1.0j],
                ]
            ),
            np.array(
                [
                    [1.0j, 0, 0, 0.0],
                    [0, 0.0, 1.0j, 0],
                    [0.0, -1.0, 0.0, 0],
                    [0.0, 0, 0, 1.0],
                ]
            ),
            np.array(
                [
                    [1.0, 0, 0, 0.0],
                    [0, 0.0, 1.0, 0],
                    [0, 1.0j, 0.0, 0],
                    [0.0, 0, 0, -1.0j],
                ]
            ),
        ],
    )
    def test_adjoint(self, input_matrix):
        mgo = MatchgateOperation(input_matrix, wires=[0, 1])
        torch.testing.assert_close(mgo.adjoint().matrix(), torch.conj(torch.from_numpy(input_matrix)).transpose(-1, -2))

    def test_label(self, comp_hh01):
        assert isinstance(comp_hh01.label(), str)

    def test_sorted_wires(self, comp_hh01):
        mgo = MatchgateOperation(comp_hh01.matrix(), wires=[0, 1])
        assert mgo.sorted_wires.tolist() == [0, 1]

    def test_cs_wires(self, comp_hh01):
        mgo = MatchgateOperation(comp_hh01.matrix(), wires=[0, 1])
        assert mgo.cs_wires.tolist() == [0, 1]

    def test_from_std_params(self):
        dtype = torch.complex64
        a, b, c, d, w, x, y, z = (1, 0, 0, -1, 1, 0, None, -1)
        mgo = MatchgateOperation.from_std_params(a, b, c, d, w, x, y, z, wires=[0, 1], dtype=dtype)
        torch.testing.assert_close(mgo.a, torch.tensor(a, dtype=dtype))
        torch.testing.assert_close(mgo.b, torch.tensor(b, dtype=dtype))
        torch.testing.assert_close(mgo.c, torch.tensor(c, dtype=dtype))
        torch.testing.assert_close(mgo.d, torch.tensor(d, dtype=dtype))
        torch.testing.assert_close(mgo.w, torch.tensor(w, dtype=dtype))
        torch.testing.assert_close(mgo.x, torch.tensor(x, dtype=dtype))
        torch.testing.assert_close(mgo.y, torch.tensor(0, dtype=dtype))
        torch.testing.assert_close(mgo.z, torch.tensor(z, dtype=dtype))

    def test_from_sub_matrices(self):
        dtype = torch.complex64
        mgo = MatchgateOperation.from_sub_matrices(PAULI_Z, PAULI_X, wires=[0, 1], dtype=dtype)
        fswap = np.array(
            [
                [1.0, 0, 0, 0.0],
                [0, 0.0, 1.0, 0],
                [0, 1.0, 0.0, 0],
                [0.0, 0, 0, -1.0],
            ]
        )
        torch.testing.assert_close(mgo.matrix(), torch.from_numpy(fswap).to(dtype=dtype))

    def test_from_polar_params(self): ...
