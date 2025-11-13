import numpy as np
import pytest
import torch

from matchcake.operations import SingleParticleTransitionMatrixOperation
from matchcake.operations.matchgate_operation import MatchgateOperation
from matchcake.utils import PAULI_X, PAULI_Z, torch_utils


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

    @pytest.mark.parametrize(
        "polar_params,std_params",
        [
            (
                    dict(r0=1, r1=1, theta0=0, theta1=0, theta2=0, theta3=0),
                    dict(a=1, b=0, c=0, d=1, w=1, x=0, y=0, z=1),
            ),
            (
                    dict(
                        r0=1,
                        r1=0,
                        theta0=0,
                        theta1=0,
                        theta2=np.pi / 2,
                        theta3=0,
                        theta4=np.pi / 2,
                    ),
                    dict(
                        a=PAULI_Z[0, 0],
                        b=PAULI_Z[0, 1],
                        c=PAULI_Z[1, 0],
                        d=PAULI_Z[1, 1],
                        w=PAULI_X[0, 0],
                        x=PAULI_X[0, 1],
                        y=PAULI_X[1, 0],
                        z=PAULI_X[1, 1],
                    ),  # fSWAP
            ),
            (
                    dict(
                        r0=1,
                        r1=1,
                        theta0=0.5 * np.pi,
                        theta1=0.5 * np.pi,
                        theta2=0.5 * np.pi,
                        theta3=0.5 * np.pi,
                        theta4=0.5 * np.pi,
                    ),
                    dict(a=0.2079, b=0, c=0, d=0.2079, w=0.2079, x=0, y=0, z=0.2079),
            ),
        ],
    )
    def test_from_polar_params(self, polar_params, std_params):
        mgo = MatchgateOperation.from_polar_params(**polar_params, wires=[0, 1])
        for k, v in std_params.items():
            assert (
                torch.allclose(getattr(mgo, k), torch.tensor(v, dtype=getattr(mgo, k).dtype)),
                f"Convertion from polar to std doesnt work. For {k=}, Got: {getattr(mgo, k)}, expected: {v}."
            )

    def test_grads(self, comp_hh01):
        assert torch.autograd.gradcheck(
            lambda x: torch.sum(MatchgateOperation(x, wires=[0, 1]).matrix()),
            torch_utils.to_tensor(comp_hh01.matrix(), torch.double).requires_grad_(),
            raise_exception=True,
        )

    def test_random_parameters(self):
        rn_params = MatchgateOperation.random_params(batch_size=None, seed=0)
        assert rn_params.shape == (7, )
        rn_params = MatchgateOperation.random_params(batch_size=3, seed=0)
        assert rn_params.shape == (3, 7)

    @pytest.mark.parametrize(
        "batch_size, seed",
        [
            (b, s)
            for b in [None, 3]
            for s in range(3)
        ]
    )
    def test_random(self, batch_size, seed):
        mgo = MatchgateOperation.random(batch_size=batch_size, seed=seed, wires=[0, 1])
        if batch_size is None:
            assert mgo.matrix().shape == (4, 4)
        else:
            assert mgo.matrix().shape == (batch_size, 4, 4)

    def test_get_padded_single_transition_particle_matrix(self, comp_hh01):
        all_wires = [0, 1, 2, 3]
        pred_padded_matrix = comp_hh01.get_padded_single_particle_transition_matrix(wires=all_wires)
        assert isinstance(pred_padded_matrix, SingleParticleTransitionMatrixOperation)
        assert pred_padded_matrix.wires.tolist() == all_wires
