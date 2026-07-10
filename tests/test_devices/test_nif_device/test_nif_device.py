import numpy as np
import pytest
import torch
from pennylane import BasisState
from pennylane.exceptions import DeviceError

from matchcake import NonInteractingFermionicDevice
from matchcake.base import NonInteractingFermionicLookupTable
from matchcake.operations import SingleParticleTransitionMatrixOperation


class TestNonInteractingFermionicDevice:
    def test_update_single_particle_transition_matrix(self):
        dev = NonInteractingFermionicDevice(wires=2)
        old_sptm = SingleParticleTransitionMatrixOperation.random(wires=[0, 1], seed=0)
        new_sptm = SingleParticleTransitionMatrixOperation.random(wires=[0, 1], seed=1)
        updated_sptm = dev.update_single_particle_transition_matrix(old_sptm, new_sptm)
        expected_sptm = old_sptm.matrix() @ new_sptm.matrix()
        np.testing.assert_allclose(updated_sptm, expected_sptm)

    def test_update_single_particle_transition_matrix_with_tensors(self):
        dev = NonInteractingFermionicDevice(wires=2)
        old_sptm = SingleParticleTransitionMatrixOperation.random(wires=[0, 1], seed=0).matrix()
        new_sptm = SingleParticleTransitionMatrixOperation.random(wires=[0, 1], seed=1).matrix()
        updated_sptm = dev.update_single_particle_transition_matrix(old_sptm, new_sptm)
        expected_sptm = old_sptm @ new_sptm
        np.testing.assert_allclose(updated_sptm, expected_sptm)

    def test_update_single_particle_transition_matrix_with_none(self):
        dev = NonInteractingFermionicDevice(wires=2)
        old_sptm = None
        new_sptm = SingleParticleTransitionMatrixOperation.random(wires=[0, 1], seed=1)
        updated_sptm = dev.update_single_particle_transition_matrix(old_sptm, new_sptm)
        expected_sptm = new_sptm.matrix()
        np.testing.assert_allclose(updated_sptm, expected_sptm)

    def test_lookup_table_property(self):
        dev = NonInteractingFermionicDevice(wires=2)
        assert isinstance(dev.lookup_table, NonInteractingFermionicLookupTable)

    def test_star_state_property(self):
        dev = NonInteractingFermionicDevice(wires=2, shots=8)
        dev.compute_star_state()
        star_state = dev.star_state
        expected_state = np.array([0, 0])
        np.testing.assert_array_equal(star_state, expected_state)

    def test_star_probability_property(self):
        dev = NonInteractingFermionicDevice(wires=2, shots=8)
        dev.compute_star_state()
        prob = dev.star_probability
        np.testing.assert_allclose(prob, 1.0)

    def test_apply_two_state_prep_op(self):
        dev = NonInteractingFermionicDevice(wires=2)
        dev.apply_state_prep(BasisState(np.array([0, 1]), wires=dev.wires), index=0)
        with pytest.raises(DeviceError):
            dev.apply_state_prep(BasisState(np.array([0, 1]), wires=dev.wires), index=1)

    def test_transition_matrix_property_setter(self):
        dev = NonInteractingFermionicDevice(wires=2)
        dev.apply_op(SingleParticleTransitionMatrixOperation(np.eye(4), wires=dev.wires).to_torch())
        dev.transition_matrix = np.eye(4)
        assert isinstance(dev.transition_matrix, torch.Tensor)
        assert dev._lookup_table is None

    def test__dot(self):
        dev = NonInteractingFermionicDevice(wires=2)
        a = torch.tensor([1, 2, 3]).reshape(1, -1).to(dev.R_DTYPE)
        b = torch.tensor([4, 5, 6]).reshape(1, -1).to(dev.R_DTYPE)
        torch.testing.assert_close(dev._dot(a, b), torch.einsum("...i,...i->...", a, b))
