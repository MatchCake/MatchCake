import numpy as np
import pytest

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
