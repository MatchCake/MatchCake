import numpy as np
import pennylane as qml

from matchcake.devices.swap_injection import MajoranaTermGroups

from ...configs import ATOL_SCALAR_COMPARISON


class TestMajoranaTermGroups:
    """Grouped Majorana term parsing: identity split, size grouping, marker lifting, and per-observable caching."""

    def test_grouping_and_identity(self):
        # The identity term is split off as a scalar weight; every other term lands in a group keyed by the size
        # of its Majorana support, carrying a (n_terms, size) index tensor and an (n_terms,) weight tensor.
        hamiltonian = qml.Hamiltonian(
            [0.7, 0.5, 0.3, 0.2],
            [qml.Identity(0), qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliX(0) @ qml.PauliX(1)],
        )
        term_groups = MajoranaTermGroups.from_observable(hamiltonian, [0, 1, 2], marker=None)
        np.testing.assert_allclose(term_groups.identity_weight, 0.7 + 0.0j, atol=ATOL_SCALAR_COMPARISON)
        n_grouped_terms = 0
        for size, index_tensor, weight_tensor in term_groups.groups:
            assert size % 2 == 0  # the basis path carries parity-even supports only
            assert index_tensor.shape[1] == size
            assert index_tensor.shape[0] == weight_tensor.shape[0]
            n_grouped_terms += index_tensor.shape[0]
        assert n_grouped_terms == 3  # every non-identity term is grouped exactly once

    def test_marker_lifts_odd_terms(self):
        # A parity-odd Pauli word vanishes on the basis path and is lifted with the marker index otherwise.
        hamiltonian = qml.Hamiltonian([1.0], [qml.PauliX(2)])
        term_groups = MajoranaTermGroups.from_observable(hamiltonian, [0, 1, 2], marker=None)
        assert term_groups.identity_weight == 0
        assert term_groups.groups == []
        marker = 7  # D - 1 on the lifted path for n = 3
        lifted = MajoranaTermGroups.from_observable(hamiltonian, [0, 1, 2], marker=marker)
        assert len(lifted.groups) == 1
        size, index_tensor, _ = lifted.groups[0]
        assert size % 2 == 0  # odd rank + marker = even submatrix size
        assert int(index_tensor[0, -1]) == marker  # the marker index is appended last

    def test_from_observable_is_cached_on_the_observable(self):
        # The parsed structure is computed once per (wires, marker) and re-served from the observable itself.
        hamiltonian = qml.Hamiltonian([0.5, 0.3], [qml.PauliZ(0), qml.PauliZ(0) @ qml.PauliZ(1)])
        first = MajoranaTermGroups.from_observable(hamiltonian, [0, 1, 2], marker=None)
        second = MajoranaTermGroups.from_observable(hamiltonian, [0, 1, 2], marker=None)
        assert first is second
        lifted = MajoranaTermGroups.from_observable(hamiltonian, [0, 1, 2], marker=7)
        assert lifted is not first
        assert len(getattr(hamiltonian, MajoranaTermGroups.CACHE_ATTRIBUTE)) == 2
