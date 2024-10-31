import numpy as np
import pytest

import matchcake as mc
from matchcake import utils
from matchcake.operations import (
    fSWAP,
)
from matchcake.operations.single_particle_transition_matrices import (
    SptmFSwap,
)
from ...configs import (
    ATOL_APPROX_COMPARISON,
    RTOL_APPROX_COMPARISON,
    set_seed,
    TEST_SEED,
)

set_seed(TEST_SEED)


def test_matchgate_equal_to_sptm_fswap():
    matchgate = fSWAP(wires=[0, 1])
    m_sptm = matchgate.single_particle_transition_matrix
    sptm = SptmFSwap(wires=[0, 1]).matrix()
    np.testing.assert_allclose(
        sptm, m_sptm,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "wire0, wire1, all_wires",
    [
        (wire0, wire1, n_wires)
        for n_wires in range(2, 16)
        for wire0 in range(n_wires - 1)
        for wire1 in range(wire0 + 1, n_wires)
    ]
)
def test_sptm_fswap_chain_equal_to_sptm_fswap(wire0, wire1, all_wires):
    all_wires = list(range(all_wires))

    def _gen():
        for tmp_wire0 in range(wire0, wire1):
            tmp_wire1 = tmp_wire0 + 1
            yield SptmFSwap(wires=[tmp_wire0, tmp_wire1])
        return

    device = mc.NIFDevice(wires=all_wires)
    device.execute_generator(_gen(), reset=True, apply=True, cache_global_sptm=True)
    chain_sptm = device.apply_metadata["global_sptm"]
    sptm = SptmFSwap(wires=[wire0, wire1]).matrix(all_wires)

    np.testing.assert_allclose(
        sptm, chain_sptm,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )


@pytest.mark.parametrize(
    "wire0, wire1, all_wires",
    [
        (wire0, wire1, n_wires)
        for n_wires in range(2, 16)
        for wire0 in range(n_wires - 1)
        for wire1 in range(wire0 + 1, n_wires)
    ]
)
def test_sptm_fswap_chain_equal_to_sptm_fswap_reverse(wire0, wire1, all_wires):
    all_wires = list(range(all_wires))

    def _gen():
        for tmp_wire0 in range(wire1 - 2, wire0 - 1, -1):
            tmp_wire1 = tmp_wire0 + 1
            yield SptmFSwap(wires=[tmp_wire0, tmp_wire1])
        return

    device = mc.NIFDevice(wires=all_wires)
    device.execute_generator(_gen(), reset=True, apply=True, cache_global_sptm=True)
    chain_sptm = device.apply_metadata["global_sptm"]
    sptm = SptmFSwap(wires=[wire0, wire1]).matrix(all_wires)

    np.testing.assert_allclose(
        sptm, chain_sptm,
        atol=ATOL_APPROX_COMPARISON,
        rtol=RTOL_APPROX_COMPARISON,
    )
