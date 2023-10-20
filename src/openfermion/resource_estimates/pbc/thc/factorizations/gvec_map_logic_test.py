# coverage: ignore
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
import itertools

import numpy as np
import pytest

from openfermion.resource_estimates import HAVE_DEPS_FOR_RESOURCE_ESTIMATES

if HAVE_DEPS_FOR_RESOURCE_ESTIMATES:
    from openfermion.resource_estimates.pbc.thc.factorizations.gvec_map_logic import (  # pylint: disable=line-too-long
        build_conjugate_map,
        build_G_vectors,
        build_gpq_mapping,
        build_transfer_map,
        get_delta_kp_kq_q,
        get_miller_indices,
    )


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
def test_get_miller_indices():
    kmesh = [3, 1, 1]
    int_scaled_kpts = get_miller_indices(kmesh)
    assert np.allclose(int_scaled_kpts[:, 0], np.arange(3))
    assert np.allclose(int_scaled_kpts[:, 1], 0)
    assert np.allclose(int_scaled_kpts[:, 2], 0)

    kmesh = [3, 2, 1]
    int_scaled_kpts = get_miller_indices(kmesh)
    assert np.allclose(int_scaled_kpts[:, 0], [0, 0, 1, 1, 2, 2])
    assert np.allclose(int_scaled_kpts[:, 1], [0, 1, 0, 1, 0, 1])


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
def test_get_delta_k1_k2_Q():
    kmesh = [3, 2, 1]
    nkpts = np.prod(kmesh)
    scaled_kpts = get_miller_indices(kmesh)
    delta_k1_k2_Q_int = get_delta_kp_kq_q(scaled_kpts)
    assert delta_k1_k2_Q_int.shape == (nkpts, nkpts, nkpts, 3)
    for kpidx, kqidx, qidx in itertools.product(range(nkpts), repeat=3):
        assert np.allclose(
            scaled_kpts[kpidx] - scaled_kpts[kqidx] - scaled_kpts[qidx],
            delta_k1_k2_Q_int[kpidx, kqidx, qidx],
        )


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
def test_transfer_map():
    kmesh = [3, 1, 1]
    nkpts = np.prod(kmesh)
    scaled_kpts = get_miller_indices(kmesh)
    transfer_map = build_transfer_map(kmesh, scaled_kpts=scaled_kpts)
    for kpidx, kqidx in itertools.product(range(nkpts), repeat=2):
        q_scaled_kpt = scaled_kpts[kpidx] - scaled_kpts[kqidx]
        q_scaled_kpt_aliased = np.array(
            [q_scaled_kpt[0] % kmesh[0], q_scaled_kpt[1] % kmesh[1], q_scaled_kpt[2] % kmesh[2]]
        )
        qidx = np.where(
            (scaled_kpts[:, 0] == q_scaled_kpt_aliased[0])
            & (scaled_kpts[:, 1] == q_scaled_kpt_aliased[1])
            & (scaled_kpts[:, 2] == q_scaled_kpt_aliased[2])
        )[0]
        assert transfer_map[qidx, kpidx] == kqidx

    true_transfer_map = np.array([[0, 1, 2], [2, 0, 1], [1, 2, 0]])
    assert np.allclose(transfer_map, true_transfer_map)
    kmesh = [4, 1, 1]
    nkpts = np.prod(kmesh)
    scaled_kpts = get_miller_indices(kmesh)
    transfer_map = build_transfer_map(kmesh, scaled_kpts=scaled_kpts)
    for kpidx, kqidx in itertools.product(range(nkpts), repeat=2):
        q_scaled_kpt = scaled_kpts[kpidx] - scaled_kpts[kqidx]
        q_scaled_kpt_aliased = np.array(
            [q_scaled_kpt[0] % kmesh[0], q_scaled_kpt[1] % kmesh[1], q_scaled_kpt[2] % kmesh[2]]
        )
        qidx = np.where(
            (scaled_kpts[:, 0] == q_scaled_kpt_aliased[0])
            & (scaled_kpts[:, 1] == q_scaled_kpt_aliased[1])
            & (scaled_kpts[:, 2] == q_scaled_kpt_aliased[2])
        )[0]
        assert transfer_map[qidx, kpidx] == kqidx

    true_transfer_map = np.array([[0, 1, 2, 3], [3, 0, 1, 2], [2, 3, 0, 1], [1, 2, 3, 0]])
    assert np.allclose(transfer_map, true_transfer_map)

    kmesh = [3, 2, 1]
    nkpts = np.prod(kmesh)
    scaled_kpts = get_miller_indices(kmesh)
    transfer_map = build_transfer_map(kmesh, scaled_kpts=scaled_kpts)
    for kpidx, kqidx in itertools.product(range(nkpts), repeat=2):
        q_scaled_kpt = scaled_kpts[kpidx] - scaled_kpts[kqidx]
        q_scaled_kpt_aliased = np.array(
            [q_scaled_kpt[0] % kmesh[0], q_scaled_kpt[1] % kmesh[1], q_scaled_kpt[2] % kmesh[2]]
        )
        qidx = np.where(
            (scaled_kpts[:, 0] == q_scaled_kpt_aliased[0])
            & (scaled_kpts[:, 1] == q_scaled_kpt_aliased[1])
            & (scaled_kpts[:, 2] == q_scaled_kpt_aliased[2])
        )[0]
        assert transfer_map[qidx, kpidx] == kqidx

    true_transfer_map = np.array(
        [
            [0, 1, 2, 3, 4, 5],
            [1, 0, 3, 2, 5, 4],
            [4, 5, 0, 1, 2, 3],
            [5, 4, 1, 0, 3, 2],
            [2, 3, 4, 5, 0, 1],
            [3, 2, 5, 4, 1, 0],
        ]
    )
    assert np.allclose(transfer_map, true_transfer_map)


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
def test_build_Gvectors():
    kmesh = [3, 2, 1]
    g_dict = build_G_vectors(kmesh)
    indx = 0
    for n1, n2, n3 in itertools.product([0, -1], repeat=3):
        assert np.isclose(g_dict[(n1 * kmesh[0], n2 * kmesh[1], n3 * kmesh[2])], indx)
        indx += 1


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
def test_gpq_mapping():
    kmesh = [3, 2, 1]
    nkpts = np.prod(kmesh)
    scaled_kpts = get_miller_indices(kmesh)
    transfer_map = build_transfer_map(kmesh, scaled_kpts=scaled_kpts)
    gpq_map = build_gpq_mapping(kmesh, scaled_kpts)
    for iq in range(nkpts):
        for ikp in range(nkpts):
            ikq = transfer_map[iq, ikp]
            q_plus_g = scaled_kpts[ikp] - scaled_kpts[ikq]
            g_val = gpq_map[ikp, ikq]
            q_val = q_plus_g - np.array(g_val)
            assert np.allclose(q_val, scaled_kpts[iq])

            assert all(g_val <= 0)
            assert g_val[0] in [0, -kmesh[0]]
            assert g_val[1] in [0, -kmesh[1]]
            assert g_val[2] in [0, -kmesh[2]]


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
def test_build_conjugate_map():
    kmesh = [4, 3, 3]
    nkpts = np.prod(kmesh)
    scaled_kpts = get_miller_indices(kmesh)
    kconj_map = build_conjugate_map(kmesh, scaled_kpts=scaled_kpts)
    for kidx in range(nkpts):
        neg_kscaled_idx = kconj_map[kidx]
        gval = scaled_kpts[neg_kscaled_idx] + scaled_kpts[kidx]
        # -Q + Q = 0 + G
        assert gval[0] in [0, kmesh[0]]
        assert gval[1] in [0, kmesh[1]]
        assert gval[2] in [0, kmesh[2]]


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
def test_compliment_g():
    # setup
    kmesh = [4, 1, 1]
    nkpts = np.prod(kmesh)
    scaled_kpts = get_miller_indices(kmesh)
    g_dict = build_G_vectors(kmesh)
    gpq_map = build_gpq_mapping(kmesh, scaled_kpts)
    transfer_map = build_transfer_map(kmesh, scaled_kpts=scaled_kpts)

    # confirm that our transfer map function is working
    for qidx, kpidx, ksidx in itertools.product(range(nkpts), repeat=3):
        kqidx = transfer_map[qidx, kpidx]
        kridx = transfer_map[qidx, ksidx]

        conserved_k = (
            scaled_kpts[kpidx] - scaled_kpts[kqidx] + scaled_kpts[kridx] - scaled_kpts[ksidx]
        )
        assert conserved_k[0] % kmesh[0] == 0
        assert conserved_k[1] % kmesh[1] == 0
        assert conserved_k[2] % kmesh[2] == 0

    # generate set of k for every (q and g)
    q_idx_g_idx = {}
    for qidx in range(nkpts):
        for gvals, gidx in g_dict.items():
            q_idx_g_idx[(qidx, gidx)] = []
            # print("Traget gval ", gvals)
            kidx_satisfying_q_g = []
            for kidx in range(nkpts):
                kmq_idx = transfer_map[qidx, kidx]
                test_gval = gpq_map[kidx, kmq_idx]
                # print(test_gval, test_gval == gvals)
                if all(test_gval == gvals):
                    kidx_satisfying_q_g.append(kidx)
            q_idx_g_idx[(qidx, gidx)] = kidx_satisfying_q_g

    # count how many total k-values and how many per q values
    total_k = 0
    for qidx in range(nkpts):
        sub_val = 0
        for g1idx, g2idx in itertools.product(range(8), repeat=2):
            total_k += len(q_idx_g_idx[(qidx, g1idx)]) * len(q_idx_g_idx[(qidx, g2idx)])
            sub_val += len(q_idx_g_idx[(qidx, g1idx)]) * len(q_idx_g_idx[(qidx, g2idx)])
        assert np.isclose(sub_val, nkpts**2)

    assert np.isclose(total_k, nkpts**3)
