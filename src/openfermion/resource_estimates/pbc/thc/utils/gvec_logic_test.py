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
from openfermion.resource_estimates.pbc.thc.utils.gvec_logic import (
    get_miller_indices,
    get_delta_kp_kq_Q,
    build_transfer_map,
    build_G_vectors,
    build_gpq_mapping,
    build_conjugate_map,
)


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


def test_get_delta_k1_k2_Q():
    kmesh = [3, 2, 1]
    nkpts = np.prod(kmesh)
    scaled_kpts = get_miller_indices(kmesh)
    delta_k1_k2_Q_int = get_delta_kp_kq_Q(scaled_kpts)
    assert delta_k1_k2_Q_int.shape == (nkpts, nkpts, nkpts, 3)
    for kpidx, kqidx, qidx in itertools.product(range(nkpts), repeat=3):
        assert np.allclose(
            scaled_kpts[kpidx] - scaled_kpts[kqidx] - scaled_kpts[qidx],
            delta_k1_k2_Q_int[kpidx, kqidx, qidx],
        )


def test_transfer_map():
    kmesh = [3, 1, 1]
    nkpts = np.prod(kmesh)
    scaled_kpts = get_miller_indices(kmesh)
    delta_k1_k2_Q_int = get_delta_kp_kq_Q(scaled_kpts)
    transfer_map = build_transfer_map(kmesh, scaled_kpts=scaled_kpts)
    for kpidx, kqidx in itertools.product(range(nkpts), repeat=2):
        q_scaled_kpt = scaled_kpts[kpidx] - scaled_kpts[kqidx]
        q_scaled_kpt_aliased = np.array(
            [
                q_scaled_kpt[0] % kmesh[0],
                q_scaled_kpt[1] % kmesh[1],
                q_scaled_kpt[2] % kmesh[2],
            ]
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
    delta_k1_k2_Q_int = get_delta_kp_kq_Q(scaled_kpts)
    transfer_map = build_transfer_map(kmesh, scaled_kpts=scaled_kpts)
    for kpidx, kqidx in itertools.product(range(nkpts), repeat=2):
        q_scaled_kpt = scaled_kpts[kpidx] - scaled_kpts[kqidx]
        q_scaled_kpt_aliased = np.array(
            [
                q_scaled_kpt[0] % kmesh[0],
                q_scaled_kpt[1] % kmesh[1],
                q_scaled_kpt[2] % kmesh[2],
            ]
        )
        qidx = np.where(
            (scaled_kpts[:, 0] == q_scaled_kpt_aliased[0])
            & (scaled_kpts[:, 1] == q_scaled_kpt_aliased[1])
            & (scaled_kpts[:, 2] == q_scaled_kpt_aliased[2])
        )[0]
        assert transfer_map[qidx, kpidx] == kqidx

    true_transfer_map = np.array(
        [[0, 1, 2, 3], [3, 0, 1, 2], [2, 3, 0, 1], [1, 2, 3, 0]]
    )
    assert np.allclose(transfer_map, true_transfer_map)

    kmesh = [3, 2, 1]
    nkpts = np.prod(kmesh)
    scaled_kpts = get_miller_indices(kmesh)
    delta_k1_k2_Q_int = get_delta_kp_kq_Q(scaled_kpts)
    transfer_map = build_transfer_map(kmesh, scaled_kpts=scaled_kpts)
    for kpidx, kqidx in itertools.product(range(nkpts), repeat=2):
        q_scaled_kpt = scaled_kpts[kpidx] - scaled_kpts[kqidx]
        q_scaled_kpt_aliased = np.array(
            [
                q_scaled_kpt[0] % kmesh[0],
                q_scaled_kpt[1] % kmesh[1],
                q_scaled_kpt[2] % kmesh[2],
            ]
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


def test_build_Gvectors():
    kmesh = [3, 2, 1]
    nkpts = np.prod(kmesh)
    g_dict = build_G_vectors(kmesh)
    indx = 0
    for n1, n2, n3 in itertools.product([0, -1], repeat=3):
        assert np.isclose(g_dict[(n1 * kmesh[0], n2 * kmesh[1], n3 * kmesh[2])], indx)
        indx += 1


def test_gpq_mapping():
    kmesh = [3, 2, 1]
    nkpts = np.prod(kmesh)
    scaled_kpts = get_miller_indices(kmesh)
    transfer_map = build_transfer_map(kmesh, scaled_kpts=scaled_kpts)
    gpq_map = build_gpq_mapping(kmesh, scaled_kpts)
    g_dict = build_G_vectors(kmesh)
    g_dict_rev = dict(zip(g_dict.values(), g_dict.keys()))
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


def test_compliment_g():
    # setup
    kmesh = [4, 1, 1]
    nkpts = np.prod(kmesh)
    scaled_kpts = get_miller_indices(kmesh)
    kpoint_dict = dict(
        zip([tuple(map(int, scaled_kpts[x])) for x in range(nkpts)], range(nkpts))
    )
    kconj_map = build_conjugate_map(kmesh, scaled_kpts=scaled_kpts)
    g_dict = build_G_vectors(kmesh)
    gpq_map = build_gpq_mapping(kmesh, scaled_kpts)
    transfer_map = build_transfer_map(kmesh, scaled_kpts=scaled_kpts)

    # confirm that our transfer map function is working
    # confirm Doniminc's Q range
    for qidx, kpidx, ksidx in itertools.product(range(nkpts), repeat=3):
        kqidx = transfer_map[qidx, kpidx]
        kridx = transfer_map[qidx, ksidx]

        conserved_k = (
            scaled_kpts[kpidx]
            - scaled_kpts[kqidx]
            + scaled_kpts[kridx]
            - scaled_kpts[ksidx]
        )
        assert conserved_k[0] % kmesh[0] == 0
        assert conserved_k[1] % kmesh[1] == 0
        assert conserved_k[2] % kmesh[2] == 0

        q_plus_g1 = scaled_kpts[kpidx] - scaled_kpts[kqidx]
        mq_plus_g2 = scaled_kpts[kridx] - scaled_kpts[ksidx]
        # print(q_plus_g1)
        # print(mq_plus_g2)
        # print(q_plus_g1 + mq_plus_g2) # this conserved_k split into two pieces
        # print()

    # to avoid having to make a superposition over G1 and G2 we notice that k - (k-Q) has twice
    # the range and thus we can define the Q variable to be over twice the range and just eliminate
    # the block encoding whenever k-Q is outside the appropriate range for our problem [0, Nkpts].
    # In order to use this strategy we need a way of computing the set of k given a Q in range [-Nkx+1, Nkx-1]
    # such that kx - Qx is in range [0, Nkx-1].  The allowed range of kx is thus [max(0, Qx), min(Nx, Nx + Qx)-1]
    nk = 4
    qidx_range = np.array(np.linspace(-nk + 1, nk - 1, 2 * nk - 1), dtype=int)
    for qidx in qidx_range:
        # find set of k such that k-q is inside the range
        dominics_kvals = []
        for kidx in range(int(np.max([0, qidx])), np.min([4, 4 + qidx])):
            dominics_kvals.append(kidx)
            # print(kidx, kidx - qidx, qidx)
        search_kvals = []
        for kidx in range(nk):
            if kidx - qidx >= 0 and kidx - qidx < nk:
                search_kvals.append(kidx)
        print(dominics_kvals)
        print(search_kvals)
        print(len(search_kvals), nk - abs(qidx))
        print()

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
            # print(qidx, g1idx, g2idx, len(q_idx_g_idx[(qidx, g1idx)]) * len(q_idx_g_idx[(qidx, g2idx)]))
            total_k += len(q_idx_g_idx[(qidx, g1idx)]) * len(q_idx_g_idx[(qidx, g2idx)])
            sub_val += len(q_idx_g_idx[(qidx, g1idx)]) * len(q_idx_g_idx[(qidx, g2idx)])
        assert np.isclose(sub_val, nkpts**2)

    assert np.isclose(total_k, nkpts**3)

    #  for qidx, kpidx in itertools.product(range(nkpts), repeat=2):
    #      kqidx = transfer_map[qidx, kpidx]
    #      nqidx = kconj_map[qidx]
    #      q_plus_g = scaled_kpts[kpidx] - scaled_kpts[kqidx]
    #      nq_plus_compliment_g = scaled_kpts[kqidx] - scaled_kpts[kpidx]
    #      assert np.allclose(q_plus_g, -nq_plus_compliment_g)
    #      g_val = gpq_map[kpidx, kqidx]
    #      q_val = q_plus_g - np.array(g_val)
    #      assert np.allclose(q_val, scaled_kpts[qidx])
    #      # print(q_plus_g)
    #      # print(g_val)
    #      # print()


if __name__ == "__main__":
    test_get_miller_indices()
    test_get_delta_k1_k2_Q()
    test_transfer_map()
    test_build_Gvectors()
    test_gpq_mapping()
    test_build_conjugate_map()
    test_compliment_g()
