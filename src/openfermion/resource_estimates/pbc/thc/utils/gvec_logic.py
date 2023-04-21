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
from pyscf.lib.numpy_helper import cartesian_prod


def get_miller_indices(kmesh):
    """Calculate the miller indices on a gamma centered non-1stBZ Monkorhst-Pack
      mesh

    Args:
      kmesh: 1-D iteratble with the number of k-points in the x,y,z direction
        [Nk_x, NK_y, NK_z] where NK_x/y/z are positive integers

    Returns:
      np.array 2D that is prod([Nk_x, NK_y, NK_z])

    """
    if kmesh[0] < 1:
        raise TypeError("Bad miller index dimension in x")
    if kmesh[1] < 1:
        raise TypeError("Bad miller index dimension in y")
    if kmesh[2] < 1:
        raise TypeError("Bad miller index dimension in z")

    ks_int_each_axis = []
    for n in kmesh:
        ks_int_each_axis.append(np.arange(n, dtype=float))
    int_scaled_kpts = cartesian_prod(ks_int_each_axis)
    return int_scaled_kpts


def get_delta_kp_kq_Q(int_scaled_kpts):
    """Generate kp - kq - Q = S for kp, kq, and Q.  The difference of the three
    integers is stored as a four tensor D_{kp, kq, Q} = S. where the dimension
    of D is (nkpts, nkpts, nkpts, 3).  The last dimension stores the x,y,z
    components of S.

    Args:
      int_scaled_kpts: array of kpts represented as miller indices
        [[nkx, nky, nkz], ...]

    Returns:
      np.array nkpts x nkpts that corresponds to

    """
    delta_k1_k2_Q_int = (int_scaled_kpts[:, None, None, :] -
                         int_scaled_kpts[None, :, None, :] -
                         int_scaled_kpts[None, None, :, :])
    return delta_k1_k2_Q_int


def build_transfer_map(kmesh, scaled_kpts):
    """Define mapping momentum_transfer_map[Q][k1] = k2 that satisfies
    k1 - k2 + G = Q.
    where k1, k2, Q are all tuples of integers [[0, Nkx-1], [0, Nky-1],
    [0, Nkz-1]]
    and G is [{0, Nkx}, {0, Nky}, {0, Nkz}].

    This is computed from `get_delta_kp_kq_Q` which computes k1 - k2 -Q = S.
    Thus k1 - k2 = Q + S which shows that S is [{0, Nkx}, {0, Nky}, {0, Nkz}].

    Thus to compute map[Q, k1] = k2

    Args:
      kmesh: kmesh [nkx, nky, nkz] the number of kpoints in each direction
        scaled_kpts: miller index representation
          [[0, nkx-1], [0, nky-1], [0, nkz-1]] of all the kpoints

    Returns:
      transfer map satisfying k1 - k2 + G = Q in matrix form map[Q, k1] = k2

    """
    nkpts = len(scaled_kpts)
    delta_k1_k2_Q_int = get_delta_kp_kq_Q(scaled_kpts)
    transfer_map = np.zeros((nkpts, nkpts), dtype=np.int32)
    for kpidx, kqidx, qidx in itertools.product(range(nkpts), repeat=3):
        # explicitly build my transfer matrix
        if np.allclose(
            [
                np.rint(delta_k1_k2_Q_int[kpidx, kqidx, qidx][0]) % kmesh[0],
                np.rint(delta_k1_k2_Q_int[kpidx, kqidx, qidx][1]) % kmesh[1],
                np.rint(delta_k1_k2_Q_int[kpidx, kqidx, qidx][2]) % kmesh[2],
            ],
                0,
        ):
            transfer_map[qidx, kpidx] = kqidx
    return transfer_map


def build_conjugate_map(kmesh, scaled_kpts):
    """build mapping such that map[k1] = -k1

    Args:
      kmesh: kpoint mesh
      scaled_kpts: integer k-points

    Returns:
      kconj_map: conjugate k-point mapping

    """
    nkpts = len(scaled_kpts)
    kpoint_dict = dict(
        zip(
            [tuple(map(int, scaled_kpts[x])) for x in range(nkpts)],
            range(nkpts),
        ))
    kconj_map = np.zeros((nkpts), dtype=int)
    for kidx in range(nkpts):
        negative_k_scaled = -scaled_kpts[kidx]
        fb_negative_k_scaled = tuple((
            int(negative_k_scaled[0]) % kmesh[0],
            int(negative_k_scaled[1]) % kmesh[1],
            int(negative_k_scaled[2]) % kmesh[2],
        ))
        kconj_map[kidx] = kpoint_dict[fb_negative_k_scaled]
    return kconj_map


def build_G_vectors(kmesh):
    """Build all 8 Gvectors

    Args:
      kmesh: returns tuple: G_dict a dictionary mapping miller index to
        appropriate
    G_vector index.  The actual cell Gvector can be recovered with
    np.einsum("x,wx->w", (n1, n2, n3), cell.reciprocal_vectors()

    Returns:
      G_dict a dictionary mapping miller index to appropriate
      G_vector index.  The actual cell Gvector can be recovered with
      np.einsum("x,wx->w", (n1, n2, n3), cell.reciprocal_vectors()

    """
    G_dict = {}
    indx = 0
    for n1, n2, n3 in itertools.product([0, -1], repeat=3):
        G_dict[(n1 * kmesh[0], n2 * kmesh[1], n3 * kmesh[2])] = indx
        indx += 1
    return G_dict


def build_gpq_mapping(kmesh, int_scaled_kpts):
    """build map for kp - kq = Q + G where G is
    [{0, -Nkx}, {0, -Nky}, {0, -Nkz}]. G will be 0 or Nkz because kp - kq
    takes on values between [-Nka + 1, Nka - 1] in each component.

    Args:
      kmesh: number of k-points along each direction [Nkx, Nky, Nkz].
      int_scaled_kpts: scaled_kpts. Each kpoint is a tuple of 3 integers
    where each integer is between [0, Nka-1].

    Returns:
      array mapping where first two indices are the index of kp and kq
      and the last dimension holds the gval that is
      [{0, Nkx}, {0, Nky}, {0, Nkz}].

    """
    momentum_map = build_transfer_map(kmesh, int_scaled_kpts)
    nkpts = len(int_scaled_kpts)
    Gpq_mapping = np.zeros((nkpts, nkpts, 3), dtype=np.int32)
    for iq in range(nkpts):
        for ikp in range(nkpts):
            ikq = momentum_map[iq, ikp]
            q_minus_g = int_scaled_kpts[ikp] - int_scaled_kpts[ikq]
            g_val = (
                0 if q_minus_g[0] >= 0 else -kmesh[0],
                0 if q_minus_g[1] >= 0 else -kmesh[1],
                0 if q_minus_g[2] >= 0 else -kmesh[2],
            )
            Gpq_mapping[ikp, ikq, :] = np.array(g_val)

    return Gpq_mapping
