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

"""Mapping RDMs to other RDMs"""
import numpy
from itertools import product


def krond(i, j):
    return 1 if i == j else 0


def tpdm_to_opdm(tpdm, particle_number):
    """
    Contract a 2-RDM to a 1-RDM

    Args:
        tpdm (numpy.ndarray): The 2-RDM as a 4-index tensor. Indices follow the
            internal convention of tpdm[p, q, r, s] ==
            :math:`a_{p}^{\dagger}a_{q}^{\dagger}a_{r}a_{s}`.
        particle_number (float): number of particles in the system

    Returns:
        opdm (numpy.ndarray): The 1-RDM contracted from the tpdm.
    """
    return numpy.einsum('prrq', tpdm) / (particle_number - 1)


def tpdm_to_tqdm(tpdm, opdm):
    """
    Map from the 2-RDM to the 2-hole-RDM

    Args:
        tpdm (numpy.ndarray): The 2-RDM as a 4-index tensor. Indices follow the
            internal convention of tpdm[p, q, r, s] ==
            :math:`a_{p}^{\dagger}a_{q}^{\dagger}a_{r}a_{s}`.
        opdm (numpy.ndarray): The 1-RDM as a 2-index tensor. Indices follow the
            internal convention of opdm[p, q] ==
            :math:`a_{p}^{\dagger}a_{q}`.

    Returns:
        tqdm (numpy.ndarray): The 2-hole matrix.
    """
    ldim = opdm.shape[0]
    tqdm = numpy.zeros_like(tpdm)
    for p, q, r, s in product(range(ldim), repeat=4):
        term1 = opdm[p, s] * krond(q, r) + opdm[q, r] * krond(p, s)
        term2 = -1 * (opdm[q, s] * krond(p, r) + opdm[p, r] * krond(q, s))
        term3 = krond(q, s) * krond(p, r) - krond(p, s) * krond(q, r)
        tqdm[s, r, q, p] = tpdm[p, q, r, s] - term1 - term2 - term3

    return tqdm


def tqdm_to_tpdm(tqdm, opdm):
    """
    Map from the 2-hole-RDM to the 2-RDM

    Args:
        tqdm (numpy.ndarray): The 2-hole-RDM as a 4-index tensor. Indices
            follow the internal convention of tqdm[p, q, r, s] ==
            :math:`a_{p}a_{q}a_{r}^{\dagger}a_{s}^{\dagger}`.
        opdm (numpy.ndarray): The 1-RDM as a 2-index tensor. Indices follow the
            internal convention of opdm[p, q] ==
            :math:`a_{p}^{\dagger}a_{q}`.

    Returns:
        tpdm (numpy.ndarray): The 2-RDM matrix.
    """
    ldim = opdm.shape[0]
    tpdm = numpy.zeros_like(tqdm)
    for p, q, r, s in product(range(ldim), repeat=4):
        term1 = opdm[p, s] * krond(q, r) + opdm[q, r] * krond(p, s)
        term2 = -1 * (opdm[q, s] * krond(p, r) + opdm[p, r] * krond(q, s))
        term3 = krond(q, s) * krond(p, r) - krond(p, s) * krond(q, r)
        tpdm[p, q, r, s] = tqdm[r, s, p, q] + term1 + term2 + term3

    return tpdm


def tqdm_to_oqdm(tqdm, hole_number):
    """
    Map from 2-hole-RDM to 1-hole-RDM

    Args:
        tqdm (numpy.ndarray): The 2-hole-RDM as a 4-index tensor. Indices
            follow the internal convention of tqdm[p, q, r, s] ==
            :math:`a_{p}a_{q}a_{r}^{\dagger}a_{s}^{\dagger}`.
        hole_number (float):

    Returns:
        oqdm (numpy.ndarray): The 1-hole-RDM contracted from the tqdm.
    """
    return numpy.einsum('prrq', tqdm) / (hole_number - 1)


def opdm_to_oqdm(opdm):
    """
    Convert a 1-RDM to a 1-hole-RDM

    Args:
        opdm (numpy.ndarray): The 1-RDM as a 2-index tensor. Indices follow the
            internal convention of opdm[p, q] ==
            :math:`a_{p}^{\dagger}a_{q}`.:

    Returns:
        oqdm (numpy.ndarray): the 1-hole-RDM transformed from a 1-RDM.
    """
    identity_matrix = numpy.eye(opdm.shape[0])
    return identity_matrix - opdm


def oqdm_to_opdm(oqdm):
    """
    Convert a 1-hole-RDM to a 1-RDM

    Args:
        oqdm (numpy.ndarray): The 1-hole-RDM as a 2-index tensor. Indices
            follow the internal convention of oqdm[p, q] ==
            :math:`a_{p}a_{q}^{\dagger}`.:

    Returns:
        oqdm (numpy.ndarray): the 1-hole-RDM transformed from a 1-RDM.
    """
    identity_matrix = numpy.eye(oqdm.shape[0])
    return identity_matrix - oqdm


def tpdm_to_phdm(tpdm, opdm):
    """
    Map the 2-RDM to the particle-hole-RDM

    Args:
        tpdm (numpy.ndarray): The 2-RDM as a 4-index tensor. Indices follow the
            internal convention of tpdm[p, q, r, s] ==
            :math:`a_{p}^{\dagger}a_{q}^{\dagger}a_{r}a_{s}`.
        opdm (numpy.ndarray): The 1-RDM as a 2-index tensor. Indices follow the
            internal convention of opdm[p, q] ==
            :math:`a_{p}^{\dagger}a_{q}`.

    Returns:
        phdm (numpy.ndarray): The particle-hole matrix.
    """
    ldim = opdm.shape[0]
    tgdm = numpy.zeros_like(tpdm)
    for p, q, r, s in product(range(ldim), repeat=4):
        tgdm[p, r, q, s] = opdm[p, s] * krond(q, r) - tpdm[p, q, r, s]

    return tgdm


def phdm_to_tpdm(phdm, opdm):
    """
    Map the 2-RDM to the particle-hole-RDM

    Args:
        phdm (numpy.ndarray): The 2-particle-hole-RDM as a 4-index tensor.
            Indices follow the internal convention of phdm[p, q, r, s] ==
            :math:`a_{p}^{\dagger}a_{q}a_{r}^{\dagger}a_{s}`.
        opdm (numpy.ndarray): The 1-RDM as a 2-index tensor. Indices follow the
            internal convention of opdm[p, q] ==
            :math:`a_{p}^{\dagger}a_{q}`.

    Returns:
        tpdm (numpy.ndarray): The 2-RDM matrix.
    """
    ldim = opdm.shape[0]
    tpdm = numpy.zeros_like(phdm)
    for p, q, r, s in product(range(ldim), repeat=4):
        tpdm[p, q, r, s] = opdm[p, s] * krond(q, r) - phdm[p, r, q, s]

    return tpdm


def phdm_to_opdm(phdm, num_particles, num_basis_functions):
    """
    Map the particle-hole-RDM to the 1-RDM

    Args:
        phdm (numpy.ndarray): The 2-particle-hole-RDM as a 4-index tensor.
            Indices follow the internal convention of phdm[p, q, r, s] ==
            :math:`a_{p}^{\dagger}a_{q}a_{r}^{\dagger}a_{s}`.
        num_particles: number of particles in the system.
        num_basis_functions: number of spin-orbitals
            (usually the number of qubits)

    Returns:
        opdm (numpy.ndarray): the 1-RDM transformed from a 1-RDM.
    """
    return numpy.einsum('prrq', phdm) / (num_basis_functions -
                                         num_particles + 1)
