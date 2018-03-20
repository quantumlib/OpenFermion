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

"""Module to create and manipulate unitary coupled cluster operators."""

from __future__ import division

import itertools

import numpy
from openfermion.ops import FermionOperator, QubitOperator
from openfermion.utils import up_index, down_index


def uccsd_generator(single_amplitudes, double_amplitudes, anti_hermitian=True):
    """Create a fermionic operator that is the generator of uccsd.

    This a the most straight-forward method to generate UCCSD operators,
    however it is slightly inefficient. In particular, it parameterizes
    all possible excitations, so it represents a generalized unitary coupled
    cluster ansatz, but also does not explicitly enforce the uniqueness
    in parametrization, so it is redundant. For example there will be a linear
    dependency in the ansatz of single_amplitudes[i,j] and
    single_amplitudes[j,i].

    Args:
        single_amplitudes(list or ndarray): list of lists with each sublist
            storing a list of indices followed by single excitation amplitudes
            i.e. [[[i,j],t_ij], ...] OR [NxN] array storing single excitation
            amplitudes corresponding to
            t[i,j] * (a_i^\dagger a_j - H.C.)
        double_amplitudes(list or ndarray): list of lists with each sublist
            storing a list of indices followed by double excitation amplitudes
            i.e. [[[i,j,k,l],t_ijkl], ...] OR [NxNxNxN] array storing double
            excitation amplitudes corresponding to
            t[i,j,k,l] * (a_i^\dagger a_j a_k^\dagger a_l - H.C.)
        anti_hermitian(Bool): Flag to generate only normal CCSD operator
            rather than unitary variant, primarily for testing

    Returns:
        uccsd_generator(FermionOperator): Anti-hermitian fermion operator that
        is the generator for the uccsd wavefunction.
    """

    generator = FermionOperator()

    # Re-format inputs (ndarrays to lists) if necessary
    if (isinstance(single_amplitudes, numpy.ndarray) or
            isinstance(double_amplitudes, numpy.ndarray)):
        single_amplitudes, double_amplitudes = uccsd_convert_amplitude_format(
            single_amplitudes,
            double_amplitudes)

    # Add single excitations
    for (i, j), t_ij in single_amplitudes:
        i, j = int(i), int(j)
        generator += FermionOperator(((i, 1), (j, 0)), t_ij)
        if anti_hermitian:
            generator += FermionOperator(((j, 1), (i, 0)), -t_ij)

    # Add double excitations
    for (i, j, k, l), t_ijkl in double_amplitudes:
        i, j, k, l = int(i), int(j), int(k), int(l)
        generator += FermionOperator(
            ((i, 1), (j, 0), (k, 1), (l, 0)), t_ijkl)
        if anti_hermitian:
            generator += FermionOperator(
                ((l, 1), (k, 0), (j, 1), (i, 0)), -t_ijkl)
    return generator


def uccsd_convert_amplitude_format(single_amplitudes, double_amplitudes):
    """Re-format single_amplitudes and double_amplitudes from ndarrays to lists.

    Args:
        single_amplitudes(ndarray): [NxN] array storing single excitation
            amplitudes corresponding to t[i,j] * (a_i^\dagger a_j - H.C.)
        double_amplitudes(ndarray): [NxNxNxN] array storing double
            excitation amplitudes corresponding to
            t[i,j,k,l] * (a_i^\dagger a_j a_k^\dagger a_l - H.C.)

    Returns:
        single_amplitudes_list(list): list of lists with each sublist storing
            a list of indices followed by single excitation amplitudes
            i.e. [[[i,j],t_ij], ...]
        double_amplitudes_list(list): list of lists with each sublist storing
            a list of indices followed by double excitation amplitudes
            i.e. [[[i,j,k,l],t_ijkl], ...]
    """
    single_amplitudes_list, double_amplitudes_list = [], []

    for i, j in zip(*single_amplitudes.nonzero()):
        single_amplitudes_list.append([[i, j], single_amplitudes[i, j]])

    for i, j, k, l in zip(*double_amplitudes.nonzero()):
        double_amplitudes_list.append([[i, j, k, l],
                                      double_amplitudes[i, j, k, l]])
    return single_amplitudes_list, double_amplitudes_list


def uccsd_singlet_paramsize(n_qubits, n_electrons):
    """Determine number of independent amplitudes for singlet UCCSD

    Args:
        n_qubits(int): Number of qubits/spin-orbitals in the system
        n_electrons(int): Number of electrons in the reference state

    Returns:
        Number of independent parameters for singlet UCCSD with a single
        reference.
    """
    if n_qubits % 2 != 0:
        raise ValueError('The total number of spin-orbitals should be even.')

    # Since the total spin S^2 is conserved, we work with spatial orbitals
    n_spatial_orbitals = n_qubits // 2
    n_occupied = int(numpy.ceil(n_electrons / 2))
    n_virtual = n_spatial_orbitals - n_occupied

    n_single_amplitudes = n_occupied * n_virtual

    # Below is equivalent to
    #     2 * n_single_amplitudes + (n_single_amplitudes choose 2)
    n_double_amplitudes = n_single_amplitudes * (n_single_amplitudes + 3) // 2

    return n_single_amplitudes + n_double_amplitudes


def uccsd_singlet_generator(packed_amplitudes, n_qubits, n_electrons):
    """Create a singlet UCCSD generator for a system with n_electrons

    This function generates a FermionOperator for a UCCSD generator designed
        to act on a single reference state consisting of n_qubits spin orbitals
        and n_electrons electrons, that is a spin singlet operator, meaning it
        conserves spin.

    Args:
        packed_amplitudes(ndarray): Compact array storing the unique single
            and double excitation amplitudes for a singlet UCCSD operator.
            The ordering lists unique single excitations before double
            excitations.
        n_qubits(int): Number of spin-orbitals used to represent the system,
            which also corresponds to number of qubits in a non-compact map.
        n_electrons(int): Number of electrons in the physical system.

    Returns:
        generator(FermionOperator): Generator of the UCCSD operator that
            builds the UCCSD wavefunction.
    """
    if n_qubits % 2 != 0:
        raise ValueError('The total number of spin-orbitals should be even.')

    n_spatial_orbitals = n_qubits // 2
    n_occupied = int(numpy.ceil(n_electrons / 2))
    n_virtual = n_spatial_orbitals - n_occupied

    # Unpack amplitudes
    n_single_amplitudes = n_occupied * n_virtual
    # Single amplitudes
    t1 = packed_amplitudes[:n_single_amplitudes]
    # Double amplitudes associated with one spatial occupied-virtual pair
    t2_1 = packed_amplitudes[n_single_amplitudes:3 * n_single_amplitudes]
    # Double amplitudes associated with two spatial occupied-virtual pairs
    t2_2 = packed_amplitudes[3 * n_single_amplitudes:]

    # Initialize operator
    generator = FermionOperator()

    # Generate all spin-conserving single and double excitations derived
    # from one spatial occupied-virtual pair
    for i, (p, q) in enumerate(
            itertools.product(range(n_virtual), range(n_occupied))):

        # Get indices of spatial orbitals
        virtual_spatial = n_occupied + p
        occupied_spatial = q
        # Get indices of spin orbitals
        virtual_up = up_index(virtual_spatial)
        virtual_down = down_index(virtual_spatial)
        occupied_up = up_index(occupied_spatial)
        occupied_down = down_index(occupied_spatial)
        
        # Generate single excitations 
        coeff = t1[i]
        # Spin up excitation
        generator += FermionOperator((
            (virtual_up, 1),
            (occupied_up, 0)),
            coeff)
        generator += FermionOperator((
            (occupied_up, 1),
            (virtual_up, 0)),
            -coeff)
        # Spin down excitation
        generator += FermionOperator((
            (virtual_down, 1),
            (occupied_down, 0)),
            coeff)
        generator += FermionOperator((
            (occupied_down, 1),
            (virtual_down, 0)),
            -coeff)

        # Generate double excitations
        # up -> up and down -> down
        coeff = t2_1[2 * i]
        generator += FermionOperator((
            (virtual_up, 1),
            (occupied_up, 0),
            (virtual_down, 1),
            (occupied_down, 0)),
            coeff)
        generator += FermionOperator((
            (occupied_down, 1),
            (virtual_down, 0),
            (occupied_up, 1),
            (virtual_up, 0)),
            -coeff)
        # up -> down and down -> up
        coeff = t2_1[2 * i + 1]
        generator += FermionOperator((
            (virtual_down, 1),
            (occupied_up, 0),
            (virtual_up, 1),
            (occupied_down, 0)),
            coeff)
        generator += FermionOperator((
            (occupied_down, 1),
            (virtual_up, 0),
            (occupied_up, 1),
            (virtual_down, 0)),
            -coeff)

    # Generate all spin-conserving double excitations derived
    # from two spatial occupied-virtual pairs
    for i, ((p, q), (r, s)) in enumerate(
            itertools.combinations(
                itertools.product(range(n_virtual), range(n_occupied)),
                2)):

        # Get indices of spatial orbitals
        virtual_spatial_1 = n_occupied + p
        occupied_spatial_1 = q
        virtual_spatial_2 = n_occupied + r
        occupied_spatial_2 = s
        # Get indices of spin orbitals
        virtual_up_1 = up_index(virtual_spatial_1)
        virtual_down_1 = down_index(virtual_spatial_1)
        occupied_up_1 = up_index(occupied_spatial_1)
        occupied_down_1 = down_index(occupied_spatial_1)
        virtual_up_2 = up_index(virtual_spatial_2)
        virtual_down_2 = down_index(virtual_spatial_2)
        occupied_up_2 = up_index(occupied_spatial_2)
        occupied_down_2 = down_index(occupied_spatial_2)

        # Generate double excitations
        coeff = t2_2[i]
        # p -> q is spin up and s -> r is spin down
        generator += FermionOperator((
            (virtual_up_1, 1),
            (occupied_up_1, 0),
            (virtual_down_2, 1),
            (occupied_down_2, 0)),
            coeff)
        generator += FermionOperator((
            (occupied_down_2, 1),
            (virtual_down_2, 0),
            (occupied_up_1, 1),
            (virtual_up_1, 0)),
            -coeff)
        # p -> q is spin down and s -> r is spin up
        generator += FermionOperator((
            (virtual_up_2, 1),
            (occupied_up_2, 0),
            (virtual_down_1, 1),
            (occupied_down_1, 0)),
            coeff)
        generator += FermionOperator((
            (occupied_down_1, 1),
            (virtual_down_1, 0),
            (occupied_up_2, 1),
            (virtual_up_2, 0)),
            -coeff)
        # Both are spin up excitations
        generator += FermionOperator((
            (virtual_up_1, 1),
            (occupied_up_1, 0),
            (virtual_up_2, 1),
            (occupied_up_2, 0)),
            coeff)
        generator += FermionOperator((
            (occupied_up_2, 1),
            (virtual_up_2, 0),
            (occupied_up_1, 1),
            (virtual_up_1, 0)),
            -coeff)
        # Both are spin down excitations
        generator += FermionOperator((
            (virtual_down_1, 1),
            (occupied_down_1, 0),
            (virtual_down_2, 1),
            (occupied_down_2, 0)),
            coeff)
        generator += FermionOperator((
            (occupied_down_2, 1),
            (virtual_down_2, 0),
            (occupied_down_1, 1),
            (virtual_down_1, 0)),
            -coeff)

    return generator
