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

import itertools
import numpy

from openfermion.ops import FermionOperator, QubitOperator


def uccsd_operator(single_amplitudes, double_amplitudes, anti_hermitian=True):
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

    uccsd_generator = FermionOperator()

    # Re-format inputs (ndarrays to lists) if necessary
    if (isinstance(single_amplitudes, numpy.ndarray) or
            isinstance(double_amplitudes, numpy.ndarray)):
        single_amplitudes, double_amplitudes = uccsd_convert_amplitude_format(
            single_amplitudes,
            double_amplitudes)

    # Add single excitations
    for (i, j), t_ij in single_amplitudes:
        i, j = int(i), int(j)
        uccsd_generator += FermionOperator(((i, 1), (j, 0)), t_ij)
        if anti_hermitian:
            uccsd_generator += FermionOperator(((j, 1), (i, 0)), -t_ij)

    # Add double excitations
    for (i, j, k, l), t_ijkl in double_amplitudes:
        i, j, k, l = int(i), int(j), int(k), int(l)
        uccsd_generator += FermionOperator(
            ((i, 1), (j, 0), (k, 1), (l, 0)), t_ijkl)
        if anti_hermitian:
            uccsd_generator += FermionOperator(
                ((l, 1), (k, 0), (j, 1), (i, 0)), -t_ijkl)
    return uccsd_generator


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
    n_occupied = int(numpy.ceil(n_electrons / 2.))
    n_virtual = n_qubits / 2 - n_occupied

    n_single_amplitudes = n_occupied * n_virtual
    n_double_amplitudes = n_single_amplitudes ** 2
    return (n_single_amplitudes + n_double_amplitudes)


def uccsd_singlet_operator(packed_amplitudes,
                           n_qubits,
                           n_electrons):
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
        uccsd_generator(FermionOperator): Generator of the UCCSD operator that
            builds the UCCSD wavefunction.
    """
    n_occupied = int(numpy.ceil(n_electrons / 2.))
    n_virtual = int(n_qubits / 2 - n_occupied)  # Virtual Spatial Orbitals
    n_t1 = int(n_occupied * n_virtual)

    t1 = packed_amplitudes[:n_t1]
    t2 = packed_amplitudes[n_t1:]

    def t1_ind(i, j):
        return i * n_occupied + j

    def t2_ind(i, j, k, l):
        return (i * n_occupied * n_virtual * n_occupied +
                j * n_virtual * n_occupied +
                k * n_occupied +
                l)

    uccsd_generator = FermionOperator()

    # Define a compound space that is partitioned into occupied, virtual, spins
    # the spin component assumes alpha spins are even, beta spins are odd
    spaces = range(n_virtual), range(n_occupied), range(2)

    # Generate all spin-conserving single excitations between occupied-virtual
    for i, j, s in itertools.product(*spaces):
        uccsd_generator += FermionOperator((
            (2 * (i + n_occupied) + s, 1),
            (2 * j + s, 0)),
            t1[t1_ind(i, j)])

        uccsd_generator += FermionOperator((
            (2 * j + s, 1),
            (2 * (i + n_occupied) + s, 0)),
            -t1[t1_ind(i, j)])

    # Generate all spin-conserving double excitations between occupied-virtual
    for i, j, s, i2, j2, s2 in itertools.product(*spaces, repeat=2):
        uccsd_generator += FermionOperator((
            (2 * (i + n_occupied) + s, 1),
            (2 * j + s, 0),
            (2 * (i2 + n_occupied) + s2, 1),
            (2 * j2 + s2, 0)),
            t2[t2_ind(i, j, i2, j2)])

        uccsd_generator += FermionOperator((
            (2 * j2 + s2, 1),
            (2 * (i2 + n_occupied) + s2, 0),
            (2 * j + s, 1),
            (2 * (i + n_occupied) + s, 0)),
            -t2[t2_ind(i, j, i2, j2)])

    return uccsd_generator
