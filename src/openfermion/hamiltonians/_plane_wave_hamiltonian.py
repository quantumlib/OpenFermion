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

"""Construct Hamiltonians in plan wave basis and its dual in 3D."""
from __future__ import absolute_import

import numpy

from openfermion.config import *
from openfermion.hamiltonians._jellium import *
from openfermion.hamiltonians._molecular_data import periodic_hash_table
from openfermion.ops import FermionOperator, QubitOperator


def wigner_seitz_length_scale(wigner_seitz_radius, n_particles, dimension):
    """Function to give length_scale associated with Wigner-Seitz radius.

    Args:
        wigner_seitz_radius (float): The radius per particle in atomic units.
        n_particles (int): The number of particles in the simulation cell.
        dimension (int): The dimension of the system.

    Returns:
        length_scale (float): The length scale for the simulation.

    Raises:
        ValueError: System dimension must be a positive integer.
    """
    if not isinstance(dimension, int) or dimension < 1:
        raise ValueError('System dimension must be a positive integer.')

    half_dimension = dimension // 2
    if dimension % 2:
        volume_per_particle = (2 * numpy.math.factorial(half_dimension) *
                               (4 * numpy.pi) ** half_dimension /
                               numpy.math.factorial(dimension) *
                               wigner_seitz_radius ** dimension)
    else:
        volume_per_particle = (numpy.pi ** half_dimension /
                               numpy.math.factorial(half_dimension) *
                               wigner_seitz_radius ** dimension)

    volume = volume_per_particle * n_particles
    length_scale = volume ** (1. / dimension)

    return length_scale


def dual_basis_external_potential(grid, geometry, spinless):
    """Return the external potential in the dual basis of arXiv:1706.00023.

    Args:
        grid (Grid): The discretization to use.
        geometry: A list of tuples giving the coordinates of each atom.
            example is [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))].
            Distances in atomic units. Use atomic symbols to specify atoms.
        spinless (bool): Whether to use the spinless model or not.

    Returns:
        FermionOperator: The dual basis operator.
    """
    prefactor = -4.0 * numpy.pi / grid.volume_scale()
    operator = None
    if spinless:
        spins = [None]
    else:
        spins = [0, 1]
    for pos_indices in grid.all_points_indices():
        coordinate_p = position_vector(pos_indices, grid)
        for nuclear_term in geometry:
            coordinate_j = numpy.array(nuclear_term[1], float)
            for momenta_indices in grid.all_points_indices():
                momenta = momentum_vector(momenta_indices, grid)
                momenta_squared = momenta.dot(momenta)
                if momenta_squared == 0:
                    continue

                exp_index = 1.0j * momenta.dot(coordinate_j - coordinate_p)
                coefficient = (prefactor / momenta_squared *
                               periodic_hash_table[nuclear_term[0]] *
                               numpy.exp(exp_index))

                for spin_p in spins:
                    orbital_p = orbital_id(grid, pos_indices, spin_p)
                    operators = ((orbital_p, 1), (orbital_p, 0))
                    if operator is None:
                        operator = FermionOperator(operators, coefficient)
                    else:
                        operator += FermionOperator(operators, coefficient)
    return operator


def plane_wave_external_potential(grid, geometry, spinless, e_cutoff=None):
    """Return the external potential operator in plane wave basis.

    Args:
        grid (Grid): The discretization to use.
        geometry: A list of tuples giving the coordinates of each atom.
            example is [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))].
            Distances in atomic units. Use atomic symbols to specify atoms.
        spinless: Bool, whether to use the spinless model or not.
        e_cutoff (float): Energy cutoff.

    Returns:
        FermionOperator: The plane wave operator.
    """
    prefactor = -4.0 * numpy.pi / grid.volume_scale()
    operator = None
    if spinless:
        spins = [None]
    else:
        spins = [0, 1]

    for indices_p in grid.all_points_indices():
        for indices_q in grid.all_points_indices():
            shift = grid.length // 2
            grid_indices_p_q = [
                (indices_p[i] - indices_q[i] + shift) % grid.length
                for i in range(grid.dimensions)]
            momenta_p_q = momentum_vector(grid_indices_p_q, grid)
            momenta_p_q_squared = momenta_p_q.dot(momenta_p_q)
            if momenta_p_q_squared == 0:
                continue

            # Energy cutoff.
            if e_cutoff is not None and momenta_p_q_squared / 2. > e_cutoff:
                continue

            for nuclear_term in geometry:
                coordinate_j = numpy.array(nuclear_term[1])
                exp_index = 1.0j * momenta_p_q.dot(coordinate_j)
                coefficient = (prefactor / momenta_p_q_squared *
                               periodic_hash_table[nuclear_term[0]] *
                               numpy.exp(exp_index))

                for spin in spins:
                    orbital_p = orbital_id(grid, indices_p, spin)
                    orbital_q = orbital_id(grid, indices_q, spin)
                    operators = ((orbital_p, 1), (orbital_q, 0))
                    if operator is None:
                        operator = FermionOperator(operators, coefficient)
                    else:
                        operator += FermionOperator(operators, coefficient)

    return operator


def plane_wave_hamiltonian(grid, geometry=None,
                           spinless=False, plane_wave=True,
                           include_constant=False, e_cutoff=None):
    """Returns Hamiltonian as FermionOperator class.

    Args:
        grid (Grid): The discretization to use.
        geometry: A list of tuples giving the coordinates of each atom.
            example is [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))].
            Distances in atomic units. Use atomic symbols to specify atoms.
        spinless (bool): Whether to use the spinless model or not.
        plane_wave (bool): Whether to return in plane wave basis (True)
            or plane wave dual basis (False).
        include_constant (bool): Whether to include the Madelung constant.
        e_cutoff (float): Energy cutoff.

    Returns:
        FermionOperator: The hamiltonian.
    """
    jellium_op = jellium_model(grid, spinless, plane_wave, include_constant,
                               e_cutoff)

    if geometry is None:
        return jellium_op

    for item in geometry:
        if len(item[1]) != grid.dimensions:
            raise ValueError("Invalid geometry coordinate.")
        if item[0] not in periodic_hash_table:
            raise ValueError("Invalid nuclear element.")

    if plane_wave:
        external_potential = plane_wave_external_potential(
            grid, geometry, spinless, e_cutoff)
    else:
        external_potential = dual_basis_external_potential(
            grid, geometry, spinless)

    return jellium_op + external_potential


def jordan_wigner_dual_basis_hamiltonian(grid, geometry=None, spinless=False,
                                         include_constant=False):
    """Return the dual basis Hamiltonian as QubitOperator.

    Args:
        grid (Grid): The discretization to use.
        geometry: A list of tuples giving the coordinates of each atom.
            example is [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))].
            Distances in atomic units. Use atomic symbols to specify atoms.
        spinless (bool): Whether to use the spinless model or not.
        include_constant (bool): Whether to include the Madelung constant.

    Returns:
        hamiltonian (QubitOperator)
    """
    jellium_op = jordan_wigner_dual_basis_jellium(
        grid, spinless, include_constant)

    if geometry is None:
        return jellium_op

    for item in geometry:
        if len(item[1]) != grid.dimensions:
            raise ValueError("Invalid geometry coordinate.")
        if item[0] not in periodic_hash_table:
            raise ValueError("Invalid nuclear element.")

    n_orbitals = grid.num_points()
    volume = grid.volume_scale()
    if spinless:
        n_qubits = n_orbitals
    else:
        n_qubits = 2 * n_orbitals
    prefactor = -2 * numpy.pi / volume
    external_potential = QubitOperator()

    for k_indices in grid.all_points_indices():
        momenta = momentum_vector(k_indices, grid)
        momenta_squared = momenta.dot(momenta)
        if momenta_squared == 0:
            continue

        for p in range(n_qubits):
            index_p = grid_indices(p, grid, spinless)
            coordinate_p = position_vector(index_p, grid)

            for nuclear_term in geometry:
                coordinate_j = numpy.array(nuclear_term[1], float)

                exp_index = 1.0j * momenta.dot(coordinate_j - coordinate_p)
                coefficient = (prefactor / momenta_squared *
                               periodic_hash_table[nuclear_term[0]] *
                               numpy.exp(exp_index))
                external_potential += (QubitOperator((), coefficient) -
                                       QubitOperator(((p, 'Z'),), coefficient))

    return jellium_op + external_potential
