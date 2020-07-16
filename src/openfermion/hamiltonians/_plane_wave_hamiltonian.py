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

import openfermion.utils._operator_utils

from openfermion.hamiltonians._jellium import *
from openfermion.hamiltonians._molecular_data import periodic_hash_table
from openfermion.ops import FermionOperator, QubitOperator


def dual_basis_external_potential(grid, geometry, spinless,
                                  non_periodic=False, period_cutoff=None):
    """Return the external potential in the dual basis of arXiv:1706.00023.

    The external potential resulting from electrons interacting with nuclei
        in the plane wave dual basis.  Note that a cos term is used which is
        strictly only equivalent under aliasing in odd grids, and amounts
        to the addition of an extra term to make the diagonals real on even
        grids.  This approximation is not expected to be significant and allows
        for use of even and odd grids on an even footing.

    Args:
        grid (Grid): The discretization to use.
        geometry: A list of tuples giving the coordinates of each atom.
            example is [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))].
            Distances in atomic units. Use atomic symbols to specify atoms.
        spinless (bool): Whether to use the spinless model or not.
        non_periodic (bool): If the system is non-periodic, default to False.
        period_cutoff (float): Period cutoff, default to
            grid.volume_scale() ** (1. / grid.dimensions)

    Returns:
        FermionOperator: The dual basis operator.
    """
    prefactor = -4.0 * numpy.pi / grid.volume_scale()
    if non_periodic and period_cutoff is None:
        period_cutoff = grid.volume_scale()**(1. / grid.dimensions)
    operator = None
    if spinless:
        spins = [None]
    else:
        spins = [0, 1]
    for pos_indices in grid.all_points_indices():
        coordinate_p = grid.position_vector(pos_indices)
        for nuclear_term in geometry:
            coordinate_j = numpy.array(nuclear_term[1], float)
            for momenta_indices in grid.all_points_indices():
                momenta = grid.momentum_vector(momenta_indices)
                momenta_squared = momenta.dot(momenta)
                if momenta_squared == 0:
                    continue

                cos_index = momenta.dot(coordinate_j - coordinate_p)
                coefficient = (prefactor / momenta_squared *
                               periodic_hash_table[nuclear_term[0]] *
                               numpy.cos(cos_index))

                for spin_p in spins:
                    orbital_p = grid.orbital_id(pos_indices, spin_p)
                    operators = ((orbital_p, 1), (orbital_p, 0))
                    if operator is None:
                        operator = FermionOperator(operators, coefficient)
                    else:
                        operator += FermionOperator(operators, coefficient)
    return operator


def plane_wave_external_potential(grid, geometry, spinless, e_cutoff=None,
                                  non_periodic=False, period_cutoff=None):
    """Return the external potential operator in plane wave basis.

    The external potential resulting from electrons interacting with nuclei.
        It is defined here as the Fourier transform of the dual basis
        Hamiltonian such that is spectrally equivalent in the case of
        both even and odd grids.  Otherwise, the two differ in the case of
        even grids.

    Args:
        grid (Grid): The discretization to use.
        geometry: A list of tuples giving the coordinates of each atom.
            example is [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))].
            Distances in atomic units. Use atomic symbols to specify atoms.
        spinless: Bool, whether to use the spinless model or not.
        e_cutoff (float): Energy cutoff.
        non_periodic (bool): If the system is non-periodic, default to False.
        period_cutoff (float): Period cutoff, default to
            grid.volume_scale() ** (1. / grid.dimensions)

    Returns:
        FermionOperator: The plane wave operator.
    """
    dual_basis_operator = dual_basis_external_potential(grid, geometry,
                                                        spinless, non_periodic,
                                                        period_cutoff)
    operator = (
        openfermion.utils.inverse_fourier_transform(dual_basis_operator,
                                                    grid, spinless))

    return operator


def plane_wave_hamiltonian(grid, geometry=None,
                           spinless=False, plane_wave=True,
                           include_constant=False, e_cutoff=None,
                           non_periodic=False, period_cutoff=None):
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
        non_periodic (bool): If the system is non-periodic, default to False.
        period_cutoff (float): Period cutoff, default to
            grid.volume_scale() ** (1. / grid.dimensions)

    Returns:
        FermionOperator: The hamiltonian.
    """
    if (geometry is not None) and (include_constant is True):
        raise ValueError('Constant term unsupported for non-uniform systems')

    jellium_op = jellium_model(grid, spinless, plane_wave, include_constant,
                               e_cutoff, non_periodic, period_cutoff)

    if geometry is None:
        return jellium_op

    for item in geometry:
        if len(item[1]) != grid.dimensions:
            raise ValueError("Invalid geometry coordinate.")
        if item[0] not in periodic_hash_table:
            raise ValueError("Invalid nuclear element.")

    if plane_wave:
        external_potential = plane_wave_external_potential(
            grid, geometry, spinless, e_cutoff, non_periodic, period_cutoff)
    else:
        external_potential = dual_basis_external_potential(
            grid, geometry, spinless, non_periodic, period_cutoff)

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
    if (geometry is not None) and (include_constant is True):
        raise ValueError('Constant term unsupported for non-uniform systems')

    jellium_op = jordan_wigner_dual_basis_jellium(
        grid, spinless, include_constant)

    if geometry is None:
        return jellium_op

    for item in geometry:
        if len(item[1]) != grid.dimensions:
            raise ValueError("Invalid geometry coordinate.")
        if item[0] not in periodic_hash_table:
            raise ValueError("Invalid nuclear element.")

    n_orbitals = grid.num_points
    volume = grid.volume_scale()
    if spinless:
        n_qubits = n_orbitals
    else:
        n_qubits = 2 * n_orbitals
    prefactor = -2 * numpy.pi / volume
    external_potential = QubitOperator()

    for k_indices in grid.all_points_indices():
        momenta = grid.momentum_vector(k_indices)
        momenta_squared = momenta.dot(momenta)
        if momenta_squared == 0:
            continue

        for p in range(n_qubits):
            index_p = grid.grid_indices(p, spinless)
            coordinate_p = grid.position_vector(index_p)

            for nuclear_term in geometry:
                coordinate_j = numpy.array(nuclear_term[1], float)

                cos_index = momenta.dot(coordinate_j - coordinate_p)
                coefficient = (prefactor / momenta_squared *
                               periodic_hash_table[nuclear_term[0]] *
                               numpy.cos(cos_index))
                external_potential += (QubitOperator((), coefficient) -
                                       QubitOperator(((p, 'Z'),), coefficient))

    return jellium_op + external_potential
