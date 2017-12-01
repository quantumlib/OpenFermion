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

"""This module constructs Hamiltonians for the uniform electron gas."""
from __future__ import absolute_import

import numpy

from openfermion.ops import FermionOperator, QubitOperator


# Exceptions.
class OrbitalSpecificationError(Exception):
    pass


def orbital_id(grid, grid_coordinates, spin=None):
    """Return the tensor factor of a orbital with given coordinates and spin.

    Args:
        grid (Grid): The discretization to use.
        grid_coordinates: List or tuple of ints giving coordinates of grid
            element. Acceptable to provide an int (instead of tuple or list)
            for 1D case.
        spin: Boole, 0 means spin down and 1 means spin up.
            If None, assume spinless model.

    Returns:
        tensor_factor (int):
            tensor factor associated with provided orbital label.
    """
    # Initialize.
    if isinstance(grid_coordinates, int):
        grid_coordinates = [grid_coordinates]

    # Loop through dimensions of coordinate tuple.
    tensor_factor = 0
    for dimension, grid_coordinate in enumerate(grid_coordinates):

        # Make sure coordinate is an integer in the correct bounds.
        if isinstance(grid_coordinate, int) and grid_coordinate < grid.length:
            tensor_factor += grid_coordinate * (grid.length ** dimension)

        else:
            # Raise for invalid model.
            raise OrbitalSpecificationError(
                'Invalid orbital coordinates provided.')

    # Account for spin and return.
    if spin is None:
        return tensor_factor
    else:
        tensor_factor *= 2
        tensor_factor += spin
        return tensor_factor


def grid_indices(qubit_id, grid, spinless):
    """This function is the inverse of orbital_id.

    Args:
        qubit_id (int): The tensor factor to map to grid indices.
        grid (Grid): The discretization to use.
        spinless (bool): Whether to use the spinless model or not.

    Returns:
        grid_indices (numpy.ndarray[int]):
            The location of the qubit on the grid.
    """
    # Remove spin degree of freedom.
    orbital_id = qubit_id
    if not spinless:
        if (orbital_id % 2):
            orbital_id -= 1
        orbital_id /= 2

    # Get grid indices.
    grid_indices = []
    for dimension in range(grid.dimensions):
        remainder = orbital_id % (grid.length ** (dimension + 1))
        grid_index = remainder // (grid.length ** dimension)
        grid_indices += [grid_index]
    return grid_indices


def position_vector(position_indices, grid):
    """Given grid point coordinate, return position vector with dimensions.

    Args:
        position_indices (int|iterable[int]):
            List or tuple of integers giving grid point coordinate.
            Allowed values are ints in [0, grid_length).
        grid (Grid): The discretization to use.

    Returns:
        position_vector (numpy.ndarray[float])
    """
    # Raise exceptions.
    if isinstance(position_indices, int):
        position_indices = [position_indices]
    if not all(0 <= e < grid.length for e in position_indices):
        raise OrbitalSpecificationError(
            'Position indices must be integers in [0, grid_length).')

    # Compute position vector.
    adjusted_vector = numpy.array(position_indices, float) - grid.length // 2
    return grid.scale * adjusted_vector / float(grid.length)


def momentum_vector(momentum_indices, grid):
    """Given grid point coordinate, return momentum vector with dimensions.

    Args:
        momentum_indices: List or tuple of integers giving momentum indices.
            Allowed values are ints in [0, grid_length).
        grid (Grid): The discretization to use.

        Returns:
            momentum_vector: A numpy array giving the momentum vector with
                dimensions.
    """
    # Raise exceptions.
    if isinstance(momentum_indices, int):
        momentum_indices = [momentum_indices]
    if not all(0 <= e < grid.length for e in momentum_indices):
        raise OrbitalSpecificationError(
            'Momentum indices must be integers in [0, grid_length).')

    # Compute momentum vector.
    adjusted_vector = numpy.array(momentum_indices, float) - grid.length // 2
    return 2. * numpy.pi * adjusted_vector / grid.scale


def plane_wave_kinetic(grid, spinless=False, e_cutoff=None):
    """Return the kinetic energy operator in the plane wave basis.

    Args:
        grid (openfermion.utils.Grid): The discretization to use.
        spinless (bool): Whether to use the spinless model or not.
        e_cutoff (float): Energy cutoff.

    Returns:
        FermionOperator: The kinetic momentum operator.
    """
    # Initialize.
    operator = FermionOperator()
    spins = [None] if spinless else [0, 1]

    # Loop once through all plane waves.
    for momenta_indices in grid.all_points_indices():
        momenta = momentum_vector(momenta_indices, grid)
        coefficient = momenta.dot(momenta) / 2.

        # Energy cutoff.
        if e_cutoff is not None and coefficient > e_cutoff:
            continue

        # Loop over spins.
        for spin in spins:
            orbital = orbital_id(grid, momenta_indices, spin)

            # Add interaction term.
            operators = ((orbital, 1), (orbital, 0))
            operator += FermionOperator(operators, coefficient)

    return operator


def plane_wave_potential(grid, spinless=False, e_cutoff=None):
    """Return the potential operator in the plane wave basis.

    Args:
        grid (Grid): The discretization to use.
        spinless (bool): Whether to use the spinless model or not.
        e_cutoff (float): Energy cutoff.

    Returns:
        operator (FermionOperator)
    """
    # Initialize.
    prefactor = 2. * numpy.pi / grid.volume_scale()
    operator = FermionOperator((), 0.0)
    spins = [None] if spinless else [0, 1]

    # Pre-Computations.
    shifted_omega_indices_dict = {}
    shifted_indices_minus_dict = {}
    shifted_indices_plus_dict = {}
    orbital_ids = {}
    for indices_a in grid.all_points_indices():
        shifted_omega_indices = [j - grid.length // 2 for j in indices_a]
        shifted_omega_indices_dict[indices_a] = shifted_omega_indices
        shifted_indices_minus_dict[indices_a] = {}
        shifted_indices_plus_dict[indices_a] = {}
        for indices_b in grid.all_points_indices():
            shifted_indices_minus_dict[indices_a][indices_b] = tuple([
                (indices_b[i] - shifted_omega_indices[i]) % grid.length
                for i in range(grid.dimensions)])
            shifted_indices_plus_dict[indices_a][indices_b] = tuple([
                (indices_b[i] + shifted_omega_indices[i]) % grid.length
                for i in range(grid.dimensions)])
        orbital_ids[indices_a] = {}
        for spin in spins:
            orbital_ids[indices_a][spin] = orbital_id(grid, indices_a, spin)

    # Loop once through all plane waves.
    for omega_indices in grid.all_points_indices():
        shifted_omega_indices = shifted_omega_indices_dict[omega_indices]

        # Get the momenta vectors.
        momenta = momentum_vector(omega_indices, grid)
        momenta_squared = momenta.dot(momenta)

        # Skip if momentum is zero.
        if momenta_squared == 0:
            continue

        # Energy cutoff.
        if e_cutoff is not None and momenta_squared / 2. > e_cutoff:
            continue

        # Compute coefficient.
        coefficient = prefactor / momenta_squared

        for grid_indices_a in grid.all_points_indices():
            shifted_indices_d = (
                shifted_indices_minus_dict[omega_indices][grid_indices_a])
            for grid_indices_b in grid.all_points_indices():
                shifted_indices_c = (
                    shifted_indices_plus_dict[omega_indices][grid_indices_b])

                # Loop over spins.
                for spin_a in spins:
                    orbital_a = orbital_ids[grid_indices_a][spin_a]
                    orbital_d = orbital_ids[shifted_indices_d][spin_a]
                    for spin_b in spins:
                        orbital_b = orbital_ids[grid_indices_b][spin_b]
                        orbital_c = orbital_ids[shifted_indices_c][spin_b]

                        # Add interaction term.
                        if ((orbital_a != orbital_b) and
                                (orbital_c != orbital_d)):
                            operators = ((orbital_a, 1), (orbital_b, 1),
                                         (orbital_c, 0), (orbital_d, 0))
                            operator += FermionOperator(operators, coefficient)

    # Return.
    return operator


def dual_basis_jellium_model(grid, spinless=False,
                             kinetic=True, potential=True,
                             include_constant=False):
    """Return jellium Hamiltonian in the dual basis of arXiv:1706.00023

    Args:
        grid (Grid): The discretization to use.
        spinless (bool): Whether to use the spinless model or not.
        kinetic (bool): Whether to include kinetic terms.
        potential (bool): Whether to include potential terms.
        include_constant (bool): Whether to include the Madelung constant.

    Returns:
        operator (FermionOperator)
    """
    # Initialize.
    n_points = grid.num_points()
    position_prefactor = 2. * numpy.pi / grid.volume_scale()
    operator = FermionOperator()
    spins = [None] if spinless else [0, 1]

    # Pre-Computations.
    position_vectors = {}
    momentum_vectors = {}
    momenta_squared_dict = {}
    orbital_ids = {}
    for indices in grid.all_points_indices():
        position_vectors[indices] = position_vector(indices, grid)
        momenta = momentum_vector(indices, grid)
        momentum_vectors[indices] = momenta
        momenta_squared_dict[indices] = momenta.dot(momenta)
        orbital_ids[indices] = {}
        for spin in spins:
            orbital_ids[indices][spin] = orbital_id(grid, indices, spin)

    # Loop once through all lattice sites.
    for grid_indices_a in grid.all_points_indices():
        coordinates_a = position_vectors[grid_indices_a]
        for grid_indices_b in grid.all_points_indices():
            coordinates_b = position_vectors[grid_indices_b]
            differences = coordinates_b - coordinates_a

            # Compute coefficients.
            kinetic_coefficient = 0.
            potential_coefficient = 0.
            for momenta_indices in grid.all_points_indices():
                momenta = momentum_vectors[momenta_indices]
                momenta_squared = momenta_squared_dict[momenta_indices]
                if momenta_squared == 0:
                    continue

                cos_difference = numpy.cos(momenta.dot(differences))
                if kinetic:
                    kinetic_coefficient += (
                        cos_difference * momenta_squared /
                        (2. * float(n_points)))
                if potential:
                    potential_coefficient += (
                        position_prefactor * cos_difference / momenta_squared)

            # Loop over spins and identify interacting orbitals.
            orbital_a = {}
            orbital_b = {}
            for spin in spins:
                orbital_a[spin] = orbital_ids[grid_indices_a][spin]
                orbital_b[spin] = orbital_ids[grid_indices_b][spin]
            if kinetic:
                for spin in spins:
                    operators = ((orbital_a[spin], 1), (orbital_b[spin], 0))
                    operator += FermionOperator(operators, kinetic_coefficient)
            if potential:
                for sa in spins:
                    for sb in spins:
                        if orbital_a[sa] == orbital_b[sb]:
                            continue
                        operators = ((orbital_a[sa], 1), (orbital_a[sa], 0),
                                     (orbital_b[sb], 1), (orbital_b[sb], 0))
                        operator += FermionOperator(operators,
                                                    potential_coefficient)

    # Include the Madelung constant if requested.
    if include_constant:
        operator += FermionOperator.identity() * (2.8372 / grid.scale)

    # Return.
    return operator


def dual_basis_kinetic(grid, spinless=False):
    """Return the kinetic operator in the dual basis of arXiv:1706.00023.

    Args:
        grid (Grid): The discretization to use.
        spinless (bool): Whether to use the spinless model or not.

    Returns:
        operator (FermionOperator)
    """
    return dual_basis_jellium_model(grid, spinless, True, False)


def dual_basis_potential(grid, spinless=False):
    """Return the potential operator in the dual basis of arXiv:1706.00023

    Args:
        grid (Grid): The discretization to use.
        spinless (bool): Whether to use the spinless model or not.

    Returns:
        operator (FermionOperator)
    """
    return dual_basis_jellium_model(grid, spinless, False, True)


def jellium_model(grid, spinless=False, plane_wave=True,
                  include_constant=False, e_cutoff=None):
    """Return jellium Hamiltonian as FermionOperator class.

    Args:
        grid (openfermion.utils.Grid): The discretization to use.
        spinless (bool): Whether to use the spinless model or not.
        plane_wave (bool): Whether to return in momentum space (True)
            or position space (False).
        include_constant (bool): Whether to include the Madelung constant.
        e_cutoff (float): Energy cutoff.

    Returns:
        FermionOperator: The Hamiltonian of the model.
    """
    if plane_wave:
        hamiltonian = plane_wave_kinetic(grid, spinless, e_cutoff)
        hamiltonian += plane_wave_potential(grid, spinless, e_cutoff)
    else:
        hamiltonian = dual_basis_jellium_model(grid, spinless)
    # Include the Madelung constant if requested.
    if include_constant:
        hamiltonian += FermionOperator.identity() * (2.8372 / grid.scale)
    return hamiltonian


def jordan_wigner_dual_basis_jellium(grid, spinless=False,
                                     include_constant=False):
    """Return the jellium Hamiltonian as QubitOperator in the dual basis.

    Args:
        grid (Grid): The discretization to use.
        spinless (bool): Whether to use the spinless model or not.
        include_constant (bool): Whether to include the Madelung constant.

    Returns:
        hamiltonian (QubitOperator)
    """
    # Initialize.
    n_orbitals = grid.num_points()
    volume = grid.volume_scale()
    if spinless:
        n_qubits = n_orbitals
    else:
        n_qubits = 2 * n_orbitals
    hamiltonian = QubitOperator()

    # Compute vectors.
    momentum_vectors = {}
    momenta_squared_dict = {}
    for indices in grid.all_points_indices():
        momenta = momentum_vector(indices, grid)
        momentum_vectors[indices] = momenta
        momenta_squared_dict[indices] = momenta.dot(momenta)

    # Compute the identity coefficient and the coefficient of local Z terms.
    identity_coefficient = 0.
    z_coefficient = 0.
    for k_indices in grid.all_points_indices():
        momenta = momentum_vectors[k_indices]
        momenta_squared = momenta.dot(momenta)
        if momenta_squared == 0:
            continue

        identity_coefficient += momenta_squared / 2.
        identity_coefficient -= (numpy.pi * float(n_orbitals) /
                                 (momenta_squared * volume))
        z_coefficient += numpy.pi / (momenta_squared * volume)
        z_coefficient -= momenta_squared / (4. * float(n_orbitals))
    if spinless:
        identity_coefficient /= 2.

    # Add identity term.
    identity_term = QubitOperator((), identity_coefficient)
    hamiltonian += identity_term

    # Add local Z terms.
    for qubit in range(n_qubits):
        qubit_term = QubitOperator(((qubit, 'Z'),), z_coefficient)
        hamiltonian += qubit_term

    # Add ZZ terms and XZX + YZY terms.
    zz_prefactor = numpy.pi / volume
    xzx_yzy_prefactor = .25 / float(n_orbitals)
    for p in range(n_qubits):
        index_p = grid_indices(p, grid, spinless)
        position_p = position_vector(index_p, grid)
        for q in range(p + 1, n_qubits):
            index_q = grid_indices(q, grid, spinless)
            position_q = position_vector(index_q, grid)

            difference = position_p - position_q

            skip_xzx_yzy = not spinless and (p + q) % 2

            # Loop through momenta.
            zpzq_coefficient = 0.
            term_coefficient = 0.
            for k_indices in grid.all_points_indices():
                momenta = momentum_vectors[k_indices]
                momenta_squared = momenta_squared_dict[k_indices]
                if momenta_squared == 0:
                    continue

                cos_difference = numpy.cos(momenta.dot(difference))

                zpzq_coefficient += (zz_prefactor * cos_difference /
                                     momenta_squared)

                if skip_xzx_yzy:
                    continue
                term_coefficient += (xzx_yzy_prefactor * cos_difference *
                                     momenta_squared)

            # Add ZZ term.
            qubit_term = QubitOperator(((p, 'Z'), (q, 'Z')), zpzq_coefficient)
            hamiltonian += qubit_term

            # Add XZX + YZY term.
            if skip_xzx_yzy:
                continue
            z_string = tuple((i, 'Z') for i in range(p + 1, q))
            xzx_operators = ((p, 'X'),) + z_string + ((q, 'X'),)
            yzy_operators = ((p, 'Y'),) + z_string + ((q, 'Y'),)
            hamiltonian += QubitOperator(xzx_operators, term_coefficient)
            hamiltonian += QubitOperator(yzy_operators, term_coefficient)

    # Include the Madelung constant if requested.
    if include_constant:
        hamiltonian += QubitOperator((),) * (2.8372 / grid.scale)

    # Return Hamiltonian.
    return hamiltonian
