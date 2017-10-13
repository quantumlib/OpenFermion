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

"""This module constructions Hamiltonians  the Fermi-Hubbard model.

The idea is that some fermions move around on a grid and the energy of the
model depends on where the fermions are. In the standard Fermi-Hubbard model
(which we call the "spinful" model), there is room  an "up" fermion and a
"down" fermion at each site on the grid. Accordingly, the Hamiltonian is

H = - tunneling sum_{<i,j>} sum_sigma (a^dagger_{i, sigma} a_{j, sigma}
  + a^dagger_{j, sigma} a_{i, sigma})
  + coulomb sum_{i} a^dagger_{i, up} a_{i, up} a^dagger_{j, down} a_{j, down}
  + chemical_potential sum_i (a^dagger_{i, up} a_{i, up}
  + a^dagger_{i, down} a_{i, down})
  + magnetic_field sum_i (a^dagger_{i, up} a_{i, up}
  - a^dagger_{i, down} a_{i, down}).

There are N sites and 2*N spin-orbitals. The operators a^dagger_i and a_i are
fermionic creation and annihilation operators. One can transm these
operators to qubit operator using the Jordan-Wigner transmation:

a^dagger_j = 0.5 (X - i Y) prod_{k = 1}^{j - 1} Z_k
a_j = 0.5 (X + i Y) prod_{k = 1}^{j - 1} Z_k

The code also allows one to construct the spinless Fermi-Hubbard model,
H = - tunneling sum_{k=1}^{N-1} (a_k^dagger a_{k + 1} + a_{k+1}^dagger a_k)
  + coulomb sum_{k=1}^{N-1} a_k^dagger a_k a_{k+1}^dagger a_{k+1}
  + magnetic_field sum_{k=1}^N (-1)^k a_k^dagger a_k
  - chemical_potential sum_{k=1}^N a_k^dagger a_k.

These Hamiltonians live a square lattice which has dimensions of
x_dimension by y_dimension. They can have periodic boundary conditions or not.
"""
from __future__ import absolute_import

from openfermion.ops import (FermionOperator,
                             hermitian_conjugated,
                             number_operator)


# Function to return up-orbital index given orbital index.
def up(index):
    return 2 * index


# Function to return down-orbital index given orbital index.
def down(index):
    return 2 * index + 1


def fermi_hubbard(x_dimension, y_dimension, tunneling, coulomb,
                  chemical_potential=None, magnetic_field=None,
                  periodic=True, spinless=False):
    """Return symbolic representation of a Fermi-Hubbard Hamiltonian.

    Args:
        x_dimension: An integer giving the number of sites in width.
        y_dimension: An integer giving the number of sites in height.
        tunneling: A float giving the tunneling amplitude.
        coulomb: A float giving the attractive local interaction strength.
        chemical_potential: An optional float giving the potential of each
            site. Default value is None.
        magnetic_field: An optional float giving a magnetic field at each
            site. Default value is None.
        periodic: If True, add periodic boundary conditions.
        spinless: An optional Boolean. If False, each site has spin up
            orbitals and spin down orbitals. If True, return a spinless
            Fermi-Hubbard model.
        verbose: An optional Boolean. If True, print all second quantized
            terms.

    Returns:
        hubbard_model: An instance of the FermionOperator class.
    """
    # Initialize fermion operator class.
    n_sites = x_dimension * y_dimension
    if spinless:
        n_spin_orbitals = n_sites
    else:
        n_spin_orbitals = 2 * n_sites
    hubbard_model = FermionOperator((), 0.0)

    # Loop through sites and add terms.
    for site in range(n_sites):

        # Add chemical potential and magnetic field terms.
        if chemical_potential and spinless:
            x_index = site % x_dimension
            y_index = (site - 1) // x_dimension
            sign = (-1.) ** (x_index + y_index)
            coefficient = sign * chemical_potential
            hubbard_model += number_operator(
                n_spin_orbitals, site, coefficient)

        if chemical_potential and not spinless:
            coefficient = -1. * chemical_potential
            hubbard_model += number_operator(
                n_spin_orbitals, up(site), coefficient)
            hubbard_model += number_operator(
                n_spin_orbitals, down(site), coefficient)

        if magnetic_field and not spinless:
            coefficient = magnetic_field
            hubbard_model += number_operator(
                n_spin_orbitals, up(site), -coefficient)
            hubbard_model += number_operator(
                n_spin_orbitals, down(site), coefficient)

        # Add local pair interaction terms.
        if not spinless:
            operators = ((up(site), 1), (up(site), 0),
                         (down(site), 1), (down(site), 0))
            hubbard_model += FermionOperator(operators, coulomb)

        # Index coupled orbitals.
        right_neighbor = site + 1
        bottom_neighbor = site + x_dimension

        # Account for periodic boundaries.
        if periodic:
            if (x_dimension > 2) and ((site + 1) % x_dimension == 0):
                right_neighbor -= x_dimension
            if (y_dimension > 2) and (site + x_dimension + 1 > n_sites):
                bottom_neighbor -= x_dimension * y_dimension

        # Add transition to neighbor on right.
        if (site + 1) % x_dimension or (periodic and x_dimension > 2):
            if spinless:
                # Add Coulomb term.
                operators = ((site, 1), (site, 0),
                             (right_neighbor, 1), (right_neighbor, 0))
                hubbard_model += FermionOperator(operators, coulomb)

                # Add hopping term.
                operators = ((site, 1), (right_neighbor, 0))
                hopping_term = FermionOperator(operators, -tunneling)
                hubbard_model += hopping_term
                hubbard_model += hermitian_conjugated(hopping_term)
            else:
                # Add hopping term.
                operators = ((up(site), 1), (up(right_neighbor), 0))
                hopping_term = FermionOperator(operators, -tunneling)
                hubbard_model += hopping_term
                hubbard_model += hermitian_conjugated(hopping_term)
                operators = ((down(site), 1), (down(right_neighbor), 0))
                hopping_term = FermionOperator(operators, -tunneling)
                hubbard_model += hopping_term
                hubbard_model += hermitian_conjugated(hopping_term)

        # Add transition to neighbor below.
        if site + x_dimension + 1 <= n_sites or (periodic and y_dimension > 2):
            if spinless:
                # Add Coulomb term.
                operators = ((site, 1), (site, 0),
                             (bottom_neighbor, 1), (bottom_neighbor, 0))
                hubbard_model += FermionOperator(operators, coulomb)

                # Add hopping term.
                operators = ((site, 1), (bottom_neighbor, 0))
                hopping_term = FermionOperator(operators, -tunneling)
                hubbard_model += hopping_term
                hubbard_model += hermitian_conjugated(hopping_term)
            else:
                # Add hopping term.
                operators = ((up(site), 1), (up(bottom_neighbor), 0))
                hopping_term = FermionOperator(operators, -tunneling)
                hubbard_model += hopping_term
                hubbard_model += hermitian_conjugated(hopping_term)
                operators = ((down(site), 1), (down(bottom_neighbor), 0))
                hopping_term = FermionOperator(operators, -tunneling)
                hubbard_model += hopping_term
                hubbard_model += hermitian_conjugated(hopping_term)

    # Return.
    return hubbard_model
