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

"""This module contains constructions for mean-field Hamiltonian models.

The mean-field d-wave model has the form

H = - tunneling sum_{<i,j>} sum_sigma (a^dagger_{i, sigma} a_{j, sigma}
  + a^dagger_{j, sigma} a_{i, sigma})
  - sum_{<i,j>} Delta_{ij}
  (a^dagger_{i, up} a^dagger_{j, down} - a^dagger_{i, down} a^dagger_{j, up}
  + a_{j, down} a_{i, up} - a_{j, up} a_{i, down})

where Delta_{ij} = +sc_gap/2 for horizontal edges and -sc_gap/2 for vertical
edges.

There are N sites and 2*N spin-orbitals. The operators a^dagger_i and a_i are
fermionic creation and annihilation operators. One can transform these
operators to qubit operator using the Jordan-Wigner transmation:

a^dagger_j = 0.5 (X - i Y) prod_{k = 1}^{j - 1} Z_k
a_j = 0.5 (X + i Y) prod_{k = 1}^{j - 1} Z_k

These Hamiltonians live on a square lattice which has dimensions of
x_dimension by y_dimension. They can have periodic boundary conditions or not.
"""
from __future__ import absolute_import

from openfermion.ops import (FermionOperator,
                             hermitian_conjugated)
from openfermion.utils._hubbard import up, down


def meanfield_dwave(x_dimension, y_dimension, tunneling, sc_gap,
                    periodic=True, verbose=False):
    """Return symbolic representation of a BCS mean-field d-wave Hamiltonian.

    Args:
        x_dimension: An integer giving the number of sites in width.
        y_dimension: An integer giving the number of sites in height.
        tunneling: A float giving the tunneling amplitude.
        sc_gap: A float giving the magnitude of the superconducting gap.
        periodic: If True, add periodic boundary conditions.
        verbose: An optional Boolean. If True, print all second quantized
            terms.

    Returns:
        meanfield_dwave_model: An instance of the FermionOperator class.
    """
    # Initialize fermion operator class.
    n_sites = x_dimension * y_dimension
    n_spin_orbitals = 2 * n_sites
    meanfield_dwave_model = FermionOperator()

    # Loop through sites and add terms.
    for site in range(n_sites):
        # Index coupled orbitals.
        right_neighbor = site + 1
        bottom_neighbor = site + x_dimension
        # Account for periodic boundaries.
        if periodic:
            if (x_dimension > 2) and ((site + 1) % x_dimension == 0):
                right_neighbor -= x_dimension
            if (y_dimension > 2) and (site + x_dimension + 1 > n_sites):
                bottom_neighbor -= x_dimension * y_dimension

        # Add transition to neighbor on right
        if (site + 1) % x_dimension or (periodic and x_dimension > 2):
            # Add spin-up hopping term.
            operators = ((up(site), 1.), (up(right_neighbor), 0.))
            hopping_term = FermionOperator(operators, -tunneling)
            meanfield_dwave_model += hopping_term
            meanfield_dwave_model += hermitian_conjugated(hopping_term)
            # Add spin-down hopping term
            operators = ((down(site), 1.), (down(right_neighbor), 0.))
            hopping_term = FermionOperator(operators, -tunneling)
            meanfield_dwave_model += hopping_term
            meanfield_dwave_model += hermitian_conjugated(hopping_term)

            # Add pairing term
            operators = ((up(site), 1.), (down(right_neighbor), 1.))
            pairing_term = FermionOperator(operators, sc_gap / 2.)
            operators = ((down(site), 1.), (up(right_neighbor), 1.))
            pairing_term += FermionOperator(operators, -sc_gap / 2.)
            meanfield_dwave_model -= pairing_term
            meanfield_dwave_model -= hermitian_conjugated(pairing_term)

        # Add transition to neighbor below.
        if site + x_dimension + 1 <= n_sites or (periodic and y_dimension > 2):
            # Add spin-up hopping term.
            operators = ((up(site), 1.), (up(bottom_neighbor), 0.))
            hopping_term = FermionOperator(operators, -tunneling)
            meanfield_dwave_model += hopping_term
            meanfield_dwave_model += hermitian_conjugated(hopping_term)
            # Add spin-down hopping term
            operators = ((down(site), 1.), (down(bottom_neighbor), 0.))
            hopping_term = FermionOperator(operators, -tunneling)
            meanfield_dwave_model += hopping_term
            meanfield_dwave_model += hermitian_conjugated(hopping_term)

            # Add pairing term
            operators = ((up(site), 1.), (down(bottom_neighbor), 1.))
            pairing_term = FermionOperator(operators, -sc_gap / 2.)
            operators = ((down(site), 1.), (up(bottom_neighbor), 1.))
            pairing_term += FermionOperator(operators, sc_gap / 2.)
            meanfield_dwave_model -= pairing_term
            meanfield_dwave_model -= hermitian_conjugated(pairing_term)
    # Return.
    return meanfield_dwave_model
