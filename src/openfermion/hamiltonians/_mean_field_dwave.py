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

"""This module constructs Hamiltonians for the BCS mean-field d-wave model."""
from __future__ import absolute_import

from openfermion.ops import FermionOperator
from openfermion.utils import (hermitian_conjugated, number_operator,
                               up_index, down_index)


def mean_field_dwave(x_dimension, y_dimension, tunneling, sc_gap,
                     chemical_potential=0., periodic=True,
                     up_map=up_index, down_map=down_index):
    """Return symbolic representation of a BCS mean-field d-wave Hamiltonian.

    The Hamiltonians of this model live on a grid of dimensions
    `x_dimension` x `y_dimension`.
    The grid can have periodic boundary conditions or not.
    Each site on the grid can have an "up" fermion and a "down" fermion.
    Therefore, there are a total of `2N` spin-orbitals,
    where `N = x_dimension * y_dimension` is the number of sites.

    The Hamiltonian for this model has the form

    .. math::

        \\begin{align}
        H = &- t \sum_{\langle i,j \\rangle} \sum_\sigma
                (a^\dagger_{i, \sigma} a_{j, \sigma} +
                 a^\dagger_{j, \sigma} a_{i, \sigma})
            - \mu \sum_i \sum_{\sigma} a^\dagger_{i, \sigma} a_{i, \sigma}
            \\\\
            &- \sum_{\langle i,j \\rangle} \Delta_{ij}
              (a^\dagger_{i, \\uparrow} a^\dagger_{j, \downarrow} -
               a^\dagger_{i, \downarrow} a^\dagger_{j, \\uparrow} +
               a_{j, \downarrow} a_{i, \\uparrow} -
               a_{j, \\uparrow} a_{i, \downarrow})
        \\end{align}

    where

        - The indices :math:`\langle i, j \\rangle` run over pairs
          :math:`i` and :math:`j` of sites that are connected to each other
          in the grid
        - :math:`\sigma \in \\{\\uparrow, \downarrow\\}` is the spin
        - :math:`t` is the tunneling amplitude
        - :math:`\Delta_{ij}` is equal to :math:`+\Delta/2` for
          horizontal edges and :math:`-\Delta/2` for vertical edges,
          where :math:`\Delta` is the superconducting gap.
        - :math:`\mu` is the chemical potential

    Args:
        x_dimension (int): The width of the grid.
        y_dimension (int): The height of the grid.
        tunneling (float): The tunneling amplitude :math:`t`.
        sc_gap (float): The superconducting gap :math:`\Delta`
        chemical_potential (float, optional): The chemical potential
            :math:`\mu` at each site. Default value is 0.
        periodic (bool, optional): If True, add periodic boundary conditions.
            Default is True.

    Returns:
        mean_field_dwave_model: An instance of the FermionOperator class.
    """
    # Initialize fermion operator class.
    n_sites = x_dimension * y_dimension
    n_spin_orbitals = 2 * n_sites
    mean_field_dwave_model = FermionOperator()

    # Loop through sites and add terms.
    for site in range(n_sites):
        # Add chemical potential
        mean_field_dwave_model += number_operator(
            n_spin_orbitals, up_map(site), -chemical_potential)
        mean_field_dwave_model += number_operator(
            n_spin_orbitals, down_map(site), -chemical_potential)

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
            operators = ((up_map(site), 1), (up_map(right_neighbor), 0))
            hopping_term = FermionOperator(operators, -tunneling)
            mean_field_dwave_model += hopping_term
            mean_field_dwave_model += hermitian_conjugated(hopping_term)
            # Add spin-down hopping term
            operators = ((down_map(site), 1),
                         (down_map(right_neighbor), 0))
            hopping_term = FermionOperator(operators, -tunneling)
            mean_field_dwave_model += hopping_term
            mean_field_dwave_model += hermitian_conjugated(hopping_term)

            # Add pairing term
            operators = ((up_map(site), 1),
                         (down_map(right_neighbor), 1))
            pairing_term = FermionOperator(operators, sc_gap / 2.)
            operators = ((down_map(site), 1),
                         (up_map(right_neighbor), 1))
            pairing_term += FermionOperator(operators, -sc_gap / 2.)
            mean_field_dwave_model -= pairing_term
            mean_field_dwave_model -= hermitian_conjugated(pairing_term)

        # Add transition to neighbor below.
        if site + x_dimension + 1 <= n_sites or (periodic and y_dimension > 2):
            # Add spin-up hopping term.
            operators = ((up_map(site), 1), (up_map(bottom_neighbor), 0))
            hopping_term = FermionOperator(operators, -tunneling)
            mean_field_dwave_model += hopping_term
            mean_field_dwave_model += hermitian_conjugated(hopping_term)
            # Add spin-down hopping term
            operators = ((down_map(site), 1),
                         (down_map(bottom_neighbor), 0))
            hopping_term = FermionOperator(operators, -tunneling)
            mean_field_dwave_model += hopping_term
            mean_field_dwave_model += hermitian_conjugated(hopping_term)

            # Add pairing term
            operators = ((up_map(site), 1),
                         (down_map(bottom_neighbor), 1))
            pairing_term = FermionOperator(operators, -sc_gap / 2.)
            operators = ((down_map(site), 1),
                         (up_map(bottom_neighbor), 1))
            pairing_term += FermionOperator(operators, sc_gap / 2.)
            mean_field_dwave_model -= pairing_term
            mean_field_dwave_model -= hermitian_conjugated(pairing_term)
    # Return.
    return mean_field_dwave_model
