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

"""This module constructs Hamiltonians for the Bose-Hubbard model."""
from __future__ import absolute_import

from openfermion.ops import BosonOperator
from openfermion.utils import (hermitian_conjugated, number_operator,
                               up_index, down_index)


def bose_hubbard(x_dimension, y_dimension, tunneling, interaction,
                 chemical_potential=0., dipole=0., periodic=True):
    r"""Return symbolic representation of a Bose-Hubbard Hamiltonian.

    In this model, bosons move around on a lattice, and the
    energy of the model depends on where the bosons are.

    The lattice is described by a 2D grid, with dimensions
    `x_dimension` x `y_dimension`. It is also possible to specify
    if the grid has periodic boundary conditions or not.

    The Hamiltonian for the Bose-Hubbard model has the form

    .. math::

        H = - t \sum_{\langle i, j \rangle} b_i^\dagger b_{j + 1}
         + \frac{U}{2} \sum_{k=1}^{N-1} b_k^\dagger b_k (b_k^\dagger b_k - 1)
         - \mu \sum_{k=1}^N b_k^\dagger b_k
         + V \sum_{\langle i, j \rangle} b_i^\dagger b_i b_j^\dagger b_j.

    where

        - The indices :math:`\langle i, j \rangle` run over pairs
          :math:`i` and :math:`j` of nodes that are connected to each other
          in the grid
        - :math:`t` is the tunneling amplitude
        - :math:`U` is the on-site interaction potential
        - :math:`\mu` is the chemical potential
        - :math:`V` is the dipole or nearest-neighbour interaction potential

    Args:
        x_dimension (int): The width of the grid.
        y_dimension (int): The height of the grid.
        tunneling (float): The tunneling amplitude :math:`t`.
        interaction (float): The attractive local interaction
            strength :math:`U`.
        chemical_potential (float, optional): The chemical potential
            :math:`\mu` at each site. Default value is 0.
        periodic (bool, optional): If True, add periodic boundary conditions.
            Default is True.
        dipole (float): The attractive dipole interaction strength :math:`V`.

    Returns:
        bose_hubbard_model: An instance of the BosonOperator class.
    """
    tunneling = float(tunneling)
    interaction = float(interaction)
    chemical_potential = float(chemical_potential)

    # Initialize boson operator class.
    n_sites = x_dimension * y_dimension
    n_modes = n_sites

    bose_hubbard_model = BosonOperator.zero()

    # Loop through sites and add terms.
    for site in range(n_sites):
        # Add local interaction potential.
        if chemical_potential:
            bose_hubbard_model += number_operator(
                n_modes, site, -chemical_potential, parity=1)

        # Add chemical potential.
        if interaction:
            int_op = number_operator(n_modes, site, interaction/2, parity=1) \
                * (number_operator(n_modes, site, parity=1)
                    - BosonOperator.identity())
            bose_hubbard_model += int_op

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
        if (right_neighbor) % x_dimension or (periodic and x_dimension > 2):
            # Add dipole interaction term.
            operator_1 = number_operator(
                n_modes, site, 1.0, parity=1)
            operator_2 = number_operator(
                n_modes, right_neighbor, 1.0, parity=1)
            bose_hubbard_model += dipole * operator_1 * operator_2

            # Add hopping term.
            operators = ((site, 1), (right_neighbor, 0))

            hopping_term = BosonOperator(operators, -tunneling)
            bose_hubbard_model += hopping_term
            bose_hubbard_model += hermitian_conjugated(hopping_term)

        # Add transition to neighbor below.
        if site + x_dimension + 1 <= n_sites or (periodic and y_dimension > 2):
            # Add dipole interaction term.
            operator_1 = number_operator(
                n_modes, site, parity=1)
            operator_2 = number_operator(
                n_modes, bottom_neighbor, parity=1)
            bose_hubbard_model += dipole * operator_1 * operator_2

            # Add hopping term.
            operators = ((site, 1), (bottom_neighbor, 0))

            hopping_term = BosonOperator(operators, -tunneling)
            bose_hubbard_model += hopping_term
            bose_hubbard_model += hermitian_conjugated(hopping_term)

    return bose_hubbard_model
