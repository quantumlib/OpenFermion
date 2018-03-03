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

"""This module constructs Hamiltonians for the Fermi-Hubbard model."""
from __future__ import absolute_import

from openfermion.ops import FermionOperator
from openfermion.utils import (hermitian_conjugated, number_operator,
                               up_index, down_index)


def fermi_hubbard(x_dimension, y_dimension, tunneling, coulomb,
                  chemical_potential=0., magnetic_field=0.,
                  periodic=True, spinless=False,
                  particle_hole_symmetry=False,
                  up_map=up_index, down_map=down_index):
    """Return symbolic representation of a Fermi-Hubbard Hamiltonian.

    The idea of this model is that some fermions move around on a grid and the
    energy of the model depends on where the fermions are.
    The Hamiltonians of this model live on a grid of dimensions
    `x_dimension` x `y_dimension`.
    The grid can have periodic boundary conditions or not.
    In the standard Fermi-Hubbard model (which we call the "spinful" model),
    there is room for an "up" fermion and a "down" fermion at each site on the
    grid. In this model, there are a total of `2N` spin-orbitals,
    where `N = x_dimension * y_dimension` is the number of sites.
    In the spinless model, there is only one spin-orbital per site
    for a total of `N`.

    The Hamiltonian for the spinful model has the form

    .. math::

        \\begin{align}
        H = &- t \sum_{\langle i,j \\rangle} \sum_{\sigma}
                     (a^\dagger_{i, \sigma} a_{j, \sigma} +
                      a^\dagger_{j, \sigma} a_{i, \sigma})
             + U \sum_{i} a^\dagger_{i, \\uparrow} a_{i, \\uparrow}
                         a^\dagger_{j, \downarrow} a_{j, \downarrow}
            \\\\
            &- \mu \sum_i \sum_{\sigma} a^\dagger_{i, \sigma} a_{i, \sigma}
             - h \sum_i (a^\dagger_{i, \\uparrow} a_{i, \\uparrow} -
                       a^\dagger_{i, \downarrow} a_{i, \downarrow})
        \\end{align}

    where

        - The indices :math:`\langle i, j \\rangle` run over pairs
          :math:`i` and :math:`j` of sites that are connected to each other
          in the grid
        - :math:`\sigma \in \\{\\uparrow, \downarrow\\}` is the spin
        - :math:`t` is the tunneling amplitude
        - :math:`U` is the Coulomb potential
        - :math:`\mu` is the chemical potential
        - :math:`h` is the magnetic field

    One can also construct the Hamiltonian for the spinless model, which
    has the form

    .. math::

        H = - t \sum_{k=1}^{N-1} (a_k^\dagger a_{k + 1} + a_{k+1}^\dagger a_k)
            + U \sum_{k=1}^{N-1} a_k^\dagger a_k a_{k+1}^\dagger a_{k+1}
            - \mu \sum_{k=1}^N a_k^\dagger a_k.

    Args:
        x_dimension (int): The width of the grid.
        y_dimension (int): The height of the grid.
        tunneling (float): The tunneling amplitude :math:`t`.
        coulomb (float): The attractive local interaction strength :math:`U`.
        chemical_potential (float, optional): The chemical potential
            :math:`\mu` at each site. Default value is 0.
        magnetic_field (float, optional): The magnetic field :math:`h`
            at each site. Default value is 0. Ignored for the spinless case.
        periodic (bool, optional): If True, add periodic boundary conditions.
            Default is True.
        spinless (bool, optional): If True, return a spinless Fermi-Hubbard
            model. Default is False.
        particle_hole_symmetry (bool, optional): If False, the repulsion
            term corresponds to:

            .. math::

                U \sum_{k=1}^{N-1} a_k^\dagger a_k a_{k+1}^\dagger a_{k+1}

            If True, the repulsion term is replaced by:

            .. math::

                U \sum_{k=1}^{N-1} (a_k^\dagger a_k - \\frac12)
                                   (a_{k+1}^\dagger a_{k+1} - \\frac12)

            which is unchanged under a particle-hole transformation.
            Default is False

    Returns:
        hubbard_model: An instance of the FermionOperator class.
    """
    tunneling = float(tunneling)
    coulomb = float(coulomb)
    chemical_potential = float(chemical_potential)
    magnetic_field = float(magnetic_field)

    # Initialize fermion operator class.
    n_sites = x_dimension * y_dimension
    n_spin_orbitals = n_sites

    if not spinless:
        n_spin_orbitals *= 2

    hubbard_model = FermionOperator.zero()

    # Select particle-hole symmetry
    if particle_hole_symmetry:
        coulomb_shift = FermionOperator((), 0.5)
    else:
        coulomb_shift = FermionOperator.zero()

    # Loop through sites and add terms.
    for site in range(n_sites):
        # Add chemical potential to the spinless case. The magnetic field
        # doesn't contribute.
        if spinless and chemical_potential:
            hubbard_model += number_operator(
                n_spin_orbitals, site, -chemical_potential)

        # With spin, add the chemical potential and magnetic field terms.
        elif not spinless:
            hubbard_model += number_operator(
                n_spin_orbitals, up_map(site),
                -chemical_potential - magnetic_field)
            hubbard_model += number_operator(
                n_spin_orbitals, down_map(site),
                -chemical_potential + magnetic_field)

            # Add local pair interaction terms.
            operator_1 = number_operator(
                n_spin_orbitals, up_map(site)) - coulomb_shift
            operator_2 = number_operator(
                n_spin_orbitals, down_map(site)) - coulomb_shift
            hubbard_model += coulomb * operator_1 * operator_2

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
            if spinless:
                # Add Coulomb term.
                operator_1 = number_operator(
                    n_spin_orbitals, site, 1.0) - coulomb_shift
                operator_2 = number_operator(
                    n_spin_orbitals, right_neighbor, 1.0) - coulomb_shift
                hubbard_model += coulomb * operator_1 * operator_2

                # Add hopping term.
                operators = ((site, 1), (right_neighbor, 0))

            else:
                # Add hopping term.
                operators = ((up_map(site), 1),
                             (up_map(right_neighbor), 0))
                hopping_term = FermionOperator(operators, -tunneling)
                hubbard_model += hopping_term
                hubbard_model += hermitian_conjugated(hopping_term)

                operators = ((down_map(site), 1),
                             (down_map(right_neighbor), 0))

            hopping_term = FermionOperator(operators, -tunneling)
            hubbard_model += hopping_term
            hubbard_model += hermitian_conjugated(hopping_term)

        # Add transition to neighbor below.
        if site + x_dimension + 1 <= n_sites or (periodic and y_dimension > 2):
            if spinless:
                # Add Coulomb term.
                operator_1 = number_operator(
                    n_spin_orbitals, site) - coulomb_shift
                operator_2 = number_operator(
                    n_spin_orbitals, bottom_neighbor) - coulomb_shift
                hubbard_model += coulomb * operator_1 * operator_2

                # Add hopping term.
                operators = ((site, 1), (bottom_neighbor, 0))

            else:
                # Add hopping term.
                operators = ((up_map(site), 1),
                             (up_map(bottom_neighbor), 0))
                hopping_term = FermionOperator(operators, -tunneling)
                hubbard_model += hopping_term
                hubbard_model += hermitian_conjugated(hopping_term)

                operators = ((down_map(site), 1),
                             (down_map(bottom_neighbor), 0))

            hopping_term = FermionOperator(operators, -tunneling)
            hubbard_model += hopping_term
            hubbard_model += hermitian_conjugated(hopping_term)

    return hubbard_model
