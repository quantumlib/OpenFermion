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

"""This module constructs Hamiltonians for the Fermi- and Bose-Hubbard models.
"""

from openfermion.ops import BosonOperator, FermionOperator, down_index, up_index
from openfermion.utils import number_operator


def fermi_hubbard(x_dimension, y_dimension, tunneling, coulomb,
                  chemical_potential=0., magnetic_field=0.,
                  periodic=True, spinless=False,
                  particle_hole_symmetry=False):
    r"""Return symbolic representation of a Fermi-Hubbard Hamiltonian.

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

        \begin{align}
        H = &- t \sum_{\langle i,j \rangle} \sum_{\sigma}
                     (a^\dagger_{i, \sigma} a_{j, \sigma} +
                      a^\dagger_{j, \sigma} a_{i, \sigma})
             + U \sum_{i} a^\dagger_{i, \uparrow} a_{i, \uparrow}
                         a^\dagger_{i, \downarrow} a_{i, \downarrow}
            \\
            &- \mu \sum_i \sum_{\sigma} a^\dagger_{i, \sigma} a_{i, \sigma}
             - h \sum_i (a^\dagger_{i, \uparrow} a_{i, \uparrow} -
                       a^\dagger_{i, \downarrow} a_{i, \downarrow})
        \end{align}

    where

        - The indices :math:`\langle i, j \rangle` run over pairs
          :math:`i` and :math:`j` of sites that are connected to each other
          in the grid
        - :math:`\sigma \in \{\uparrow, \downarrow\}` is the spin
        - :math:`t` is the tunneling amplitude
        - :math:`U` is the Coulomb potential
        - :math:`\mu` is the chemical potential
        - :math:`h` is the magnetic field

    One can also construct the Hamiltonian for the spinless model, which
    has the form

    .. math::

        H = - t \sum_{\langle i, j \rangle} (a^\dagger_i a_j + a^\dagger_j a_i)
            + U \sum_{\langle i, j \rangle} a^\dagger_i a_i a^\dagger_j a_j
            - \mu \sum_i a_i^\dagger a_i.

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

                U \sum_{k=1}^{N-1} (a_k^\dagger a_k - \frac12)
                                   (a_{k+1}^\dagger a_{k+1} - \frac12)

            which is unchanged under a particle-hole transformation.
            Default is False

    Returns:
        hubbard_model: An instance of the FermionOperator class.
    """
    if spinless:
        return _spinless_fermi_hubbard_model(
                x_dimension, y_dimension, tunneling, coulomb,
                chemical_potential, magnetic_field,
                periodic, particle_hole_symmetry)
    else:
        return _spinful_fermi_hubbard_model(
                x_dimension, y_dimension, tunneling, coulomb,
                chemical_potential, magnetic_field,
                periodic, particle_hole_symmetry)


def _spinful_fermi_hubbard_model(
        x_dimension, y_dimension, tunneling, coulomb,
        chemical_potential, magnetic_field,
        periodic, particle_hole_symmetry):

    # Initialize operator.
    n_sites = x_dimension * y_dimension
    n_spin_orbitals = 2 * n_sites
    hubbard_model = FermionOperator()

    # Loop through sites and add terms.
    for site in range(n_sites):

        # Get indices of right and bottom neighbors
        right_neighbor = _right_neighbor(
                site, x_dimension, y_dimension, periodic)
        bottom_neighbor = _bottom_neighbor(
                site, x_dimension, y_dimension, periodic)

        # Avoid double-counting edges when one of the dimensions is 2
        # and the system is periodic
        if x_dimension == 2 and periodic and site % 2 == 1:
            right_neighbor = None
        if y_dimension == 2 and periodic and site >= x_dimension:
            bottom_neighbor = None

        # Add hopping terms with neighbors to the right and bottom.
        if right_neighbor is not None:
            hubbard_model += _hopping_term(
                    up_index(site), up_index(right_neighbor), -tunneling)
            hubbard_model += _hopping_term(
                    down_index(site), down_index(right_neighbor), -tunneling)
        if bottom_neighbor is not None:
            hubbard_model += _hopping_term(
                    up_index(site), up_index(bottom_neighbor), -tunneling)
            hubbard_model += _hopping_term(
                    down_index(site), down_index(bottom_neighbor), -tunneling)

        # Add local pair Coulomb interaction terms.
        hubbard_model += _coulomb_interaction_term(
                n_spin_orbitals, up_index(site), down_index(site), coulomb,
                particle_hole_symmetry)

        # Add chemical potential and magnetic field terms.
        hubbard_model += number_operator(
                n_spin_orbitals, up_index(site),
                -chemical_potential - magnetic_field)
        hubbard_model += number_operator(
                n_spin_orbitals, down_index(site),
                -chemical_potential + magnetic_field)

    return hubbard_model


def _spinless_fermi_hubbard_model(
        x_dimension, y_dimension, tunneling, coulomb,
        chemical_potential, magnetic_field,
        periodic, particle_hole_symmetry):

    # Initialize operator.
    n_sites = x_dimension * y_dimension
    hubbard_model = FermionOperator()

    # Loop through sites and add terms.
    for site in range(n_sites):

        # Get indices of right and bottom neighbors
        right_neighbor = _right_neighbor(
                site, x_dimension, y_dimension, periodic)
        bottom_neighbor = _bottom_neighbor(
                site, x_dimension, y_dimension, periodic)

        # Avoid double-counting edges when one of the dimensions is 2
        # and the system is periodic
        if x_dimension == 2 and periodic and site % 2 == 1:
            right_neighbor = None
        if y_dimension == 2 and periodic and site >= x_dimension:
            bottom_neighbor = None

        # Add terms that couple with neighbors to the right and bottom.
        if right_neighbor is not None:
            # Add hopping term
            hubbard_model += _hopping_term(site, right_neighbor, -tunneling)
            # Add local Coulomb interaction term
            hubbard_model += _coulomb_interaction_term(
                    n_sites, site, right_neighbor, coulomb,
                    particle_hole_symmetry)
        if bottom_neighbor is not None:
            # Add hopping term
            hubbard_model += _hopping_term(site, bottom_neighbor, -tunneling)
            # Add local Coulomb interaction term
            hubbard_model += _coulomb_interaction_term(
                    n_sites, site, bottom_neighbor, coulomb,
                    particle_hole_symmetry)

        # Add chemical potential. The magnetic field doesn't contribute.
        hubbard_model += number_operator(n_sites, site, -chemical_potential)

    return hubbard_model


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

        H = - t \sum_{\langle i, j \rangle} (b_i^\dagger b_j + b_j^\dagger b_i)
         + V \sum_{\langle i, j \rangle} b_i^\dagger b_i b_j^\dagger b_j
         + \frac{U}{2} \sum_i b_i^\dagger b_i (b_i^\dagger b_i - 1)
         - \mu \sum_i b_i^\dagger b_i.

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

    # Initialize operator.
    n_sites = x_dimension * y_dimension
    hubbard_model = BosonOperator()

    # Loop through sites and add terms.
    for site in range(n_sites):

        # Get indices of right and bottom neighbors
        right_neighbor = _right_neighbor(
                site, x_dimension, y_dimension, periodic)
        bottom_neighbor = _bottom_neighbor(
                site, x_dimension, y_dimension, periodic)

        # Avoid double-counting edges when one of the dimensions is 2
        # and the system is periodic
        if x_dimension == 2 and periodic and site % 2 == 1:
            right_neighbor = None
        if y_dimension == 2 and periodic and site >= x_dimension:
            bottom_neighbor = None

        # Add terms that couple with neighbors to the right and bottom.
        if right_neighbor is not None:
            # Add hopping term
            hubbard_model += _hopping_term(
                    site, right_neighbor, -tunneling, bosonic=True)
            # Add local Coulomb interaction term
            hubbard_model += _coulomb_interaction_term(
                    n_sites, site, right_neighbor, dipole,
                    particle_hole_symmetry=False,
                    bosonic=True)
        if bottom_neighbor is not None:
            # Add hopping term
            hubbard_model += _hopping_term(
                    site, bottom_neighbor, -tunneling, bosonic=True)
            # Add local Coulomb interaction term
            hubbard_model += _coulomb_interaction_term(
                    n_sites, site, bottom_neighbor, dipole,
                    particle_hole_symmetry=False,
                    bosonic=True)

        # Add on-site interaction.
        hubbard_model += (
            number_operator(n_sites, site, 0.5 * interaction, parity=1)
            * (number_operator(n_sites, site, parity=1) - BosonOperator(()))
        )

        # Add chemical potential.
        hubbard_model += number_operator(
                n_sites, site, -chemical_potential, parity=1)

    return hubbard_model


def _hopping_term(i, j, coefficient, bosonic=False):
    op_class = BosonOperator if bosonic else FermionOperator
    hopping_term = op_class(((i, 1), (j, 0)), coefficient)
    hopping_term += op_class(((j, 1), (i, 0)), coefficient.conjugate())
    return hopping_term


def _coulomb_interaction_term(
        n_sites, i, j, coefficient, particle_hole_symmetry, bosonic=False):
    op_class = BosonOperator if bosonic else FermionOperator
    number_operator_i = number_operator(n_sites, i, parity=2*bosonic - 1)
    number_operator_j = number_operator(n_sites, j, parity=2*bosonic - 1)
    if particle_hole_symmetry:
        number_operator_i -= op_class((), 0.5)
        number_operator_j -= op_class((), 0.5)
    return coefficient * number_operator_i * number_operator_j


def _right_neighbor(site, x_dimension, y_dimension, periodic):
    if x_dimension == 1:
        return None
    if (site + 1) % x_dimension == 0:
        if periodic:
            return site + 1 - x_dimension
        else:
            return None
    return site + 1


def _bottom_neighbor(site, x_dimension, y_dimension, periodic):
    if y_dimension == 1:
        return None
    if site + x_dimension + 1 > x_dimension*y_dimension:
        if periodic:
            return site + x_dimension - x_dimension*y_dimension
        else:
            return None
    return site + x_dimension
