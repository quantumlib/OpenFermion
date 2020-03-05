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

"""This module constructs Hamiltonians for the multiband Fermi-Hubbard model.
"""

from collections import namedtuple

from openfermion.ops import FermionOperator
from openfermion.utils import (SpinPairs, Spin)

TunnelingParameter = namedtuple('TunnelingParameter',
                                ('edge_type', 'dofs', 'coefficient'))
InteractionParameter = namedtuple(
        'InteractionParameter',
        ('edge_type', 'dofs', 'coefficient', 'spin_pairs'))
PotentialParameter = namedtuple('PotentialParameter',
                                ('dof', 'coefficient'))


def number_operator(i, coefficient=1., particle_hole_symmetry=False):
    op = FermionOperator(((i, 1), (i, 0)), coefficient)
    if particle_hole_symmetry:
        op -= FermionOperator((), 0.5)
    return op


def interaction_operator(i, j, coefficient=1., particle_hole_symmetry=False):
    return (number_operator(
        i, coefficient, particle_hole_symmetry=particle_hole_symmetry) *
            number_operator(j, particle_hole_symmetry=particle_hole_symmetry))


def tunneling_operator(i, j, coefficient=1.):
    return (FermionOperator(((i, 1), (j, 0)), coefficient) + FermionOperator(
        ((j, 1), (i, 0)), coefficient.conjugate()))


def number_difference_operator(i, j, coefficient=1.):
    return number_operator(i, coefficient) - number_operator(j, coefficient)


class FermiHubbardModel:
    r"""A general, parameterized Fermi-Hubbard model.

    The general (AKA 'multiband') Fermi-Hubbard model has `k` degrees of
    freedom per site in a lattice.
    For a lattice with `n` sites, there are `N = k * n` spatial orbitals.
    Additionally, in what we call the "spinful" model each spatial orbital is
    associated with "up" and "down" spin orbitals, for a total of `2N` spin
    orbitals; in the spinless model, there is only one spin-orbital per site
    for a total of `N`.

    For a lattice with only one type of site and edges from each site only to
    itself and its neighbors, the Hamiltonian for the spinful model has the
    form

    .. math::

        \begin{align}
        H = &- \sum_{a < b} t_{a, b}^{(\mathrm{onsite})}
               \sum_{i} \sum_{\sigma}
                     (a^\dagger_{i, a, \sigma} a_{i, b, \sigma} +
                      a^\dagger_{i, b, \sigma} a_{i, a, \sigma})
            \\
            &- \sum_{a} t_{a, a}^{(\mathrm{nghbr})}
               \sum_{\{i, j\}} \sum_{\sigma}
                     (a^\dagger_{i, a, \sigma} a_{j, a, \sigma} +
                      a^\dagger_{j, a, \sigma} a_{i, a, \sigma})
             - \sum_{a < b} t_{a, b}^{(\mathrm{nghbr})}
               \sum_{(i, j)} \sum_{\sigma}
                     (a^\dagger_{i, a, \sigma} a_{j, b, \sigma} +
                      a^\dagger_{j, b, \sigma} a_{i, a, \sigma})
            \\
            &+ \sum_{a < b} U_{a, b}^{(\mathrm{onsite}, +)}
               \sum_{i} \sum_{\sigma}
                     n_{i, a, \sigma} n_{i, b, \sigma}
            \\
            &+ \sum_{a} U_{a, a}^{(\mathrm{nghbr}, +)}
               \sum_{\{i, j\}} \sum_{\sigma}
                     n_{i, a, \sigma} n_{j, a, \sigma}
             + \sum_{a < b} U_{a, b}^{(\mathrm{nghbr}, +)}
               \sum_{(i, j)} \sum_{\sigma}
                     n_{i, a, \sigma} n_{j, b, \sigma}
            \\
            &+ \sum_{a \leq b} U_{a, b}^{(\mathrm{onsite}, -)}
               \sum_{i} \sum_{\sigma}
                     n_{i, a, \sigma} n_{i, b, -\sigma}
            \\
            &+ \sum_{a} U_{a, a}^{(\mathrm{nghbr}, -)}
               \sum_{\{ i, j \}} \sum_{\sigma}
                     n_{i, a, \sigma} n_{j, a, -\sigma}
             + \sum_{a < b} U_{a, b}^{(\mathrm{nghbr}, -)}
               \sum_{( i, j )} \sum_{\sigma}
                     n_{i, a, \sigma} n_{j, b, -\sigma}
            \\
            &- \sum_{a} \mu_a
               \sum_i \sum_{\sigma} n_{i, a, \sigma}
            \\
            &- h \sum_{i} \sum_{a}
                \left(n_{i, a, \uparrow} - n_{i, a, \downarrow}\right)
        \end{align}

    where

        - The indices :math:`(i, j)` and :math:`\{i, j\}` run over ordered and
          unordered pairs, respectively of sites :math:`i` and :math:`j` of
          neighboring sites in the lattice,
        - :math:`a` and :math:`b` index degrees of freedom on each site,
        - :math:`\sigma \in \{\uparrow, \downarrow\}` is the spin,
        - :math:`t_{a, b}^{(\mathrm{onsite})}` is the tunneling amplitude
          between spin orbitals on the same site,
        - :math:`t_{a, b}^{(\mathrm{nghbr})}` is the tunneling amplitude
          between spin orbitals on neighboring sites,
        - :math:`U_{a, b}^{(\mathrm{onsite, \pm})}` is the Coulomb potential
          between spin orbitals on the same site with the same (+) or different
          (-) spins,
        - :math:`U_{a, b}^{(\mathrm{nghbr, \pm})}` is the Coulomb potential
          betwen spin orbitals on neighborings sites with the same (+) or
          different (-) spins,
        - :math:`\mu_{a}` is the chemical potential, and
        - :math:`h` is the magnetic field.

    One can also construct the Hamiltonian for the spinless model, which
    has the form

    .. math::

        \begin{align}
        H = &- \sum_{a < b} t_{a, b}^{(\mathrm{onsite})}
               \sum_{i}
                     (a^\dagger_{i, a} a_{i, b} +
                      a^\dagger_{i, b} a_{i, a})
            \\
            &- \sum_{a} t_{a, a}^{(\mathrm{nghbr})}
               \sum_{\{i, j\}}
                     (a^\dagger_{i, a} a_{j, a} +
                      a^\dagger_{j, a} a_{i, a})
             - \sum_{a < b} t_{a, b}^{(\mathrm{nghbr})}
               \sum_{(i, j)}
                     (a^\dagger_{i, a} a_{j, b} +
                      a^\dagger_{j, b} a_{i, a})
            \\
            &+ \sum_{a < b} U_{a, b}^{(\mathrm{onsite})}
               \sum_{i}
                     n_{i, a} n_{i, b}
            \\
            &+ \sum_{a} U_{a, a}^{(\mathrm{nghbr})}
               \sum_{\{i, j\}}
                     n_{i, a} n_{j, a}
             + \sum_{a < b} U_{a, b}^{(\mathrm{nghbr})}
               \sum_{(i, j)}
                     n_{i, a} n_{j, b}
            \\
            &- \sum_{a} \mu_a
               \sum_i n_{i, a}
        \end{align}
    """


    def __init__(self, lattice,
                 tunneling_parameters=None,
                 interaction_parameters=None,
                 potential_parameters=None,
                 magnetic_field=0.,
                 particle_hole_symmetry=False
                 ):
        r"""A Hubbard model defined on a lattice.

        Args:
            lattice (HubbardLattice): The lattice on which the model is defined.
            tunneling_parameters (Iterable[Tuple[Hashable, Tuple[int, int],
                float]], optional): The tunneling parameters.
            interaction_parameters (Iterable[Tuple[Hashable, Tuple[int, int],
                float, int?]], optional): The interaction parameters.
            potential_parameters (Iterable[Tuple[int, float]], optional): The
                potential parameters.
            magnetic_field (float, optional): The magnetic field. Default is 0.
            particle_hole_symmetry: If true, each number operator :math:`n` is
                replaced with :math:`n - 1/2`.

        Each group of parameters is specified as an iterable of tuples.

        Each tunneling parameter is a tuple ``(edge_type, dofs, coefficient)``.

        In the spinful, model, the tunneling parameter corresponds to the terms

        .. math::

            t \sum_{(i, j) \in E^{(\mathrm{edge type})}}
            \sum_{\sigma}
            \left(a_{i, a, \sigma}^{\dagger} a_{j, b, \sigma}
            + a_{j, b, \sigma}^{\dagger} a_{i, a, \sigma}\right)

        and in the spinless model to

        .. math::

            -t \sum_{(i, j) \in E^{(\mathrm{edge type})}}
            \left(a_{i, a}^{\dagger} a_{j, b}
            + a_{j, b}^{\dagger} a_{i, a}\right),

        where

            - :math:`(a, b)` is the pair of degrees
            of freedom given by ``dofs``;
            - :math:`E^{(\mathrm{edge type})}` is the set of ordered pairs of
              site indices returned by ``lattice.site_pairs_iter(edge_type, a !=
              b)``; and
            - :math:`t` is the ``coefficient``.

        Each interaction parameter is a tuple ``(edge_type, dofs,
        coefficient, spin_pairs)``. The final ``spin_pairs`` element is
        optional, and will default to ``SpinPairs.ALL``. In any case, it is
        ignored for spinless lattices.

        For example, in the spinful model if `dofs`
        indicates distinct degrees of freedom then the
        parameter corresponds to the terms

        .. math::
        U \sum_{(i, j) \in E^{(\mathrm{edge type})}} \sum_{(\sigma, \sigma')}
        n_{i, a, \sigma} n_{j, b, \sigma'}

        where

            - :math:`(a, b)` is the pair of degrees of
            freedom given by ``dofs``;
            - :math:`E^{(\mathrm{edge type})}` is the set of ordered pairs of
              site indices returned by ``lattice.site_pairs_iter(edge_type)``;
            - :math:`U` is the ``coefficient``; and
            - :math:`(\sigma, \sigma')` runs over
                - all four possible pairs of spins
                if `spin_pairs == SpinPairs.ALL`,
                - :math:`\{(\uparrow, \downarrow), (\downarrow, \uparrow)\}`
                if `spin_pairs == SpinPairs.DIFF`, and
                - :math:`\{(\uparrow, \uparrow), (\downarrow, \downarrow)\}'
                if 'spin_pairs == SpinPairs.SAME`.

        Each potential parameter is a tuple ``(dof, coefficient)``.
        For example, in the spinful model, it corresponds to the terms

        .. math::
            -\mu \sum_{i} \sum_{\sigma} n_{i, a, \sigma},

        where

            - :math:`i` runs over the sites of the lattice;
            - :math:`a` is the degree of freedom ``dof``; and
            - :math:`\mu` is the ``coefficient``.

        In the spinless model, the magnetic field is ignored.
        """

        self.lattice = lattice

        self.tunneling_parameters = self.parse_tunneling_parameters(
                tunneling_parameters)
        self.interaction_parameters = self.parse_interaction_parameters(
                interaction_parameters)
        self.potential_parameters = self.parse_potential_parameters(
                potential_parameters)
        self.magnetic_field = magnetic_field
        self.particle_hole_symmetry = particle_hole_symmetry


    def parse_tunneling_parameters(self, parameters):
        if parameters is None:
            return []
        parsed_parameters = []
        for parameter in parameters:
            parameter = TunnelingParameter(*parameter)
            self.lattice.validate_edge_type(parameter.edge_type)
            self.lattice.validate_dofs(parameter.dofs, 2)
            if ((parameter.edge_type in self.lattice.onsite_edge_types) and
                (len(set(parameter.dofs)) == 1)):
                raise ValueError('Invalid onsite tunneling parameter between '
                                 'same dof {}.'.format(parameter.dofs))
            parsed_parameters.append(parameter)
        return parsed_parameters


    def parse_interaction_parameters(self, parameters):
        if parameters is None:
            return []
        parsed_parameters = []
        for parameter in parameters:
            if len(parameter) not in (3, 4):
                raise ValueError('len(parameter) not in (3, 4)')
            spin_pairs = (SpinPairs.ALL
                          if len(parameter) < 4 else parameter[-1])
            parameter = InteractionParameter(*parameter[:3],
                                             spin_pairs=spin_pairs)
            self.lattice.validate_edge_type(parameter.edge_type)
            self.lattice.validate_dofs(parameter.dofs, 2)
            if ((len(set(parameter.dofs)) == 1) and
                (parameter.edge_type in self.lattice.onsite_edge_types) and
                (parameter.spin_pairs == SpinPairs.SAME)):
                raise ValueError('Parameter {} specifies '.format(parameter) +
                        'invalid interaction between spin orbital and itself.')
            parsed_parameters.append(parameter)
        return parsed_parameters


    def parse_potential_parameters(self, parameters):
        if parameters is None:
            return []
        parsed_parameters = []
        for parameter in parameters:
            parameter = PotentialParameter(*parameter)
            self.lattice.validate_dof(parameter.dof)
            parsed_parameters.append(parameter)
        return parsed_parameters


    def tunneling_terms(self):
        terms = FermionOperator()
        for param in self.tunneling_parameters:
            a, aa = param.dofs
            site_pairs = self.lattice.site_pairs_iter(param.edge_type, a != aa)
            for r, rr in site_pairs:
                for spin_index in self.lattice.spin_indices:
                    i = self.lattice.to_spin_orbital_index(r, a, spin_index)
                    j = self.lattice.to_spin_orbital_index(rr, aa, spin_index)
                    terms += tunneling_operator(i, j, -param.coefficient)
        return terms


    def interaction_terms(self):
        terms = FermionOperator()
        for param in self.interaction_parameters:
            a, aa = param.dofs
            for r, rr in self.lattice.site_pairs_iter(param.edge_type, a != aa):
                same_spatial_orbital = (a, r) == (aa, rr)
                for s, ss in self.lattice.spin_pairs_iter(
                        SpinPairs.DIFF if same_spatial_orbital
                        else param.spin_pairs,
                        not same_spatial_orbital):
                    i = self.lattice.to_spin_orbital_index(r, a, s)
                    j = self.lattice.to_spin_orbital_index(rr, aa, ss)
                    terms += interaction_operator(i, j, param.coefficient,
                            particle_hole_symmetry=self.particle_hole_symmetry)
        return terms


    def potential_terms(self):
        terms = FermionOperator()
        for param in self.potential_parameters:
            for site_index in self.lattice.site_indices:
                for spin_index in self.lattice.spin_indices:
                    i = self.lattice.to_spin_orbital_index(
                            site_index, param.dof, spin_index)
                    terms += number_operator(
                        i,
                        -param.coefficient,
                        particle_hole_symmetry=self.particle_hole_symmetry)
        return terms


    def field_terms(self):
        terms = FermionOperator()
        if self.lattice.spinless or not self.magnetic_field:
            return terms
        for site_index in self.lattice.site_indices:
            for dof in self.lattice.dof_indices:
                i = self.lattice.to_spin_orbital_index(site_index, dof, Spin.UP)
                j = self.lattice.to_spin_orbital_index(site_index, dof,
                                                       Spin.DOWN)
                terms += number_difference_operator(i, j, -self.magnetic_field)
        return terms


    def hamiltonian(self):
        return (self.tunneling_terms() + self.interaction_terms() +
                self.potential_terms() + self.field_terms())
