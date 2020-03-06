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

"""Tests for multiband Hubbard model module."""

from collections import defaultdict
import itertools
import random
import pytest


from openfermion.hamiltonians import fermi_hubbard, FermiHubbardModel
from openfermion.utils import HubbardSquareLattice, SpinPairs
from openfermion.hamiltonians._general_hubbard import InteractionParameter


def fermi_hubbard_from_general(x_dimension, y_dimension, tunneling, coulomb,
                  chemical_potential=0.,
                  periodic=True, spinless=False, magnetic_field=0):
    lattice = HubbardSquareLattice(x_dimension,
                                   y_dimension,
                                   periodic=periodic,
                                   spinless=spinless)
    interaction_edge_type = 'neighbor' if spinless else 'onsite'
    model = FermiHubbardModel(lattice,
                              tunneling_parameters=(('neighbor', (0, 0),
                                                     tunneling),),
                              interaction_parameters=((interaction_edge_type,
                                                       (0, 0), coulomb),),
                              potential_parameters=((0, chemical_potential),),
                              magnetic_field=magnetic_field)
    return model.hamiltonian()


@pytest.mark.parametrize(
    'x_dimension,y_dimension,tunneling,coulomb,' +
    'chemical_potential,spinless,periodic,magnetic_field',
    itertools.product(range(1, 4), range(1, 4), (random.uniform(0, 2.),),
                      (random.uniform(0, 2.),), (random.uniform(0, 2.),),
                      (True, False), (True, False), (random.uniform(-1, 1),)))
def test_fermi_hubbard_square_special_general_equivalence(
        x_dimension, y_dimension, tunneling, coulomb,
        chemical_potential, spinless, periodic, magnetic_field):
    hubbard_model_special = fermi_hubbard(
            x_dimension, y_dimension, tunneling, coulomb,
            chemical_potential=chemical_potential, spinless=spinless,
            periodic=periodic, magnetic_field=magnetic_field)
    hubbard_model_general = fermi_hubbard_from_general(
            x_dimension, y_dimension, tunneling, coulomb,
            chemical_potential=chemical_potential, spinless=spinless,
            periodic=periodic, magnetic_field=magnetic_field)
    assert hubbard_model_special == hubbard_model_general

def random_parameters(lattice, probability=0.5, distinguish_edges=False):
    parameters = {}
    edge_types = (('onsite', 'horizontal_neighbor',
                   'vertical_neighbor') if distinguish_edges else
                  ('onsite', 'neighbor'))

    parameters['tunneling_parameters'] = [
            (edge_type, dofs, random.uniform(-1, 1))
            for edge_type in edge_types
            for dofs in lattice.dof_pairs_iter(edge_type == 'onsite')
            if random.random() <= probability]

    possible_spin_pairs = (SpinPairs.ALL,) if lattice.spinless else (
        SpinPairs.SAME, SpinPairs.DIFF)
    parameters['interaction_parameters'] = [
        (edge_type, dofs, random.uniform(-1, 1), spin_pairs)
        for edge_type in edge_types for spin_pairs in possible_spin_pairs
        for dofs in lattice.dof_pairs_iter(edge_type == 'onsite' and spin_pairs
                                           in (SpinPairs.ALL, SpinPairs.SAME))
        if random.random() <= probability
    ]

    parameters['potential_parameters'] = [
            (dof, random.uniform(-1, 1))
            for dof in lattice.dof_indices
            if random.random() <= probability]

    if random.random() <= probability:
        parameters['magnetic_field'] = random.uniform(-1, 1)

    return parameters


def test_fermi_hubbard_default_parameters():
    lattice = HubbardSquareLattice(3, 3)
    model = FermiHubbardModel(lattice)
    assert model.tunneling_parameters == []
    assert model.interaction_parameters == []
    assert model.potential_parameters == []
    assert model.magnetic_field == 0


def test_fermi_hubbard_bad_parameters():
    lattice = HubbardSquareLattice(3, 3)
    with pytest.raises(ValueError):
        tunneling_parameters = [('onsite', (0, 0), 1)]
        FermiHubbardModel(lattice, tunneling_parameters=tunneling_parameters)

    with pytest.raises(ValueError):
        FermiHubbardModel(lattice, interaction_parameters=[(0, 0)])
    with pytest.raises(ValueError):
        FermiHubbardModel(lattice, interaction_parameters=[(0,) * 5])
    with pytest.raises(ValueError):
        interaction_parameters = [('onsite', (0, 0), 1, SpinPairs.SAME)]
        FermiHubbardModel(lattice,
                          interaction_parameters=interaction_parameters)


lattices = [
    HubbardSquareLattice(random.randrange(3, 10),
                         random.randrange(3, 10),
                         periodic=periodic,
                         spinless=spinless) for periodic in (False, True)
    for spinless in (False, True) for _ in range(2)
]


@pytest.mark.parametrize('lattice,parameters,distinguish_edges', [
    (lattice, random_parameters(lattice, distinguish_edges=distinguish_edges),
     distinguish_edges)
    for lattice in lattices
    for _ in range(2)
    for distinguish_edges in (True, False)
    ])
def test_fermi_hubbard_square_lattice_random_parameters(
        lattice, parameters, distinguish_edges):
    model = FermiHubbardModel(lattice, **parameters)
    hamiltonian = model.hamiltonian()
    terms_per_parameter = defaultdict(int)
    for term, coefficient in hamiltonian.terms.items():
        spin_orbitals = set(i for i, _ in term)
        if len(spin_orbitals) == 2:
            (i, a, s), (ii, aa, ss) = (
                    lattice.from_spin_orbital_index(i) for i in spin_orbitals)
            edge_type = ({
                (0, 0): 'onsite',
                (0, 1): 'vertical_neighbor',
                (1, 0): 'horizontal_neighbor'
            } if distinguish_edges else {
                (0, 0): 'onsite',
                (0, 1): 'neighbor',
                (1, 0): 'neighbor'
            })[lattice.delta_mag(i, ii, True)]
            dofs = tuple(sorted((a, aa)))
            if len(term) == 2:
                parameter = (edge_type, dofs, -coefficient)
                assert parameter in parameters['tunneling_parameters']
                terms_per_parameter['tunneling', parameter] += 1
            else:
                assert len(term) == 4
                spin_pairs = (SpinPairs.ALL if lattice.spinless else
                              (SpinPairs.SAME if s == ss else SpinPairs.DIFF))
                parameter = (edge_type, dofs, coefficient, spin_pairs)
                assert parameter in parameters['interaction_parameters']
                terms_per_parameter['interaction', parameter] += 1
        else:
            assert len(term) == 2
            assert len(spin_orbitals) == 1
            spin_orbital = spin_orbitals.pop()
            _, dof, spin_index = lattice.from_spin_orbital_index(spin_orbital)
            potential_coefficient = -coefficient
            if not lattice.spinless:
                (-1)**spin_index
                potential_coefficient -= (((-1)**spin_index) *
                                          parameters.get('magnetic_field', 0))
            if not potential_coefficient:
                continue
            parameter = (dof, potential_coefficient)
            assert parameter in parameters['potential_parameters']
            terms_per_parameter['potential', parameter] += 1
    edge_type_to_n_site_pairs = {
        'onsite': lattice.n_sites,
        'neighbor': lattice.n_neighbor_pairs(False),
        'vertical_neighbor': lattice.n_vertical_neighbor_pairs(False),
        'horizontal_neighbor': lattice.n_horizontal_neighbor_pairs(False)
    }
    for (term_type, parameter), n_terms in terms_per_parameter.items():
        if term_type == 'potential':
            assert n_terms == lattice.n_sites * lattice.n_spin_values
            continue
        n_site_pairs = edge_type_to_n_site_pairs[parameter[0]]
        if term_type == 'tunneling':
            assert n_terms == 2 * n_site_pairs * lattice.n_spin_values
        else:
            assert term_type == 'interaction'
            parameter = InteractionParameter(*parameter)
            expected_n_terms = n_site_pairs
            if parameter.edge_type == 'neighbor':
                expected_n_terms *= len(set(parameter.dofs))
            if not lattice.spinless:
                assert parameter.spin_pairs != SpinPairs.ALL
                if (parameter.edge_type == 'onsite' and
                        parameter.spin_pairs == SpinPairs.DIFF):
                    expected_n_terms *= len(set(parameter.dofs))
                else:
                    expected_n_terms *= 2
            assert n_terms == expected_n_terms
