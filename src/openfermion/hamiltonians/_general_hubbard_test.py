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
import pytest
import random


from openfermion.hamiltonians import fermi_hubbard, FermiHubbardModel
from openfermion.utils import HubbardSquareLattice, SpinPairs
from openfermion.hamiltonians._general_hubbard import InteractionParameter


def fermi_hubbard_from_general(x_dimension, y_dimension, tunneling, coulomb,
                  chemical_potential=0.,
                  periodic=True, spinless=False):
    lattice = HubbardSquareLattice(x_dimension, y_dimension, 
                             periodic=periodic, spinless=spinless)
    interaction_edge_type = 'neighbor' if spinless else 'onsite'
    model = FermiHubbardModel(lattice, 
            tunneling_parameters=(('neighbor', (0, 0), tunneling),),
            interaction_parameters=((interaction_edge_type, (0, 0), coulomb),),
            potential_parameters=((0, chemical_potential),))
    return model.hamiltonian()


@pytest.mark.parametrize(
        'x_dimension,y_dimension,tunneling,coulomb,' + 
        'chemical_potential,spinless,periodic',
        itertools.product(
            range(1, 4), range(1, 4),
            (random.uniform(0, 2.),), (random.uniform(0, 2.),),
            (random.uniform(0, 2.),), (True, False), (True, False))
        )
def test_fermi_hubbard_square_special_general_equivalance(
        x_dimension, y_dimension, tunneling, coulomb,
        chemical_potential, spinless, periodic):
    hubbard_model_special = fermi_hubbard(
            y_dimension, x_dimension, tunneling, coulomb,
            chemical_potential=chemical_potential, spinless=spinless,
            periodic=periodic)
    hubbard_model_general = fermi_hubbard_from_general(
            x_dimension, y_dimension, tunneling, coulomb,
            chemical_potential=chemical_potential, spinless=spinless,
            periodic=periodic)
    print(x_dimension, y_dimension)
    print(len(hubbard_model_special.terms))
    print(len(hubbard_model_general.terms))
    assert hubbard_model_special == hubbard_model_general

def random_parameters(lattice, probability):
    parameters = {}

    parameters['tunneling_parameters'] = [
            (edge_type, dofs, random.uniform(-1, 1))
            for edge_type in lattice.edge_types
            for dofs in lattice.dof_pairs_iter(edge_type == 'onsite')
            if random.random() <= probability]

    possible_spin_pairs = (SpinPairs.ALL,) if lattice.spinless else (SpinPairs.SAME, SpinPairs.DIFF)
    parameters['interaction_parameters'] = [
            (edge_type, dofs, random.uniform(-1, 1), spin_pairs)
            for edge_type in lattice.edge_types
            for spin_pairs in possible_spin_pairs
            for dofs in lattice.dof_pairs_iter(edge_type == 'onsite' and 
                                               spin_pairs in (SpinPairs.ALL, SpinPairs.SAME))
            if random.random() <= probability]

    parameters['potential_parameters'] = [
            (dof, random.uniform(-1, 1))
            for dof in lattice.dof_indices
            if random.random() <= probability]
    
    return parameters

lattices = [HubbardSquareLattice(random.randrange(3, 10), random.randrange(3, 10),
                                 periodic=periodic, spinless=spinless)
    for periodic in (False, True)
    for spinless in (False, True)
    for _ in range(2)
    ]
@pytest.mark.parametrize('lattice,parameters', [
    (lattice, random_parameters(lattice, 0.5))
    for lattice in lattices
    for _ in range(2)
    ])
def test_fermi_hubbard_square_lattice_random_parameters(lattice, parameters):
    model = FermiHubbardModel(lattice, **parameters)
    hamiltonian = model.hamiltonian()
    terms_per_parameter = defaultdict(int)
    for term, coefficient in hamiltonian.terms.items():
        spin_orbitals = set(i for i, _ in term)
        if len(spin_orbitals) == 2:
            (i, a, s), (ii, aa, ss) = (
                    lattice.from_spin_orbital_index(i) for i in spin_orbitals)
            distance = lattice.manhattan_distance(i, ii, True)
            edge_type = {0: 'onsite', 1: 'neighbor'}[distance]
            dofs = tuple(sorted((a, aa)))
            if len(term) == 2:
                parameter = (edge_type, dofs, -coefficient)
                assert parameter in parameters['tunneling_parameters']
                terms_per_parameter['tunneling', parameter] += 1
            elif len(term) == 4 and len(spin_orbitals) == 2:
                spin_pairs = (SpinPairs.ALL if lattice.spinless else 
                           (SpinPairs.SAME if s == ss else SpinPairs.DIFF))
                parameter = (edge_type, dofs, coefficient, spin_pairs)
                assert parameter in parameters['interaction_parameters']
                terms_per_parameter['interaction', parameter] += 1
            else:
                assert False
        elif len(term) == 2 and len(spin_orbitals) == 1:
            spin_orbital = spin_orbitals.pop()
            _, dof, _ = lattice.from_spin_orbital_index(spin_orbital)
            parameter = (dof, -coefficient)
            assert parameter in parameters['potential_parameters']
            terms_per_parameter['potential', parameter] += 1
        else:
            assert False
    edge_type_to_n_site_pairs = {
            'onsite': lattice.n_sites, 
            'neighbor': lattice.n_neighbor_pairs(False)}
    for (term_type, parameter), n_terms in terms_per_parameter.items():
        if term_type == 'potential':
            assert n_terms == lattice.n_sites * lattice.n_spin_values
            continue
        n_site_pairs = edge_type_to_n_site_pairs[parameter[0]]
        if term_type == 'tunneling':
            assert n_terms == 2 * n_site_pairs * lattice.n_spin_values
        elif term_type == 'interaction':
            parameter = InteractionParameter(*parameter)
            expected_n_terms = n_site_pairs
            if parameter.edge_type == 'neighbor':
                expected_n_terms *= len(set(parameter.dofs))
            if not lattice.spinless:
                if parameter.spin_pairs == SpinPairs.ALL:
                    expected_n_terms *= 4
                elif (parameter.edge_type == 'onsite' and 
                      parameter.spin_pairs == SpinPairs.DIFF):
                    expected_n_terms *= len(set(parameter.dofs))
                else:
                    expected_n_terms *= 2
            assert n_terms == expected_n_terms
        else:
            assert False
