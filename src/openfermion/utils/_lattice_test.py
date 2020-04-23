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

"""Tests for Hubbard model lattice module."""

import itertools
import random
import pytest

from openfermion.utils import (HubbardSquareLattice, SpinPairs, Spin)

def test_spin():
    lattice = HubbardSquareLattice(3, 3)
    assert tuple(lattice.spin_indices) == (Spin.UP, Spin.DOWN)


@pytest.mark.parametrize("x_dimension,y_dimension,n_dofs,spinless,periodic",
                         itertools.product(random.sample(range(3, 10), 3),
                                           random.sample(range(3, 10), 3),
                                           range(1, 4), (False, True),
                                           (False, True)))
def test_hubbard_square_lattice(
    x_dimension, y_dimension, n_dofs, spinless, periodic):
    lattice = HubbardSquareLattice(x_dimension, y_dimension,
            n_dofs=n_dofs, spinless=spinless, periodic=periodic)

    n_spin_values = 2 - spinless
    sites = tuple(
        (x, y)
        for y, x in itertools.product(range(y_dimension), range(x_dimension)))
    site_indices = tuple(lattice.to_site_index(site) for site in sites)
    assert (sites == tuple(
        lattice.from_site_index(site_index) for site_index in site_indices))
    assert (site_indices == tuple(lattice.site_indices) == tuple(
        range(x_dimension * y_dimension)))

    tuple(
        itertools.product(range(x_dimension), range(y_dimension), range(n_dofs),
                          range(n_spin_values)))

    spin_orbital_linear_indices = tuple(lattice.to_spin_orbital_index(*indices)
            for indices in itertools.product(
                site_indices, range(n_dofs), range(n_spin_values)))
    assert spin_orbital_linear_indices == tuple(range(lattice.n_spin_orbitals))

    for i, ii in zip(range(lattice.n_sites),
                     lattice.site_pairs_iter('onsite')):
        assert ii == (i, i)

    n_neighbor_pairs = 2 * (
            (x_dimension * (y_dimension - (not periodic))) +
            ((x_dimension - (not periodic)) * y_dimension))
    neighbor_pairs = tuple(lattice.site_pairs_iter('neighbor'))
    assert (2 * len(tuple(lattice.site_pairs_iter('neighbor', False))) ==
            len(neighbor_pairs) == n_neighbor_pairs)
    for i, j in neighbor_pairs:
        assert sum(lattice.delta_mag(i, j, True)) == 1

    assert len(tuple(lattice.dof_pairs_iter(False))) == \
        n_dofs * (n_dofs + 1) / 2
    assert len(tuple(lattice.dof_pairs_iter(True))) == n_dofs * (n_dofs - 1) / 2
    spin_pairs_all = tuple(lattice.spin_pairs_iter())
    assert len(spin_pairs_all) == n_spin_values ** 2
    spin_pairs_same = tuple(lattice.spin_pairs_iter(SpinPairs.SAME))
    assert spin_pairs_same == tuple((s, s) for s in range(n_spin_values))
    spin_pairs_diff = tuple(lattice.spin_pairs_iter(SpinPairs.DIFF))
    assert spin_pairs_diff == () if spinless else ((0, 1), (1, 0))


def test_hubbard_square_lattice_defaults():
    lattice = HubbardSquareLattice(3, 3)
    assert lattice.n_dofs == 1
    assert lattice.spinless == False
    assert lattice.periodic == True


@pytest.mark.parametrize('n_dofs', range(1, 4))
def test_hubbard_square_lattice_dof_validation(n_dofs):
    lattice = HubbardSquareLattice(3, 3, n_dofs=n_dofs)
    for a in range(n_dofs):
        lattice.validate_dof(a)
    for ab in itertools.combinations(range(n_dofs), 2):
        lattice.validate_dofs(ab, 2)
    for abc in itertools.combinations(range(n_dofs), 3):
        lattice.validate_dofs(abc, 3)
    with pytest.raises(ValueError):
        lattice.validate_dof(-1)
    with pytest.raises(ValueError):
        lattice.validate_dof(n_dofs)
    with pytest.raises(ValueError):
        lattice.validate_dofs((0, 0), 1)


def test_hubbard_square_lattice_edge_types():
    lattice = HubbardSquareLattice(3, 3)
    assert sorted(lattice.edge_types) == sorted((
            'onsite', 'neighbor', 'diagonal_neighbor',
            'vertical_neighbor', 'horizontal_neighbor'))
    lattice.validate_edge_type('onsite')
    lattice.validate_edge_type('neighbor')
    lattice.validate_edge_type('diagonal_neighbor')
    lattice.validate_edge_type('vertical_neighbor')
    lattice.validate_edge_type('horizontal_neighbor')
    with pytest.raises(ValueError):
        lattice.validate_edge_type('banana')
    with pytest.raises(ValueError):
        lattice.site_pairs_iter('banana')


@pytest.mark.parametrize('d', random.sample(range(3, 10), 3))
def test_hubbard_square_lattice_1xd(d):
    for shape, periodic in itertools.product(((1, d), (d, 1)), (True, False)):
        lattice = HubbardSquareLattice(*shape, periodic=periodic)
        assert lattice.n_sites == d
        assert (len(tuple(lattice.neighbors_iter())) ==
                2 * len(tuple(lattice.neighbors_iter(False))) ==
                2 * (d - (not periodic)))
        assert (len(tuple(lattice.diagonal_neighbors_iter())) ==
                len(tuple(lattice.diagonal_neighbors_iter(False))) ==
                0)


@pytest.mark.parametrize('x,y',
                         (random.sample(range(3, 10), 2) for _ in range(3)))
def test_hubbard_square_lattice_neighbors(x, y):
    for periodic in (True, False):
        lattice = HubbardSquareLattice(x, y, periodic=periodic)
        n_horizontal_neighbors = 2 * y * (x - (not periodic))
        assert (len(tuple(lattice.horizontal_neighbors_iter())) ==
                n_horizontal_neighbors)
        n_vertical_neighbors = 2 * x * (y - (not periodic))
        assert (len(tuple(lattice.vertical_neighbors_iter())) ==
                n_vertical_neighbors)
        n_diagonal_neighbors = 4 * (x - (not periodic)) * (y - (not periodic))
        assert (len(tuple(lattice.diagonal_neighbors_iter())) ==
                n_diagonal_neighbors)


@pytest.mark.parametrize('d', random.sample(range(3, 10), 3))
def test_hubbard_square_lattice_2xd(d):
    for shape, periodic in itertools.product(((2, d), (d, 2)), (True, False)):
        lattice = HubbardSquareLattice(*shape, periodic=periodic)
        assert lattice.n_sites == 2 * d
        n_neighbor_pairs = 2 * (2 * (d - (not periodic)) + d)
        assert (len(tuple(lattice.neighbors_iter())) ==
                2 * len(tuple(lattice.neighbors_iter(False))) ==
                n_neighbor_pairs)
        n_diagonal_neighbor_pairs = 4 * (d - (not periodic))
        assert (len(tuple(lattice.diagonal_neighbors_iter())) ==
                2 * len(tuple(lattice.diagonal_neighbors_iter(False))) ==
                n_diagonal_neighbor_pairs)


def test_hubbard_square_lattice_2x2():
    for periodic in (True, False):
        lattice = HubbardSquareLattice(2, 2, periodic=periodic)
        assert lattice.n_sites == 4
        assert len(tuple(lattice.neighbors_iter(False))) == 4
        assert len(tuple(lattice.neighbors_iter())) == 8
        assert len(tuple(lattice.diagonal_neighbors_iter(False))) == 2
        assert len(tuple(lattice.diagonal_neighbors_iter())) == 4


@pytest.mark.parametrize('kwargs', [{
    'x_dimension': random.randrange(1, 10),
    'y_dimension': random.randrange(1, 10),
    'n_dofs': random.randrange(1, 10),
    'spinless': random.choice((True, False)),
    'periodic': random.choice((True, False))
    } for _ in range(5)])
def test_hubbard_square_lattice_repr(kwargs):
    lattice = HubbardSquareLattice(**kwargs)
    params = ('x_dimension', 'y_dimension', 'n_dofs', 'spinless', 'periodic')
    param_template = ', '.join('{0}={{{0}}}'.format(param) for param in params)
    param_str = param_template.format(**kwargs)
    assert repr(lattice) == 'HubbardSquareLattice({})'.format(param_str)


def test_spin_pairs_iter():
    spinful_lattice = HubbardSquareLattice(3, 3)

    with pytest.raises(ValueError):
        spinful_lattice.spin_pairs_iter(10)

    assert (tuple(spinful_lattice.spin_pairs_iter(SpinPairs.ALL,
                                                  True)) == ((0, 0), (0, 1),
                                                             (1, 0), (1, 1)))
    assert (tuple(spinful_lattice.spin_pairs_iter(SpinPairs.ALL, False)) ==
            ((0, 0), (0, 1), (1, 1)))
    assert (tuple(spinful_lattice.spin_pairs_iter(SpinPairs.SAME, True)) ==
            tuple(spinful_lattice.spin_pairs_iter(SpinPairs.SAME, False)) ==
            ((0, 0), (1, 1)))
    assert (tuple(spinful_lattice.spin_pairs_iter(SpinPairs.DIFF, True)) ==
            ((0, 1), (1, 0)))
    assert (tuple(spinful_lattice.spin_pairs_iter(SpinPairs.DIFF, False)) ==
            ((0, 1),))

    spinless_lattice = HubbardSquareLattice(3, 3, spinless=True)
    assert (tuple(spinless_lattice.spin_pairs_iter(SpinPairs.ALL, True)) ==
            tuple(spinless_lattice.spin_pairs_iter(SpinPairs.ALL, False)) ==
            tuple(spinless_lattice.spin_pairs_iter(SpinPairs.SAME, True)) ==
            tuple(spinless_lattice.spin_pairs_iter(SpinPairs.SAME, False)) ==
            ((0, 0),))
    assert (tuple(spinless_lattice.spin_pairs_iter(
        SpinPairs.DIFF, True)) == tuple(
            spinless_lattice.spin_pairs_iter(SpinPairs.DIFF, False)) == tuple())
