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

import abc
from enum import Enum, IntEnum
import itertools


class Spin(IntEnum):
    UP = 0
    DOWN = 1


class SpinPairs(Enum):
    """The spin orbitals corresponding to a pair of spatial orbitals."""

    SAME = 1
    ALL = 0
    DIFF = -1


class HubbardLattice(metaclass=abc.ABCMeta):
    """Base class for a Hubbard model lattice.

    Subclasses must define the following properties:
        n_dofs (int): The number of degrees of freedom per site (and spin if
            applicable).
        n_sites (int): The number of sites in the lattice.
        spinless (bool): Whether or not the fermion has spin (False if so).
        edge_types (Tuple[Hashable, ...]): The types of edges that a term could
            correspond to. Examples include 'onsite', 'neighbor',
            'diagonal_neighbor', etc.
        onsite_edge_types (Sequence[Hashable]): The edge types that connect
            sites to themselves.

    And the following methods:
        site_pairs_iter(edge_type: Hashable) \\
            -> Iterable[Tuple[int, int]]: Iterable
            over pairs of sites corresponding to the given edge type.

    For 'spinful' lattices, the ``spin_indices`` ``0`` and ``1`` correspond to
    'up' and 'down', respectively.
    """

    @abc.abstractproperty
    def n_dofs(self):
        """The number of degrees of freedom per site
        (and spin if applicable)."""


    @abc.abstractproperty
    def n_sites(self):
        """The number of sites in the lattice."""


    @abc.abstractproperty
    def spinless(self):
        """Whether or not the fermion has spin (False if so)."""


    @abc.abstractproperty
    def edge_types(self):
        """The types of edges that a term could correspond to.

        Examples include 'onsite', 'neighbor', 'diagonal_neighbor', etc.
        """


    @abc.abstractproperty
    def onsite_edge_types(self):
        """The edge types that connect sites to themselves."""


    @abc.abstractmethod
    def site_pairs_iter(self, edge_type, ordered=True):
        """Iterable over pairs of sites corresponding to the given edge type."""


    # properties

    @property
    def n_spin_values(self):
        return 1 if self.spinless else 2


    @property
    def n_spin_orbitals_per_site(self):
        return self.n_dofs * self.n_spin_values


    @property
    def n_spin_orbitals(self):
        return self.n_sites * self.n_spin_orbitals_per_site


    # indexing

    def site_index_offset(self, site_index):
        return site_index * self.n_spin_orbitals_per_site


    def dof_index_offset(self, dof_index):
        return dof_index * self.n_spin_values


    def to_spin_orbital_index(self, site_index, dof_index, spin_index):
        """The index of the spin orbital."""
        return (self.site_index_offset(site_index) +
                self.dof_index_offset(dof_index) +
                spin_index)


    def from_spin_orbital_index(self, spin_orbital_index):
        site_index, offset = divmod(spin_orbital_index,
                                    self.n_spin_orbitals_per_site)
        dof_index, spin_index = divmod(offset, self.n_spin_values)
        return site_index, dof_index, spin_index


    # iteration

    @property
    def site_indices(self):
        return range(self.n_sites)


    @property
    def dof_indices(self):
        return range(self.n_dofs)


    def dof_pairs_iter(self, exclude_same=False):
        return ((a, b) for a in range(self.n_dofs)
                       for b in range(a + exclude_same, self.n_dofs))


    @property
    def spin_indices(self):
        return range(self.n_spin_values)


    def spin_pairs_iter(self, spin_pairs=SpinPairs.ALL, ordered=True):
        if spin_pairs == SpinPairs.ALL:
            return (itertools.product(self.spin_indices, repeat=2) if ordered
                    else itertools.combinations_with_replacement(
                        self.spin_indices, 2))
        elif spin_pairs == SpinPairs.SAME:
            return ((s, s) for s in self.spin_indices)
        elif spin_pairs == SpinPairs.DIFF:
            return (itertools.permutations(self.spin_indices, 2) if ordered
                    else itertools.combinations(self.spin_indices, 2))
        raise ValueError('{} not a valid SpinPairs specification.'.format(
            spin_pairs))


    # validation

    def validate_edge_type(self, edge_type):
        if edge_type not in self.edge_types:
            raise ValueError(
                    '{} not a valid edge type {}.'.format(
                        edge_type, self.edge_types))


    def validate_dof(self, dof, length=None):
        if not (0 <= dof < self.n_dofs):
            raise ValueError('not (0 <= {} < n_dofs = {})'.format(
                dof, self.n_dofs))


    def validate_dofs(self, dofs, length=None):
        for dof in dofs:
            self.validate_dof(dof)
        if (length is not None) and (len(dofs) != length):
            raise ValueError('len({}) != {}'.format(dofs, length))


class HubbardSquareLattice(HubbardLattice):
    r"""A square lattice for a Hubbard model.

    Valid edge types are:
        * 'onsite'
        * 'horizontal_neighbor'
        * 'vertical_neighbor'
        * 'neighbor': union of 'horizontal_neighbor' and 'vertical_neighbor'
        * 'diagonal_neighbor'
    """

    def __init__(self, x_dimension, y_dimension,
                 n_dofs=1, spinless=False, periodic=True):
        """
        Args:
            x_dimension (int): The width of the grid.
            y_dimension (int): The height of the grid.
            n_dofs (int, optional): The number of degrees of freedom per site
                (and spin if applicable). Defaults is 1.
            periodic (bool, optional): If True, add periodic boundary
                conditions. Default is True.
            spinless (bool, optional): If True, return a spinless Fermi-Hubbard
                model. Default is False.
        """

        self.x_dimension = x_dimension
        self.y_dimension = y_dimension
        self._n_dofs = n_dofs
        self._spinless = spinless
        self.periodic = periodic


    @property
    def n_dofs(self):
        return self._n_dofs


    @property
    def spinless(self):
        return self._spinless


    @property
    def n_sites(self):
        return self.x_dimension * self.y_dimension


    @property
    def edge_types(self):
        return ('onsite', 'neighbor', 'diagonal_neighbor',
                'horizontal_neighbor', 'vertical_neighbor')

    @property
    def onsite_edge_types(self):
        return ('onsite',)

    def site_pairs_iter(self, edge_type, ordered=True):
        if edge_type == 'onsite':
            return ((i, i) for i in self.site_indices)
        elif edge_type == 'neighbor':
            return self.neighbors_iter(ordered)
        elif edge_type == 'horizontal_neighbor':
            return self.horizontal_neighbors_iter(ordered)
        elif edge_type == 'vertical_neighbor':
            return self.vertical_neighbors_iter(ordered)
        elif edge_type == 'diagonal_neighbor':
            return self.diagonal_neighbors_iter(ordered)
        raise ValueError('Edge type {} is not valid.'.format(edge_type))


    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, ', '.join((
            ('x_dimension={}'.format(self.x_dimension)),
            ('y_dimension={}'.format(self.y_dimension)),
            ('n_dofs={}'.format(self.n_dofs)),
            ('spinless={}'.format(self.spinless)),
            ('periodic={}'.format(self.periodic))
            )))


    # site indexing

    def to_site_index(self, site):
        """The index of a site."""
        x, y = site
        return x + y * self.x_dimension


    def from_site_index(self, site_index):
        return divmod(site_index, self.x_dimension)[::-1]


    # neighbor counting and iteration

    def n_horizontal_neighbor_pairs(self, ordered=True):
        """Number of horizontally neighboring (unordered) pairs of sites."""
        n_horizontal_edges_per_y = (
            self.x_dimension - (self.x_dimension <= 2 or not self.periodic))
        return (self.y_dimension * n_horizontal_edges_per_y *
                (2 if ordered else 1))


    def n_vertical_neighbor_pairs(self, ordered=True):
        """Number of vertically neighboring (unordered) pairs of sites."""
        n_vertical_edges_per_x = (self.y_dimension -
                                  (self.y_dimension <= 2 or not self.periodic))
        return (self.x_dimension * n_vertical_edges_per_x *
                (2 if ordered else 1))

    def n_neighbor_pairs(self, ordered=True):
        """Number of neighboring (unordered) pairs of sites."""
        return (self.n_horizontal_neighbor_pairs(ordered) +
                self.n_vertical_neighbor_pairs(ordered))


    def neighbors_iter(self, ordered=True):
        return itertools.chain(
                self.horizontal_neighbors_iter(ordered),
                self.vertical_neighbors_iter(ordered))

    def diagonal_neighbors_iter(self, ordered=True):
        n_sites_per_y = (self.x_dimension -
                         (self.x_dimension <= 2 or not self.periodic))
        n_sites_per_x = (
                self.y_dimension -
                (self.y_dimension <= 2 or not self.periodic))
        for x in range(n_sites_per_y):
            for y in range(n_sites_per_x):
                for dy in (-1, 1):
                    i = self.to_site_index((x, y))
                    j = self.to_site_index(((x + 1) % self.x_dimension,
                                            (y + dy) % self.y_dimension))
                    yield (i, j)
                    if ordered:
                        yield (j, i)

    def horizontal_neighbors_iter(self, ordered=True):
        n_horizontal_edges_per_y = (
            self.x_dimension - (self.x_dimension <= 2 or not self.periodic))
        for x in range(n_horizontal_edges_per_y):
            for y in range(self.y_dimension):
                i = self.to_site_index((x, y))
                j = self.to_site_index(((x + 1) % self.x_dimension, y))
                yield (i, j)
                if ordered:
                    yield (j, i)


    def vertical_neighbors_iter(self, ordered=True):
        n_vertical_edges_per_x = (self.y_dimension -
                                  (self.y_dimension <= 2 or not self.periodic))
        for y in range(n_vertical_edges_per_x):
            for x in range(self.x_dimension):
                i = self.to_site_index((x, y))
                j = self.to_site_index((x, (y + 1) % self.y_dimension))
                yield (i, j)
                if ordered:
                    yield (j, i)


    # square-specific geometry

    @property
    def shape(self):
        return (self.x_dimension, self.y_dimension)


    def delta_mag(self, X, Y, by_index=False):
        """The distance between sites X and Y in each dimension."""
        if by_index:
            return self.delta_mag(self.from_site_index(X),
                                  self.from_site_index(Y))
        if self.periodic:
            return tuple(min(abs((s * (x - y)) % d) for s in (-1, 1))
                         for d, x, y in zip(self.shape, X, Y))
        return tuple(abs(x - xx) for x, xx in zip(X, Y))


    def manhattan_distance(self, X, Y, by_index=False):
        return sum(self.delta_mag(X, Y, by_index))
