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

from __future__ import absolute_import

import itertools
import numpy
import scipy
import scipy.linalg


class Grid:
    """
    A multi-dimensional grid of points.
    """

    def __init__(self, dimensions, length, scale):
        """
        Args:
            dimensions (int): The number of dimensions the grid lives in.
            length (int or tuple): The number of points along each grid axis
                that will be taken in both reciprocal and real space.
                If tuple, it is read for each dimension, otherwise assumed
                uniform.
            scale (float or ndarray): The total length of each grid dimension.
                If a float is passed, the uniform cubic unit cell is assumed.
                For an ndarray, dimensions independent vectors of the correct
                dimension must be passed.  We assume column vectors define
                the supercell vectors.
        """
        if not isinstance(dimensions, int) or dimensions < 0:
            raise ValueError(
                'dimensions must be a non-negative int but was {} {}'.format(
                    type(dimensions), repr(dimensions)))
        if ((not isinstance(length, int) or length < 0) and
                (not isinstance(length, tuple))):
            raise ValueError(
                'length must be a non-negative int or tuple '
                'but was {} {}'.format(
                    type(length), repr(length)))
        if ((not isinstance(scale, float) or not scale > 0) and
                (not isinstance(scale, numpy.ndarray))):
            raise ValueError(
                'scale must be a positive float or ndarray but was '
                '{} {}'.format(
                    type(scale), repr(scale)))

        self.dimensions = dimensions

        # If single integer, assume uniform
        if isinstance(length, int):
            self.length = (length, ) * dimensions
        else:
            self.length = length

        self.shifts = [self.length[i] // 2 for i in range(dimensions)]

        # If single float, construct cubic unit cell
        if isinstance(scale, float):
            self.scale = numpy.diagonal([scale] * self.dimensions)
        else:
            self.scale = scale

        # Compute the volume of the super cell
        self.volume = numpy.abs(scipy.linalg.det(self.scale))

        # Compute total number of points
        self.num_points = numpy.prod(self.length)

        # Compute the reciprocal lattice basis
        self.reciprocal_scale = 2 * numpy.pi * scipy.linalg.inv(self.scale).T


    def index_to_momentum_ints(self, index):
        """
        Args:
            index (tuple): d-dimensional tuple specifying index in the grid
        Returns:
            Integer momentum vector
        """
        # Set baseline for grid between [-N//2, N//2]
        momentum_int = [(index[i] % self.length[i]) - self.shifts[i]
                        for i in range(self.dimensions)]

        # Adjust for even grids without 0 point
        momentum_int = [v + ((v > 0) and (self.length[i] % 2) == 0) for i, v in
                        enumerate(momentum_int)]

        return momentum_int

    def index_to_momentum_value(self, index):
        """
        Args:
            index (tuple): d-dimensional tuple specifying index in the grid
        Returns:
            ndarray of momentum on the reciprocal lattice
        """
        # Calculate the momentum index and translate to vectors using recip lat
        momentum_ints = self.index_to_momentum_ints(index)
        momentum_vector = self.momentum_ints_to_value(momentum_ints)

        return momentum_vector

    def index_to_position(self, index):
        """
        Args:
            index (tuple): d-dimensional tuple specifying index in the grid
        Returns:
            ndarray of position on the lattice
        """
        position_vector = sum([(float(n) / self.length[i]) * self.scale[:, i]
                              for i, n in enumerate(index)])
        return position_vector

    def momentum_ints_to_index(self, momentum_ints):
        """
        Args:
            momentum_ints (tuple): d-dimensional tuple momentum integers
        Returns:
            d-dimensional tuples of indices
        """
        indices = [n + self.shifts[i] for i, n in enumerate(momentum_ints)]

        # Take care of even dimension
        indices = [n - ((n > 0) and (self.length[i] % 2) == 0)
                   for i, v in enumerate(indices)]

        # Wrap dimensions
        indices = [n % self.length[i] for i, n in enumerate(indices)]

        return indices

    def momentum_ints_to_value(self, momentum_ints, periodic=True):
        """
        Args:
            momentum_ints (tuple): d-dimensional tuple momentum integers
            periodic (bool): Alias the momentum
        Returns:

        """
        # If we want to alias the momentum, do so in the index space
        if periodic:
            momentum_ints = self.index_to_momentum_ints(
                self.momentum_ints_to_index(momentum_ints))

        momentum_vector = sum([n * self.reciprocal_scale[:, i]
                               for i, n in enumerate(momentum_ints)])
        return momentum_vector


    def volume_scale(self):
        """
        Returns:
            float: The volume of a length-scale hypercube within the grid.
        """
        return self.volume

    def num_points(self):
        """
        Returns:
            int: The number of points in the grid.
        """
        return self.num_points

    def all_points_indices(self):
        """
        Returns:
            iterable[tuple[int]]:
                The index-coordinate tuple of each point in the grid.
        """
        return itertools.product(range(self.length), repeat=self.dimensions)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (self.dimensions == other.dimensions and
                self.scale == other.scale and
                self.length == other.length)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((Grid, self.dimensions, self.length, self.scale))

    def __repr__(self):
        return 'Grid(dimensions={}, length={}, scale={})'.format(
            repr(self.dimensions),
            repr(self.length),
            repr(self.scale))

    def __str__(self):
        s = self.scale / self.length
        return "{}**{}".format(
            [(i - self.length // 2) * s for i in range(self.length)],
            self.dimensions)
