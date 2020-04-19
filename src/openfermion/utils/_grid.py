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


import itertools
import numpy
import scipy
import scipy.linalg


# Exceptions.
class OrbitalSpecificationError(Exception):
    pass


class Grid:
    """
    A multi-dimension grid of points with an assigned length scale.

    This grid acts as a helper class for parallelpiped super cells.  It
        tracks a mapping from indices to grid points and stores the associated
        reciprocal lattice with respect to the original real-space lattice.
        This enables calculations with non-trivial unit cells.

    Attributes:
        dimensions (int): Number of spatial dimensions the grid occupys
        length (tuple of ints): d-length tuple specifying number of points
            along each dimension.
        shifts (list of ints): Integer shifts in position to center grid.
        scale (ndarray): Vectors defining the super cell being simulated,
            vectors are stored as columns in the matrix.
        volume (float): Total volume of the supercell parallelpiped.
        num_points (int): Total number of points in the grid.
        reciprocal_scale (ndarray): Vectors defining the reciprocal lattice.
            The vectors are stored as the columns in the matrix.
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
        if not isinstance(dimensions, int) or dimensions <= 0:
            raise ValueError(
                'dimensions must be a positive int but was {} {}'.format(
                    type(dimensions), repr(dimensions)))
        if ((not isinstance(length, int) or length < 0) and
                (not isinstance(length, tuple)) and
                (not isinstance(length, list))):
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
            self.scale = numpy.diag([scale] * self.dimensions)
        else:
            self.scale = scale

        # Compute the volume of the super cell
        self.volume = numpy.abs(scipy.linalg.det(self.scale))

        # Compute total number of points
        self.num_points = numpy.prod(self.length)

        # Compute the reciprocal lattice basis
        self.reciprocal_scale = 2 * numpy.pi * scipy.linalg.inv(self.scale).T

    def volume_scale(self):
        """
        Returns:
            float: The volume of a length-scale hypercube within the grid.
        """
        return self.volume

    def all_points_indices(self):
        """
        Returns:
            iterable[tuple[int]]:
                The index-coordinate tuple of each point in the grid.
        """
        return itertools.product(*[range(self.length[i])
                                   for i in range(self.dimensions)])

    def position_vector(self, position_indices):
        """Given grid point coordinate, return position vector with dimensions.

        Args:
            position_indices (int|iterable[int]):
                List or tuple of integers giving grid point coordinate.
                Allowed values are ints in [0, grid_length).

        Returns:
            position_vector (numpy.ndarray[float])
        """
        # Raise exceptions.
        if isinstance(position_indices, int):
            position_indices = [position_indices]
        if not all(0 <= e < self.length[i]
                   for i, e in enumerate(position_indices)):
            raise OrbitalSpecificationError(
                'Position indices must be integers in [0, grid_length).')

        # Compute position vector
        vector = sum([(float(n - self.shifts[i]) /
                       self.length[i]) * self.scale[:, i]
                      for i, n in enumerate(position_indices)])
        return vector

    def momentum_vector(self, momentum_indices, periodic=True):
        """Given grid point coordinate, return momentum vector with dimensions.

        Args:
            momentum_indices (list): integers giving momentum
                indices. Allowed values are ints in [0, grid_length).
            periodic (bool): Wrap the momentum indices according to periodicity

            Returns:
                momentum_vector: A numpy array giving the momentum vector with
                    dimensions.
        """
        # Raise exceptions.
        if isinstance(momentum_indices, int):
            momentum_indices = [momentum_indices]
        if (not all(0 <= e < self.length[i]
                    for i, e in enumerate(momentum_indices))):
            raise OrbitalSpecificationError(
                'Momentum indices must be integers in [0, grid_length).')

        # Compute momentum vector.
        momentum_ints = self.index_to_momentum_ints(momentum_indices)
        vector = self.momentum_ints_to_value(momentum_ints, periodic)

        return vector

    def index_to_momentum_ints(self, index):
        """
        Args:
            index (tuple): d-dimensional tuple specifying index in the grid
        Returns:
            Integer momentum vector
        """
        # Set baseline for grid between [-N//2, N//2]
        momentum_int = [index[i] - self.shifts[i]
                        for i in range(self.dimensions)]

        return numpy.array(momentum_int, dtype=int)

    def momentum_ints_to_index(self, momentum_ints):
        """
        Args:
            momentum_ints (tuple): d-dimensional tuple momentum integers
        Returns:
            d-dimensional tuples of indices
        """

        indices = momentum_ints

        # Shift to indices
        indices = [n + self.shifts[i] for i, n in enumerate(indices)]

        # Wrap dimensions
        indices = [n % self.length[i] for i, n in enumerate(indices)]

        return indices

    def momentum_ints_to_value(self, momentum_ints, periodic=True):
        """
        Args:
            momentum_ints (tuple): d-dimensional tuple momentum integers
            periodic (bool): Alias the momentum
        Returns:
            ndarray containing the momentum vector.

        """
        # Alias the higher momentum modes
        if periodic:
            momentum_ints = self.index_to_momentum_ints(
                self.momentum_ints_to_index(momentum_ints))

        momentum_vector = sum([n * self.reciprocal_scale[:, i]
                               for i, n in enumerate(momentum_ints)])
        return momentum_vector

    def orbital_id(self, grid_coordinates, spin=None):
        """Return the tensor factor of a orbital
        with given coordinates and spin.

        Args:
            grid_coordinates: List or tuple of ints giving coordinates of grid
                element. Acceptable to provide an int(instead of tuple or list)
                for 1D case.
            spin (bool): 0 means spin down and 1 means spin up.
                If None, assume spinless model.

        Returns:
            tensor_factor (int):
                tensor factor associated with provided orbital label.
        """
        # Initialize.
        if isinstance(grid_coordinates, int):
            grid_coordinates = [grid_coordinates]

        # Loop through dimensions of coordinate tuple.
        tensor_factor = 0
        for dimension, grid_coordinate in enumerate(grid_coordinates):

            # Make sure coordinate is an integer in the correct bounds.
            if (isinstance(grid_coordinate, int) and
                    grid_coordinate < self.length[dimension]):
                tensor_factor += (grid_coordinate *
                                  int(numpy.product(self.length[:dimension])))
            else:
                # Raise for invalid model.
                raise OrbitalSpecificationError(
                    'Invalid orbital coordinates provided.')

        # Account for spin and return.
        if spin is None:
            return tensor_factor
        else:
            tensor_factor *= 2
            tensor_factor += spin
            return tensor_factor

    def grid_indices(self, qubit_id, spinless):
        """This function is the inverse of orbital_id.

        Args:
            qubit_id (int): The tensor factor to map to grid indices.
            spinless (bool): Whether to use the spinless model or not.

        Returns:
            grid_indices (numpy.ndarray[int]):
                The location of the qubit on the grid.
        """
        if not (numpy.product(self.length) * (2 - spinless) > qubit_id >= 0):
            raise OrbitalSpecificationError('Invalid qubit_id provided.')

        # Remove spin degree of freedom if it exists.
        orbital_id = qubit_id

        if not spinless:
            orbital_id //= 2

        # Get grid indices.
        grid_indices = []
        for dimension in range(self.dimensions):
            remainder = (orbital_id %
                         int(numpy.product(self.length[:dimension + 1])))
            grid_index = (remainder //
                          int(numpy.product(self.length[:dimension])))
            grid_indices += [grid_index]
        return grid_indices

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (self.dimensions == other.dimensions and
                (self.scale == other.scale).all() and
                self.length == other.length)

    def __ne__(self, other):
        return not self == other
