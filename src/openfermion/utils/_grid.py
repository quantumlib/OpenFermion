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


class Grid:
    """
    A multi-dimensional grid of points.
    """

    def __init__(self, dimensions, length, scale):
        """
        Args:
            dimensions (int): The number of dimensions the grid lives in.
            length (int): The number of points along each grid axis.
            scale (float): The total length of each grid dimension.
        """
        if not isinstance(dimensions, int) or dimensions < 0:
            raise ValueError(
                'dimensions must be a non-negative int but was {} {}'.format(
                    type(dimensions), repr(dimensions)))
        if not isinstance(length, int) or length < 0:
            raise ValueError(
                'length must be a non-negative int but was {} {}'.format(
                    type(length), repr(length)))
        if not isinstance(scale, float) or not scale > 0:
            raise ValueError(
                'scale must be a positive float but was {} {}'.format(
                    type(scale), repr(scale)))

        self.dimensions = dimensions
        self.length = length
        self.scale = scale

    def volume_scale(self):
        """
        Returns:
            float: The volume of a length-scale hypercube within the grid.
        """
        return self.scale ** float(self.dimensions)

    def num_points(self):
        """
        Returns:
            int: The number of points in the grid.
        """
        return self.length ** self.dimensions

    def all_points_indices(self):
        """
        Returns:
            iterable[tuple[int]]:
                The index-coordinate tuple of each point in the grid.
        """
        return itertools.product(range(self.length), repeat=self.dimensions)
