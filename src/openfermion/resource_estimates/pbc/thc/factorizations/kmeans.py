# coverage: ignore
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
import numpy as np
import numpy.typing as npt

"""Module for performing K-Means CVT algorithm to find interpolating points.

Provides centroidal Veronoi tesselation of the grid of real space points
weighted by the electron density using a K-Means classification.
"""


class KMeansCVT(object):
    def __init__(self, grid: npt.NDArray, max_iteration: int = 100, threshold: float = 1e-6):
        """Initialize k-means solver to find interpolating points for ISDF.

        Args:
        grid: Real space grid of dimension [Ng,Ndim], where Ng is the number
            of (dense) real space grid points and Ndim is number of spatial
            dimensions.
        max_iteration: Maximum number of iterations to perform when
            classifying grid points. Default 100.
        threshold: Threshold for exiting classification. Default 1e-6.

        Returns:
        """
        self.grid = grid
        self.max_iteration = max_iteration
        self.threshold = threshold

    @staticmethod
    def classify_grid_points(grid_points: npt.NDArray, centroids: npt.NDArray) -> npt.NDArray:
        r"""Assign grid points to centroids.

        Find centroid closest to each given grid point.

        Note we don't use instance variable self.grid as we can abuse this
        function and use it to map grid to centroid and centroid to grid point.

        Args:
          grid_points: grid points to assign.
          centroids: Centroids to which grid points should be assigned,
            array of length num_interp_points.

        Returns:
          1D np.array assigning grid point to centroids
        """
        # Build N_g x N_mu matrix of distances.
        num_grid_points = grid_points.shape[0]
        num_interp_points = centroids.shape[0]
        distances = np.zeros((num_grid_points, num_interp_points))
        # For loop is faster than broadcasting by 2x.
        for ig in range(num_grid_points):
            distances[ig] = np.linalg.norm(grid_points[ig] - centroids, axis=1)
        # Find shortest distance for each grid point.
        classification = np.argmin(distances, axis=1)
        return classification

    def compute_new_centroids(self, weighting, grid_mapping, current_centroids) -> npt.NDArray:
        r"""
        Centroids are defined via:

        .. math::

            c(C_\mu) = \frac{\sum_{j in C(\mu)} r_j \rho(r_j)}{\sum_{j in
                C(\mu)} \rho(r_j)},

        where :math:`\rho(r_j)` is the weighting factor.

        Args:
            weighting: Weighting function.
            grid_mapping: maps grid points to centroids
            current_centroids: centroids for current iteration.

        Returns:
            new_centroids: updated centroids
        """
        num_interp_points = current_centroids.shape[0]
        new_centroids = np.zeros_like(current_centroids)
        for interp_indx in range(num_interp_points):
            # get grid points belonging to this centroid
            grid_indx = np.where(grid_mapping == interp_indx)[0]
            grid_points = self.grid[grid_indx]
            weight = weighting[grid_indx]
            numerator = np.einsum("Jx,J->x", grid_points, weight)
            denominator = np.einsum("J->", weight)
            if denominator < 1e-12:
                print("Warning very small denominator, something seems wrong!")
                print("{interp_indx}")
            new_centroids[interp_indx] = numerator / denominator
        return new_centroids

    def map_centroids_to_grid(self, centroids):
        grid_mapping = self.classify_grid_points(centroids, self.grid)
        return grid_mapping

    def find_interpolating_points(
        self, num_interp_points: int, weighting_factor: npt.NDArray, centroids=None, verbose=True
    ) -> npt.NDArray:
        """Find interpolating points using KMeans-CVT algorithm.

        Args:
            num_interp_points: number of points to select.
            weighting_factor: weighting function for K-Means procedure.
            centroids: initial guess at centroids, if None centroids are
                selected randomly from the grid points.
            verbose: Controls if information is printed about convergence.
                Default value = True.

        Returns:
            interp_pts: index associated with interpolating points.
        """
        num_grid_points = self.grid.shape[0]
        if centroids is None:
            # Randomly select grid points as centroids.
            centroids_indx = np.random.choice(num_grid_points, num_interp_points, replace=False)
            centroids = self.grid[centroids_indx].copy()
        else:
            assert len(centroids) == num_interp_points
        # define here to avoid linter errors about possibly undefined.
        new_centroids = np.zeros_like(centroids)
        delta_grid = 1.0
        if verbose:
            print("{:<10s}  {:>13s}".format("iteration", "Error"))
        for iteration in range(self.max_iteration):
            grid_mapping = self.classify_grid_points(self.grid, centroids)
            # Global reduce
            new_centroids[:] = self.compute_new_centroids(weighting_factor, grid_mapping, centroids)
            delta_grid = np.linalg.norm(new_centroids - centroids)
            if verbose and iteration % 10 == 0:
                print(f"{iteration:<9d}  {delta_grid:13.8e}")
            if delta_grid < self.threshold:
                if verbose:
                    print("KMeansCVT successfully completed.")
                    print(f"Final error {delta_grid:13.8e}.")
                return self.map_centroids_to_grid(new_centroids)
            centroids[:] = new_centroids[:]
        print("Warning K-Means not converged.")
        print(f"Final error {delta_grid:13.8e}.")
        return self.map_centroids_to_grid(new_centroids)
