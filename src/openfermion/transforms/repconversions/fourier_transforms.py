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
"""Fourier transforms on operators"""

import numpy

from openfermion.ops.operators import FermionOperator


def _fourier_transform_helper(hamiltonian, grid, spinless, phase_factor,
                              vec_func_1, vec_func_2):
    hamiltonian_t = FermionOperator.zero()
    normalize_factor = numpy.sqrt(1.0 / float(grid.num_points))

    for term in hamiltonian.terms:
        transformed_term = FermionOperator.identity()
        for ladder_op_mode, ladder_op_type in term:
            indices_1 = grid.grid_indices(ladder_op_mode, spinless)
            vec1 = vec_func_1(indices_1)
            new_basis = FermionOperator.zero()
            for indices_2 in grid.all_points_indices():
                vec2 = vec_func_2(indices_2)
                spin = None if spinless else ladder_op_mode % 2
                orbital = grid.orbital_id(indices_2, spin)
                exp_index = phase_factor * 1.0j * numpy.dot(vec1, vec2)
                if ladder_op_type == 1:
                    exp_index *= -1.0

                element = FermionOperator(((orbital, ladder_op_type),),
                                          numpy.exp(exp_index))
                new_basis += element

            new_basis *= normalize_factor
            transformed_term *= new_basis

        # Coefficient.
        transformed_term *= hamiltonian.terms[term]

        hamiltonian_t += transformed_term

    return hamiltonian_t


def fourier_transform(hamiltonian, grid, spinless):
    r"""Apply Fourier transform to change hamiltonian in plane wave basis.

    $$
        c^\dagger_v = \sqrt{1/N} \sum_m {a^\dagger_m \exp(-i k_v r_m)}
        c_v = \sqrt{1/N} \sum_m {a_m \exp(i k_v r_m)}
    $$

    Args:
        hamiltonian (FermionOperator): The hamiltonian in plane wave basis.
        grid (Grid): The discretization to use.
        spinless (bool): Whether to use the spinless model or not.

    Returns:
        FermionOperator: The fourier-transformed hamiltonian.
    """
    return _fourier_transform_helper(hamiltonian=hamiltonian,
                                     grid=grid,
                                     spinless=spinless,
                                     phase_factor=+1,
                                     vec_func_1=grid.momentum_vector,
                                     vec_func_2=grid.position_vector)


def inverse_fourier_transform(hamiltonian, grid, spinless):
    r"""Apply inverse Fourier transform to change hamiltonian in
    plane wave dual basis.

    $$
        a^\dagger_v = \sqrt{1/N} \sum_m {c^\dagger_m \exp(i k_v r_m)}
        a_v = \sqrt{1/N} \sum_m {c_m \exp(-i k_v r_m)}
    $$

    Args:
        hamiltonian (FermionOperator):
            The hamiltonian in plane wave dual basis.
        grid (Grid): The discretization to use.
        spinless (bool): Whether to use the spinless model or not.

    Returns:
        FermionOperator: The inverse-fourier-transformed hamiltonian.
    """
    return _fourier_transform_helper(hamiltonian=hamiltonian,
                                     grid=grid,
                                     spinless=spinless,
                                     phase_factor=-1,
                                     vec_func_1=grid.position_vector,
                                     vec_func_2=grid.momentum_vector)
