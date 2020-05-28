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

"""This module contains methods to lift tensors to higher spaces
through the Grassmann wedge product."""
from itertools import product
from math import factorial
import copy
import numpy

# numpy einsum alphabet
EINSUM_CHARS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'


def generate_parity_permutations(seq):
    """
    Generates the permutations and sign of a sequence by constructing a tree
    where the nth level contains all n-permutations of m (n < m) objects.

    At the last level where n == m all permutations are generated.  The sign
    is kept updated by determining where the next number is inserted into the
    current leaf's set of numbers.

    Example:
         Constructing the permutations of the sequence [A, B, C] constructs the
         following tree:

         [[(A, +1)], [(AB, +1), (BA, -1)], [(ABC, +1), (ACB, -1), (CAB, +1],
         [(BAC, -1), (BCA, +1), (CBA, -1)]]

    Args:
        seq: a sequence of a string to provide permutations

    returns:
        a permutation list with the elements of the seq permuted and a sign
        associated with the permutation.
    """
    if isinstance(seq, str):
        seq = [x for x in seq]

    indices = seq[1:]
    permutations = [([seq[0]], 1)]
    while indices:
        index_to_inject = indices.pop(0)

        new_permutations = []  # permutations in the tree
        for perm in permutations:
            # now loop over positions to insert
            for put_index in range(len(perm[0]) + 1):
                new_index_list = copy.deepcopy(perm[0])
                # insert new object starting at end of the list
                new_index_list.insert(len(perm[0]) - put_index, index_to_inject)

                new_permutations.append(
                    (new_index_list, perm[1] * (-1) ** (put_index)))

        permutations = new_permutations

    return permutations


def wedge(left_tensor, right_tensor, left_index_ranks, right_index_ranks):
    """
    Implement the wedge product between left_tensor and right_tensor

    The wedge product is defined as

    .. math::
        \\begin{align}
        a_{j_{1}, j_{2}, ...,j_{p}}^{i_{1}, i_{2}, ..., i_{p}} \\wedge
        b_{j_{p+1}, j_{p+2}, ..., j_{N}}^{i_{p+1}, i_{p + 2}, ..., i_{N}} =
        \\left(\\frac{1}{N!}\\right)^{2} = \\sum_{\\pi, \\sigma}\\epsilon(\\pi)
        \\epsilon(\\sigma)\\pi \\sigma
        a_{j_{1}, j_{2}, ...,j_{p}}^{i_{1}, i_{2}, ..., i_{p}}
        b_{j_{p+1}, j_{p+2}, ..., j_{N}}^{i_{p+1}, i_{p + 2}, ..., i_{N}}
        \\end{align}

    The top indices are those that transform contravariently.  The bottom
    indices transform covariently.

    The tensor storage convention for marginals follows the OpenFermion
    convention. tpdm[i, j, k, l] = <i^ j^ k l>,
    rtensor[u1, u2, u3, d1] = <u1^ u2^ u3^ d1>

    Args:
        left_tensor: left tensor to wedge product
        right_tensor: right tensor to wedge product
        left_index_ranks: tuple of number of indices that transform
                          contravariently and covariently
        right_index_ranks: tuple of number of indices that transform
                           contravariently and covariently
    Returns:
        new tensor constructed as the wedge product of the left_tensor and
        right_tensor
    """
    if left_tensor.ndim != sum(left_index_ranks):
        raise IndexError(
            "n_tensor shape is not consistent with the input n_index rank")
    if right_tensor.ndim != sum(right_index_ranks):
        raise IndexError(
            "n_tensor shape is not consistent with the input n_index rank")
    # assign upper and lower indices for n_tensor
    total_upper = left_index_ranks[0] + right_index_ranks[0]
    total_lower = left_index_ranks[1] + right_index_ranks[1]
    upper_characters = EINSUM_CHARS[:total_upper]
    lower_characters = EINSUM_CHARS[total_upper:total_upper + total_lower]
    new_tensor = numpy.zeros(left_tensor.shape + right_tensor.shape,
                             dtype=complex)
    ordered_einsum_string = upper_characters + lower_characters

    for upper_order_parities, lower_order_parities in product(
            generate_parity_permutations(upper_characters),
            generate_parity_permutations(lower_characters[::-1])):
        # we reverse the order in the lower_chars so because
        # <a^ b^ c d> = D_{dc}^{ab} in this code.
        n_upper_einsum_chars = upper_order_parities[0][:left_index_ranks[0]]
        m_upper_einsum_chars = upper_order_parities[0][left_index_ranks[0]:]
        n_lower_einsum_chars = lower_order_parities[0] \
            [:left_index_ranks[1]][::-1]
        m_lower_einsum_chars = lower_order_parities[0] \
            [left_index_ranks[1]:][::-1]

        n_string = "".join(n_upper_einsum_chars + n_lower_einsum_chars)
        m_string = "".join(m_upper_einsum_chars + m_lower_einsum_chars)

        # we are doing lots of extra += operations but with the benefit of not
        # having to write a python loop over the entire new_tensor object.
        new_tensor += upper_order_parities[1] * lower_order_parities[1] * \
                      numpy.einsum('{},{}->{}'.format(n_string, m_string,
                                                      ordered_einsum_string),
                                   left_tensor, right_tensor)

    new_tensor /= factorial(total_upper) * factorial(total_lower)

    return new_tensor
