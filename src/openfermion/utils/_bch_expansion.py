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

"""Module to efficiently compute the Baker-Campbell-Hausdorff formula."""

import itertools
from scipy.misc import comb, factorial


def bch_expand(x, y, order=6):
    """Compute log[e^x e^y] using the Baker-Campbell-Hausdorff formula.

    This implementation is explained in arXiv:1712.01348.

    Args:
        x: An operator for which multiplication and addition are supported.
            For instance, a QubitOperator, FermionOperator or scipy sparse
            matrix.
        y: The same type as x.
        order(int): The max degree of monomial with respect to X and Y
            to truncate the BCH expansions.

    Returns:
        z: The truncated BCH operator.

    Raises:
        ValueError: operator x is not same type as operator y.
        ValueError: invalid order parameter.
        ValueError: order exceeds maximum order supported.
    """
    if (not isinstance(order, int)) or order < 0:
        raise ValueError('Invalid order parameter.')
    if type(x) != type(y):
        raise ValueError('Operator x is not same type as operator y.')

    z = None
    term_list, coeff_list = generate_nested_commutator(order)
    for bin_str, coeff in zip(term_list, coeff_list):
        term = bin_str_to_commutator(bin_str, x, y)
        if z is None:
            z = term * coeff
        else:
            z += term * coeff

    # Return.
    return z


def bin_str_to_commutator(bin_str, x, y):
    """
    Generate nested commutator in Dynkin's style with binary string
    representation e.g. '010...' -> [X,[Y,[X, ...]]]
    """
    from openfermion.utils import commutator

    def char_to_xy(char):
        if char == '0':
            return x
        else:
            return y

    next_term = char_to_xy(bin_str[0])
    later_terms = bin_str[1:]
    if len(bin_str) == 1:
        return next_term
    else:
        return commutator(next_term, bin_str_to_commutator(later_terms, x, y))


def generate_nested_commutator(order):
    """
    using bin strings to encode nested commutators up to given order
    e.g. terms like [X,[Y,[X, ...]]] as '010...'
    """
    term_list = []
    coeff_list = []

    for i in range(1, order + 1):
        term_of_order_i = [list(x) for x in itertools.product(['0', '1'],
                                                              repeat=i)]

        # filter out trivially zero terms by checking if last two terms are
        # the same
        if i > 1:
            term_of_order_i = filter(lambda x: x[-1] != x[-2], term_of_order_i)
        term_of_order_i = ["".join(x) for x in term_of_order_i]
        term_list += term_of_order_i

    for term in term_list:
        split_bin_str = split_by_descending_edge(term)
        coeff_list.append(compute_coeff(split_bin_str))
    return term_list, coeff_list


def split_by_descending_edge(bin_str):
    """
    Split binary string representation by descending edges,
    i.e. '0101' -> '01 | 01'
    e.g. '01001101' -> ['01', '0011', '01']
    """
    prev = '0'
    split_idx = [0]

    # generate a list of indices where split needs to happen
    for idx, i in enumerate(bin_str):
        if prev == '1' and i == '0':
            split_idx.append(idx)
        prev = i

    # split by taking substrings between each two split indices
    if len(split_idx) == 1:
        return [bin_str]
    else:
        return [bin_str[i:j] for i, j in zip(split_idx, split_idx[1:]+[None])]


def compute_coeff(split_bin_str):
    """
    Compute coefficient from split binary string representation
    """
    order = len(''.join(split_bin_str))
    num_block = len(split_bin_str) - 1

    def cn(n):
        return coeff_monomial(split_bin_str, n, len(split_bin_str))

    c = sum([(-1)**(n+1) / float(n) * cn(n)
             for n in range(num_block + 1, order + 1)])
    return c/order


def coeff_monomial(split_bin_str, n, l):
    """
    Compute Coefficient for each monomial in Dynkin's formula represented by
    split binary string. Sum over all possible combinations of number of
    partitions in each block. We want to put (n) partitions inside (l + 1)
    blocks, with each block has at least one partition. Each possible
    combination is discovered and computed by the sub function
    depth_first_search.
    """

    # Python 2 compatible solution for nonlocal variable `coeff
    class context:
        coeff = 0

    def depth_first_search(split_bin_str, n, l, sol=[], cur_sum=0):
        ''' Partition an integer value of n into l bins each with min 1
        '''
        cur_idx = len(sol)
        if cur_idx < l:
            m = len(split_bin_str[cur_idx])
            n_avail = n - cur_sum
            for j in range(1, min(m, n_avail - (l - 1 - cur_idx)) + 1):
                depth_first_search(split_bin_str, n, l, sol=sol + [j],
                                   cur_sum=cur_sum + j)
        elif cur_idx == l:
            if cur_sum == n:
                partition_list = sol
                context.coeff += coeff_monomial_with_partition(split_bin_str,
                                                               partition_list)

    # start from the root
    depth_first_search(split_bin_str, n, l)
    return context.coeff


def coeff_monomial_with_partition(split_bin_str, parition_lst):
    "Given fixed parition numbers in blocks, return monomial coefficient"
    assert len(split_bin_str) == len(parition_lst)
    ret = 1
    for block, num_partition in zip(split_bin_str, parition_lst):
        cnt_x = block.count('0')
        cnt_y = block.count('1')
        ret *= coeff_for_non_descending_block(cnt_x, cnt_y, num_partition)
    return ret


def coeff_for_non_descending_block(cnt_x, cnt_y, eta):
    "Coefficient component within one block of non-descending bin_string"
    if cnt_x == 0:
        return coeff_for_consectutive_op(cnt_y, eta)
    if cnt_y == 0:
        return coeff_for_consectutive_op(cnt_x, eta)

    ret = 0
    for eta_x in range(1, eta):
        ret += (coeff_for_consectutive_op(cnt_x, eta_x) *
                coeff_for_consectutive_op(cnt_y, eta - eta_x))
    for eta_x in range(1, eta + 1):
        ret += (coeff_for_consectutive_op(cnt_x, eta_x) *
                coeff_for_consectutive_op(cnt_y, eta + 1 - eta_x))
    return ret


def coeff_for_consectutive_op(cnt_x, num_partition):
    """
    Coefficient component within only X or only Y block with given numbers of
    partition eta
    """
    ret = 0
    for num_zero in range(num_partition):
        ret += ((-1) ** num_zero * (num_partition - num_zero) ** cnt_x *
                comb(num_partition, num_zero))
    return ret / float(factorial(cnt_x))
