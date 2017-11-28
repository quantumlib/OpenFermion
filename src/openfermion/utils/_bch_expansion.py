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

"""Module to efficiently compute the Baker–Campbell–Hausdorff formula."""

import itertools
import numpy as np
from scipy.misc import comb, factorial

# from openfermion.utils import commutator

def bch_expand(x, y, order):
    """Compute log[e^x e^y] using the Baker-Campbell-Hausdorff formula
    Args:
        x: An operator for which multiplication and addition are supported.
            For instance, a QubitOperator, FermionOperator or numpy array.
        y: The same type as x.
        order(int): The order to truncate the BCH expansions.

    Returns:
        z: The truncated BCH operator.

    Raises:
        ValueError: operator x is not same type as operator y.
        ValueError: invalid order parameter.
        ValueError: order exceeds maximum order supported.
    """

    max_order = 11
    if order > max_order:
        raise ValueError('Order exceeds maximum order supported.')
    if type(x) != type(y):
        raise ValueError('Operator x is not same type as operator y.')
    elif (not isinstance(order, int)) or order < 0:
        raise ValueError('Invalid order parameter.')

    z = None
    terms, coeff = generate_nested_commutator(order)
    for bin_str, c in zip(terms, coeff):
        t = bin_str_to_commutator(bin_str, x, y)
        if z is None:
            z = t * c
        else:
            z += t * c

    # Return.
    return z

def bin_str_to_commutator(bin_str, x, y):
    char_to_xy = lambda char: x if char == '0' else y
    op = char_to_xy(bin_str[0])

    if len(bin_str) == 1:
        return op
    else:
        return commutator(op, bin_str_to_commutator(bin_str[1:], x, y))

def generate_nested_commutator(order):
    """
    using bin strings to encode nested commutators like [X,[Y,[X, ...]]] as '010...'
    """
    terms = []
    coeff = []

    for i in range(1, order + 1):
        t = [list(x) for x in itertools.product(['0', '1'], repeat=i)]
        if i > 1:
            t = filter(lambda x: x[-1] != x[-2], t)
        t = ["".join(x) for x in t]
        terms += t

    for t in terms:
        split_bin_str = split_by_descending_edge(t)
        coeff.append(compute_coeff(split_bin_str))
    return terms, coeff

def split_by_descending_edge(bin_str):
    prev = '0'
    split_idx = [0]

    for idx, i in enumerate(bin_str):
        if prev == '1' and i == '0':
            split_idx.append(idx)
        prev = i

    if len(split_idx) == 1:
        return [bin_str]
    else:
        return [bin_str[i:j] for i, j in zip(split_idx, split_idx[1:]+[None])]

def compute_coeff(split_bin_str):
    N = len(''.join(split_bin_str))
    l = len(split_bin_str) - 1
    cn = lambda n: dfs_root(split_bin_str, n, len(split_bin_str))
    c = sum([(-1)**(n+1) / n * cn(n) for n in range(l+1, N+1)])
    return c/N

def dfs_root(split_bin_str, n, l):
    cn = 0
    def dfs(split_bin_str, n, l, sol=[], cur_sum=0):
        ''' Partition an integer value of n into l bins each with min 1
        '''
        nonlocal cn
        cur_idx = len(sol)
        if cur_idx < l: 
            m = len(split_bin_str[cur_idx])
            n_avail = n - cur_sum
            for j in range(1, min(m, n_avail - (l - 1 - cur_idx)) + 1):
                dfs(split_bin_str, n, l, sol=sol + [j], cur_sum=cur_sum + j)
        elif cur_idx == l:
            if cur_sum == n:
                eta_lst = sol
                cn += compute_block(split_bin_str, eta_lst)
    dfs(split_bin_str, n, l)
    return cn

def compute_block(split_bin_str, eta_lst):
    assert len(split_bin_str) == len(eta_lst)
    ret = 1
    for block, eta in zip(split_bin_str, eta_lst):
        cnt_x = block.count('0')
        cnt_y = block.count('1')
        ret *= g(cnt_x, cnt_y, eta)
    return ret


def g(cnt_x, cnt_y, eta):
    "Coefficient component within one block of non-descending bin_string"

    if cnt_x == 0:
        return f(cnt_y, eta)
    if cnt_y == 0:
        return f(cnt_x, eta)

    ret = 0
    for eta_x in range(1, eta):
        ret += f(cnt_x, eta_x) * f(cnt_y, eta - eta_x)
    for eta_x in range(1, eta + 1):
        ret += f(cnt_x, eta_x) * f(cnt_y, eta + 1 - eta_x)
    return ret


def f(cnt_x, eta_x):
    "Coefficient component within only X or only Y block with given numbers of partition eta"
    ret = 0
    for z in range(eta_x):
        ret += (-1)**z * (eta_x - z)**cnt_x * comb(eta_x, z)
    return ret / factorial(cnt_x)
