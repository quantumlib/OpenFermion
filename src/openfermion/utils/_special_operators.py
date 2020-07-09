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

"""Commonly used operators (mainly instances of SymbolicOperator)."""

import numpy

from openfermion.ops import BosonOperator, FermionOperator, down_index, up_index

def s_plus_operator(n_spatial_orbitals):
    r"""Return the s+ operator.

    .. math::
        \begin{align}
        S^{+} = \sum_{i=1}^{n} a_{i, \alpha}^{\dagger}a_{i, \beta}
        \end{align}

    Args:
        n_spatial_orbitals: number of spatial orbitals (n_qubits + 1 // 2).

    Returns:
        operator (FermionOperator): corresponding to the s+ operator over
        n_spatial_orbitals.

    Note:
        The indexing convention used is that even indices correspond to
        spin-up (alpha) modes and odd indices correspond to spin-down (beta)
        modes.
    """
    if not isinstance(n_spatial_orbitals, int):
        raise TypeError("n_orbitals must be specified as an integer")

    operator = FermionOperator()
    for ni in range(n_spatial_orbitals):
        operator += FermionOperator(((up_index(ni), 1), (down_index(ni), 0)))

    return operator


def s_minus_operator(n_spatial_orbitals):
    r"""Return the s+ operator.

    .. math::
        \begin{align}
        S^{-} = \sum_{i=1}^{n} a_{i, \beta}^{\dagger}a_{i, \alpha}
        \end{align}

    Args:
        n_spatial_orbitals: number of spatial orbitals (n_qubits + 1 // 2).

    Returns:
        operator (FermionOperator): corresponding to the s- operator over
        n_spatial_orbitals.

    Note:
        The indexing convention used is that even indices correspond to
        spin-up (alpha) modes and odd indices correspond to spin-down (beta)
        modes.
    """
    if not isinstance(n_spatial_orbitals, int):
        raise TypeError("n_orbitals must be specified as an integer")

    operator = FermionOperator()
    for ni in range(n_spatial_orbitals):
        operator += FermionOperator(((down_index(ni), 1), (up_index(ni), 0)))

    return operator


def sx_operator(n_spatial_orbitals):
    r"""Return the sx operator.

    .. math::
        \begin{align}
        S^{x} = \frac{1}{2}\sum_{i = 1}^{n}(S^{+} + S^{-})
        \end{align}

    Args:
        n_spatial_orbitals: number of spatial orbitals (n_qubits // 2).

    Returns:
        operator (FermionOperator): corresponding to the sx operator over
        n_spatial_orbitals.

    Note:
        The indexing convention used is that even indices correspond to
        spin-up (alpha) modes and odd indices correspond to spin-down (beta)
        modes.
    """
    if not isinstance(n_spatial_orbitals, int):
        raise TypeError("n_orbitals must be specified as an integer")

    operator = FermionOperator()
    for ni in range(n_spatial_orbitals):
        operator += FermionOperator(((up_index(ni), 1), (down_index(ni), 0)),
                                    .5)
        operator += FermionOperator(((down_index(ni), 1), (up_index(ni), 0)),
                                    .5)

    return operator


def sy_operator(n_spatial_orbitals):
    r"""Return the sy operator.

    .. math::
        \begin{align}
        S^{y} = \frac{-i}{2}\sum_{i = 1}^{n}(S^{+} - S^{-})
        \end{align}

    Args:
        n_spatial_orbitals: number of spatial orbitals (n_qubits // 2).

    Returns:
        operator (FermionOperator): corresponding to the sx operator over
        n_spatial_orbitals.

    Note:
        The indexing convention used is that even indices correspond to
        spin-up (alpha) modes and odd indices correspond to spin-down (beta)
        modes.
    """
    if not isinstance(n_spatial_orbitals, int):
        raise TypeError("n_orbitals must be specified as an integer")

    operator = FermionOperator()
    for ni in range(n_spatial_orbitals):
        operator += FermionOperator(((up_index(ni), 1), (down_index(ni), 0)),
                                    -.5j)
        operator += FermionOperator(((down_index(ni), 1), (up_index(ni), 0)),
                                    .5j)

    return operator


def sz_operator(n_spatial_orbitals):
    r"""Return the sz operator.

    .. math::
        \begin{align}
        S^{z} = \frac{1}{2}\sum_{i = 1}^{n}(n_{i, \alpha} - n_{i, \beta})
        \end{align}

    Args:
        n_spatial_orbitals: number of spatial orbitals (n_qubits // 2).

    Returns:
        operator (FermionOperator): corresponding to the sz operator over
        n_spatial_orbitals.

    Note:
        The indexing convention used is that even indices correspond to
        spin-up (alpha) modes and odd indices correspond to spin-down (beta)
        modes.
    """
    if not isinstance(n_spatial_orbitals, int):
        raise TypeError("n_orbitals must be specified as an integer")

    operator = FermionOperator()
    n_spinless_orbitals = 2 * n_spatial_orbitals
    for ni in range(n_spatial_orbitals):
        operator += (number_operator(n_spinless_orbitals,
                                     up_index(ni), 0.5) +
                     number_operator(n_spinless_orbitals,
                                     down_index(ni), -0.5))

    return operator


def s_squared_operator(n_spatial_orbitals):
    r"""Return the s^{2} operator.

    .. math::
        \begin{align}
        S^{2} = S^{-} S^{+} + S^{z}( S^{z} + 1)
        \end{align}

    Args:
        n_spatial_orbitals: number of spatial orbitals (n_qubits + 1 // 2).

    Returns:
        operator (FermionOperator): corresponding to the s+ operator over
        n_spatial_orbitals.

    Note:
        The indexing convention used is that even indices correspond to
        spin-up (alpha) modes and odd indices correspond to spin-down (beta)
        modes.
    """
    if not isinstance(n_spatial_orbitals, int):
        raise TypeError("n_orbitals must be specified as an integer")

    fermion_identity = FermionOperator(())
    operator = (s_minus_operator(n_spatial_orbitals) *
                s_plus_operator(n_spatial_orbitals))
    operator += (sz_operator(n_spatial_orbitals) *
                 (sz_operator(n_spatial_orbitals) + fermion_identity))
    return operator


def majorana_operator(term=None, coefficient=1.):
    r"""Initialize a Majorana operator.

    Args:
        term(tuple or string): The first element of the tuple indicates the
            mode on which the Majorana operator acts, starting from zero.
            The second element of the tuple is an integer, either 0 or 1,
            indicating which type of Majorana operator it is:

                Type 0: :math:`a^\dagger_p + a_p`

                Type 1: :math:`i (a^\dagger_p - a_p)`

            where the :math:`a^\dagger_p` and :math:`a_p` are the usual
            fermionic ladder operators.
            Alternatively, one can provide a string such as 'c2', which
            is a Type 0 operator on mode 2, or 'd3', which is a Type 1
            operator on mode 3.
            Default will result in the zero operator.
        coefficient(complex or float, optional): The coefficient of the term.
            Default value is 1.0.

    Returns:
        FermionOperator
    """
    if not isinstance(coefficient, (int, float, complex)):
        raise ValueError('Coefficient must be scalar.')

    # If term is a string, convert it to a tuple
    if isinstance(term, str):
        operator_type = term[0]
        mode = int(term[1:])
        if operator_type == 'c':
            operator_type = 0
        elif operator_type == 'd':
            operator_type = 1
        else:
            raise ValueError('Invalid operator type: {}'.format(operator_type))
        term = (mode, operator_type)

    # Process term

    # Zero operator
    if term is None:
        return FermionOperator()

    # Tuple
    if isinstance(term, tuple):
        mode, operator_type = term

        if operator_type == 0:
            majorana_op = FermionOperator(((mode, 1),), coefficient)
            majorana_op += FermionOperator(((mode, 0),), coefficient)
        elif operator_type == 1:
            majorana_op = FermionOperator(((mode, 1),), 1.j * coefficient)
            majorana_op -= FermionOperator(((mode, 0),), 1.j * coefficient)
        else:
            raise ValueError('Invalid operator type: {}'.format(
                str(operator_type)))

        return majorana_op

    # Invalid input.
    else:
        raise ValueError('Operator specified incorrectly.')


def number_operator(n_modes, mode=None, coefficient=1., parity=-1):
    """Return a fermionic or bosonic number operator.

    Args:
        n_modes (int): The number of modes in the system.
        mode (int, optional): The mode on which to return the number
            operator. If None, return total number operator on all sites.
        coefficient (float): The coefficient of the term.
        parity (int): Returns the fermionic number operator
                    if parity=-1 (default),
                    and returns the bosonic number operator
                    if parity=1.
    Returns:
        operator (BosonOperator or FermionOperator)
    """
    if parity == -1:
        Op = FermionOperator
    elif parity == 1:
        Op = BosonOperator

    if mode is None:
        operator = Op()
        for m in range(n_modes):
            operator += number_operator(n_modes, m, coefficient, parity)
    else:
        operator = Op(((mode, 1), (mode, 0)), coefficient)
    return operator
