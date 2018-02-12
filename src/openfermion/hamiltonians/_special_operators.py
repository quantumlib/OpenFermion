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
from openfermion.ops import FermionOperator


def up_index(index):
    """Function to return up-orbital index given a spatial orbital index.

    Args:
        index (Int): spatial orbital index
    """
    return 2 * index


def down_index(index):
    """Function to return down-orbital index given a spatial orbital index.

    Args:
        index (Int): spatial orbital index
    """
    return 2 * index + 1


def sz_operator(n_spatial_orbitals, up_map=up_index, down_map=down_index):
    """Return the sz operator.

    .. math::
        \\begin{align}
        S^{z} = \\frac{1}{2}\sum_{i = 1}^{n}(n_{i, \\alpha} - n_{i, \\beta})
        \\end{align}

    Args:
        n_spatial_orbitals: number of spatial orbitals (n_qubits // 2).
        up_map: function mapping a spatial index to a spin-orbital index.
                Default is the canonical spin-up corresponds to even
                spin-orbitals and spin-down corresponds to odd spin-orbitals
        down_map: function mapping spatial index to spin-orbital index.
                  Default is canonical spin-up corresponds to even
                  spin-orbitals and spin-down corresponds to odd
                  spin-orbitals.

    Returns:
        operator (FermionOperator): corresponding to the sz operator over
        n_spatial_orbitals.

    Warnings:
        Default assumes a number occupation vector representation with even
        spin-less fermions corresponding to spin-up (alpha) and odd spin-less
        fermions corresponding to spin-down (beta).
    """
    if not isinstance(n_spatial_orbitals, int):
        raise TypeError("n_orbitals must be specified as an integer")

    operator = FermionOperator()
    n_spinless_orbitals = 2 * n_spatial_orbitals
    for ni in range(n_spatial_orbitals):
        operator += number_operator(n_spinless_orbitals, up_map(ni), 0.5) + \
                    number_operator(n_spinless_orbitals, down_map(ni), -0.5)

    return operator


def s_plus_operator(n_spatial_orbitals, up_map=up_index, down_map=down_index):
    """Return the s+ operator.

    .. math::
        \\begin{align}
        S^{+} = \sum_{i=1}^{n} a_{i, \\alpha}^{\dagger}a_{i, \\beta}
        \\end{align}

    Args:
        n_spatial_orbitals: number of spatial orbitals (n_qubits + 1 // 2).
        up_map: function mapping a spatial index to a spin-orbital index.
                Default is the canonical spin-up corresponds to even
                spin-orbitals and spin-down corresponds to odd spin-orbitals
        down_map: function mapping spatial index to spin-orbital index.
                  Default is canonical spin-up corresponds to even
                  spin-orbitals and spin-down corresponds to odd
                  spin-orbitals.

    Returns:
        operator (FermionOperator): corresponding to the s+ operator over
        n_spatial_orbitals.

    Warnings:
        Default assumes a number occupation vector representation with even
        spin-less fermions corresponding to spin-up (alpha) and odd spin-less
        fermions corresponding to spin-down (beta).
    """
    if not isinstance(n_spatial_orbitals, int):
        raise TypeError("n_orbitals must be specified as an integer")

    operator = FermionOperator()
    for ni in range(n_spatial_orbitals):
        operator += FermionOperator(((up_map(ni), 1), (down_map(ni), 0)))

    return operator


def s_minus_operator(n_spatial_orbitals, up_map=up_index, down_map=down_index):
    """Return the s+ operator.

    .. math::
        \\begin{align}
        S^{-} = \sum_{i=1}^{n} a_{i, \\beta}^{\dagger}a_{i, \\alpha}
        \\end{align}

    Args:
        n_spatial_orbitals: number of spatial orbitals (n_qubits + 1 // 2).
        up_map: function mapping a spatial index to a spin-orbital index.
                Default is the canonical spin-up corresponds to even
                spin-orbitals and spin-down corresponds to odd spin-orbitals
        down_map: function mapping spatial index to spin-orbital index.
                  Default is canonical spin-up corresponds to even
                  spin-orbitals and spin-down corresponds to odd
                  spin-orbitals.

    Returns:
        operator (FermionOperator): corresponding to the s- operator over
        n_spatial_orbitals.

    Warnings:
        Default assumes a number occupation vector representation with even
        spin-less fermions corresponding to spin-up (alpha) and odd spin-less
        fermions corresponding to spin-down (beta).
    """
    if not isinstance(n_spatial_orbitals, int):
        raise TypeError("n_orbitals must be specified as an integer")

    operator = FermionOperator()
    for ni in range(n_spatial_orbitals):
        operator += FermionOperator(((down_map(ni), 1), (up_map(ni), 0)))

    return operator


def s_squared_operator(n_spatial_orbitals):
    """Return the s^{2} operator.

    .. math::
        \\begin{align}
        S^{2} = S^{-} S^{+} + S^{z}( S^{z} + 1)
        \\end{align}

    Args:
        n_spatial_orbitals: number of spatial orbitals (n_qubits + 1 // 2).

    Returns:
        operator (FermionOperator): corresponding to the s+ operator over
        n_spatial_orbitals.

    Warnings:
        assumes a number occupation vector representation with even spin-less
        fermions corresponding to spin-up (alpha) and odd spin-less fermions
        corresponding to spin-down (beta).

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
    """Initialize a Majorana operator.

    Args:
        term(tuple): The first element of the tuple indicates the mode
            on which the Majorana operator acts, starting from zero.
            The second element of the tuple is an integer, either 1 or 0,
            indicating which type of Majorana operator it is:

                Type 1: :math:`\\frac{1}{\sqrt{2}} (a^\dagger_p + a_p)`

                Type 0: :math:`\\frac{i}{\sqrt{2}} (a^\dagger_p - a_p)`

            where the :math:`a^\dagger_p` and :math:`a_p` are the usual
            fermionic ladder operators.
            Default will result in the zero operator.
        coefficient(complex or float, optional): The coefficient of the term.
            Default value is 1.0.

    Returns:
        FermionOperator
    """
    if not isinstance(coefficient, (int, float, complex)):
        raise ValueError('Coefficient must be scalar.')

    if term is None:
        # Return zero operator
        return FermionOperator()
    elif isinstance(term, tuple):
        mode, operator_type = term
        if operator_type == 1:
            majorana_op = FermionOperator(
                ((mode, 1),), coefficient / numpy.sqrt(2.))
            majorana_op += FermionOperator(
                ((mode, 0),), coefficient / numpy.sqrt(2.))
        elif operator_type == 0:
            majorana_op = FermionOperator(
                ((mode, 1),), 1.j * coefficient / numpy.sqrt(2.))
            majorana_op -= FermionOperator(
                ((mode, 0),), 1.j * coefficient / numpy.sqrt(2.))
        else:
            raise ValueError('Operator specified incorrectly.')
        return majorana_op
    else:
        # Invalid input.
        raise ValueError('Operator specified incorrectly.')


def number_operator(n_orbitals, orbital=None, coefficient=1.):
    """Return a number operator.

    Args:
        n_orbitals (int): The number of spin-orbitals in the system.
        orbital (int, optional): The orbital on which to return the number
            operator. If None, return total number operator on all sites.
        coefficient (float): The coefficient of the term.
    Returns:
        operator (FermionOperator)
    """
    if orbital is None:
        operator = FermionOperator()
        for spin_orbital in range(n_orbitals):
            operator += number_operator(n_orbitals, spin_orbital, coefficient)
    else:
        operator = FermionOperator(((orbital, 1), (orbital, 0)), coefficient)
    return operator
