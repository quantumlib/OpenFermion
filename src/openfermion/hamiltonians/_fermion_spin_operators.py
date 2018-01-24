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

"""Implementing angular momentum generators."""
from openfermion.ops import FermionOperator, number_operator


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

    SZ = 0.5 * \sum_{i = 1}^{n} n_{i, \alpha} - n_{i, \beta}

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

    S+ = \sum_{i=1}^{n} a_{i, \alpha}^{\dagger}a_{i, \beta}

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

    S- = \sum_{i=1}^{n} a_{i, \beta}^{\dagger}a_{i, \alpha}

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
        operator += FermionOperator(((down_map(ni), 1), (up_map(ni), 0)))

    return operator


def s_squared_operator(n_spatial_orbitals):
    """Return the s^{2} operator.

    S^{2} = S- S+ + Sz( SZ + 1)

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