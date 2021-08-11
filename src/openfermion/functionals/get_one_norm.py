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
"""
Function to calculate the 1-Norm of a molecular Hamiltonian
(in the restricted formalism) after fermion-to-qubit transformation
from a MolecularData class. See https://arxiv.org/abs/2103.14753
for more information on the 1-norm.
"""

import numpy as np


def get_one_norm_mol(molecule, no_constant=True):
    r"""

    Returns the 1-Norm of a RHF or ROHF Hamiltonian described in
    https://arxiv.org/abs/2103.14753 after a fermion-to-qubit
    transformation given nuclear constant, one-body (2D np.array)
    and two-body (4D np.array) integrals in spatial orbital basis.

    Parameters
    ----------

    molecule : MolecularData class representing a molecular Hamiltonian
    constant(float) : Nuclear repulsion or adjustment to constant shift in
        Hamiltonian from integrating out core orbitals
    one_body_integrals(ndarray) : An array of the one-electron integrals having
        shape of (n_orb, n_orb), where n_orb is the number of spatial orbitals.
    two_body_integrals(ndarray) : An array of the two-electron integrals having
        shape of (n_orb, n_orb, n_orb, n_orb).
    no_constant (default: True) : If True, do not return the constant term in
                in the (majorana/qubit) Hamiltonian

    Returns
    -------
    one_norm : 1-Norm of the qubit Hamiltonian
    """
    if no_constant:
        return _get_one_norm_woconst(molecule.one_body_integrals,
                                     molecule.two_body_integrals)
    else:
        return _get_one_norm(molecule.nuclear_repulsion,
                             molecule.one_body_integrals,
                             molecule.two_body_integrals)


def get_one_norm_int(constant,
                     one_body_integrals,
                     two_body_integrals,
                     no_constant=True):
    """

    Parameters
    ----------
    constant(float) : Nuclear repulsion or adjustment to constant shift in
        Hamiltonian from integrating out core orbitals
    one_body_integrals(ndarray) : An array of the one-electron integrals having
        shape of (n_orb, n_orb), where n_orb is the number of spatial orbitals.
    two_body_integrals(ndarray) : An array of the two-electron integrals having
        shape of (n_orb, n_orb, n_orb, n_orb).
    no_constant (default: True) : If True, do not return the constant term in
                in the (majorana/qubit) Hamiltonian

    Returns
    -------
    one_norm : 1-Norm of the qubit Hamiltonian

    """
    if no_constant:
        return _get_one_norm_woconst(one_body_integrals, two_body_integrals)
    else:
        return _get_one_norm(constant, one_body_integrals, two_body_integrals)


def _get_one_norm(constant, one_body_integrals, two_body_integrals):
    """Compute 1-norm given molecular integrals"""
    n_orb = one_body_integrals.shape[0]

    htilde = constant
    for p in range(n_orb):
        htilde += one_body_integrals[p, p]
        for q in range(n_orb):
            htilde += ((1 / 2 * two_body_integrals[p, q, q, p]) -
                       (1 / 4 * two_body_integrals[p, q, p, q]))

    htildepq = np.zeros(one_body_integrals.shape)
    for p in range(n_orb):
        for q in range(n_orb):
            htildepq[p, q] = one_body_integrals[p, q]
            for r in range(n_orb):
                htildepq[p, q] += ((two_body_integrals[p, r, r, q]) -
                                   (1 / 2 * two_body_integrals[p, r, q, r]))

    one_norm = abs(htilde) + np.sum(np.absolute(htildepq))

    anti_sym_integrals = two_body_integrals - np.transpose(
        two_body_integrals, (0, 1, 3, 2))

    one_norm += 1 / 8 * np.sum(np.absolute(anti_sym_integrals))
    one_norm += 1 / 4 * np.sum(np.absolute(two_body_integrals))

    return one_norm


def _get_one_norm_woconst(one_body_integrals, two_body_integrals):
    """Compute 1-norm given molecular integrals and emit the constant term
    in the qubit Hamiltonian"""
    n_orb = one_body_integrals.shape[0]

    htildepq = np.zeros(one_body_integrals.shape)
    for p in range(n_orb):
        for q in range(n_orb):
            htildepq[p, q] = one_body_integrals[p, q]
            for r in range(n_orb):
                htildepq[p, q] += ((two_body_integrals[p, r, r, q]) -
                                   (1 / 2 * two_body_integrals[p, r, q, r]))

    one_norm = np.sum(np.absolute(htildepq))

    anti_sym_integrals = two_body_integrals - np.transpose(
        two_body_integrals, (0, 1, 3, 2))

    one_norm += 1 / 8 * np.sum(np.absolute(anti_sym_integrals))
    one_norm += 1 / 4 * np.sum(np.absolute(two_body_integrals))

    return one_norm
