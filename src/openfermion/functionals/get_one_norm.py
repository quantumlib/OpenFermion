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
Function to calculate the 1-Norm of a molecular Hamiltonian after
fermion-to-qubit transformation from an InteractionOperator
"""

import numpy as np

from openfermion import MolecularData

def get_one_norm(mol_or_int, return_constant=True):
    r"""
    Returns the 1-Norm of the Hamiltonian after a fermion-to-qubit
    transformation given nuclear constant, one-body (2D np.array)
    and two-body (4D np.array) integrals.

    Parameters
    ----------

    mol_or_int : Tuple of (constant, one_body_integrals, two_body_integrals)
    constant : Nuclear repulsion or adjustment to constant shift in Hamiltonian
            from integrating out core orbitals
    one_body_integrals : An array of the one-electron integrals having
                shape of (n_orb, n_orb).
    two_body_integrals : An array of the two-electron integrals having
                shape of (n_orb, n_orb, n_orb, n_orb).

    -----OR----

    mol_or_int : MolecularData class object

    -----------

    return_constant (optional) : If False, do not return the constant term in
                in the (majorana/qubit) Hamiltonian

    Returns
    -------
    one_norm : 1-Norm of the Qubit Hamiltonian
    """
    if isinstance(mol_or_int, MolecularData):
        return _get_one_norm_mol(mol_or_int, return_constant)
    else:
        if return_constant:
            constant, one_body_integrals, two_body_integrals = mol_or_int
            return _get_one_norm(constant,
                                 one_body_integrals,
                                 two_body_integrals)
        else:
            _, one_body_integrals, two_body_integrals = mol_or_int
            return _get_one_norm_woconst(one_body_integrals,
                                         two_body_integrals)


def _get_one_norm_mol(molecule, return_constant):
    if return_constant:
        return _get_one_norm(molecule.nuclear_repulsion,
                             molecule.one_body_integrals,
                             molecule.two_body_integrals)
    else:
        return _get_one_norm_woconst(molecule.one_body_integrals,
                                     molecule.two_body_integrals)

def _get_one_norm(constant, one_body_integrals, two_body_integrals):
    n_orb = one_body_integrals.shape[0]

    htilde = constant
    for p in range(n_orb):
        htilde += one_body_integrals[p,p]
        for q in range(n_orb):
            htilde += ((1/2 * two_body_integrals[p,q,q,p]) -
                       (1/4 * two_body_integrals[p,q,p,q]))

    htildepq = np.zeros(one_body_integrals.shape)
    for p in range(n_orb):
        for q in range(n_orb):
            htildepq[p,q] = one_body_integrals[p,q]
            for r in range(n_orb):
                htildepq[p,q] += ((two_body_integrals[p,r,r,q]) -
                                  (1/2 * two_body_integrals[p,r,q,r]))

    one_norm = abs(htilde) + np.sum(np.absolute(htildepq))

    for p in range(n_orb):
        for q in range(n_orb):
            for r in range(n_orb):
                for s in range(n_orb):
                    if p>q and r>s:
                        one_norm += 1/2  * abs(two_body_integrals[p,q,r,s] -
                                               two_body_integrals[p,q,s,r])
                    one_norm += 1/4 * abs(two_body_integrals[p,q,r,s])

    return one_norm


def _get_one_norm_woconst(one_body_integrals, two_body_integrals):
    n_orb = one_body_integrals.shape[0]

    htildepq = np.zeros(one_body_integrals.shape)
    for p in range(n_orb):
        for q in range(n_orb):
            htildepq[p,q] = one_body_integrals[p,q]
            for r in range(n_orb):
                htildepq[p,q] += ((two_body_integrals[p,r,r,q]) -
                                  (1/2 * two_body_integrals[p,r,q,r]))

    one_norm = np.sum(np.absolute(htildepq))

    for p in range(n_orb):
        for q in range(n_orb):
            for r in range(n_orb):
                for s in range(n_orb):
                    if p>q and r>s:
                        one_norm += 1/2  * abs(two_body_integrals[p,q,r,s] -
                                               two_body_integrals[p,q,s,r])
                    one_norm += 1/4 * abs(two_body_integrals[p,q,r,s])
    return one_norm
