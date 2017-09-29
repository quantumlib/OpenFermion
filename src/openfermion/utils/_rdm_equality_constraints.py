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

"""Constraints on fermionic reduced density matrices"""
from openfermion.ops import FermionOperator


def one_body_fermion_constraints(n_orbitals, n_fermions):
    """Generates one-body positivity constraints on fermionic RDMs.

        The specific constraints implemented are known positivity constraints
        on the one-fermion reduced density matrices. Constraints are generated
        in the form of FermionOperators whose expectation value is known to be
        zero for any N-Representable state. Generators are used for efficiency.

    Args:
        n_orbitals(int): number of spin-orbitals on which operators act.
        n_fermions(int): number of fermions in the system.

    Yields:
        Constraint is a FermionOperator with zero expectation value.
    """
    # One-RDM trace condition.
    constraint_operator = FermionOperator()
    for i in range(n_orbitals):
        constraint_operator += FermionOperator(((i, 1), (i, 0)))
    if len(constraint_operator.terms):
        constraint_operator -= FermionOperator((), n_fermions)
        yield constraint_operator

    # One-RDM Hermiticity condition.
    for i in range(n_orbitals):
        for j in range(i + 1, n_orbitals):
            constraint_operator = FermionOperator(((i, 1), (j, 0)))
            constraint_operator -= FermionOperator(((j, 1), (i, 0)))
            if len(constraint_operator.terms):
                yield constraint_operator


def two_body_fermion_constraints(n_orbitals, n_fermions):
    """Generates two-body positivity constraints on fermionic RDMs.

        The specific constraints implemented are known positivity constraints
        on the two-fermion reduced density matrices. Constraints are generated
        in the form of FermionOperators whose expectation value is known to be
        zero for any N-Representable state. Generators are used for efficiency.

    Args:
        n_orbitals(int): number of spin-orbitals on which operators act.
        n_fermions(int): number of fermions in the system.

    Yields:
        Constraint is a FermionOperator with zero expectation value.
    """
    # Two-body trace condition.
    constraint_operator = FermionOperator()
    for i in range(n_orbitals):
        for j in range(n_orbitals):
            constraint_operator += FermionOperator(
                ((i, 1), (j, 1), (j, 0), (i, 0)))
    if len(constraint_operator.terms):
        constraint_operator -= FermionOperator(
            (), n_fermions * (n_fermions - 1))
        yield constraint_operator

    # Two-body Hermiticity condition.
    for ij in range(n_orbitals ** 2):
        i, j = (ij // n_orbitals), (ij % n_orbitals)
        for kl in range(ij + 1, n_orbitals ** 2):
            k, l = (kl // n_orbitals), (kl % n_orbitals)
            constraint_operator = FermionOperator(
                ((i, 1), (j, 1), (l, 0), (k, 0)))
            constraint_operator -= FermionOperator(
                ((k, 1), (l, 1), (j, 0), (i, 0)))
            if len(constraint_operator.terms):
                yield constraint_operator

    # Contraction to One-RDM from Two-RDM.
    for i in range(n_orbitals):
        for j in range(n_orbitals):
            constraint_operator = FermionOperator()
            for p in range(n_orbitals):
                constraint_operator += FermionOperator(
                    ((i, 1), (p, 1), (p, 0), (j, 0)))
            constraint_operator += FermionOperator(
                ((i, 1), (j, 0)), -(n_fermions - 1))
            if len(constraint_operator.terms):
                yield constraint_operator

    # Linear relations between two-particle matrices.
    for ij in range(n_orbitals ** 2):
        i, j = (ij // n_orbitals), (ij % n_orbitals)
        for kl in range(ij, n_orbitals ** 2):
            k, l = (kl // n_orbitals), (kl % n_orbitals)

            # G-matrix condition.
            constraint_operator = FermionOperator(
                ((i, 1), (k, 0)), 1.0 * (j == l))
            constraint_operator += FermionOperator(
                ((i, 1), (l, 1), (k, 0), (j, 0)))
            constraint_operator += FermionOperator(
                ((i, 1), (l, 1), (j, 0), (k, 0)))
            constraint_operator -= FermionOperator(
                ((i, 1), (k, 0)), 1.0 * (j == l))
            if len(constraint_operator.terms):
                yield constraint_operator
