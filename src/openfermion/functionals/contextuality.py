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
"""tests for whether operators are contextual contextuality"""
from typing import List

from openfermion.ops.operators import QubitOperator


def _commutes(operator1: QubitOperator, operator2: QubitOperator) -> bool:
    return operator1 * operator2 == operator2 * operator1


def _non_fully_commuting_terms(hamiltonian: QubitOperator
                              ) -> List[QubitOperator]:
    terms = list([QubitOperator(key) for key in hamiltonian.terms.keys()])
    T = []  # will contain the subset of terms that do not
    # commute universally in terms
    for i in range(len(terms)):
        if any(not _commutes(terms[i], terms[j]) for j in range(len(terms))):
            T.append(terms[i])
    return T


def is_contextual(hamiltonian: QubitOperator) -> bool:
    """
    Determine whether a hamiltonian (instance of QubitOperator) is contextual,
    in the sense of https://arxiv.org/abs/1904.02260.

    Args:
        hamiltonian (QubitOperator): the hamiltonian whose
            contextuality is to be evaluated.

    Returns:
        boolean indicating whether hamiltonian is contextual or not.
    """
    if not isinstance(hamiltonian, QubitOperator):
        raise TypeError('Only supported for QubitOperators.')
    T = _non_fully_commuting_terms(hamiltonian)
    # Search in T for triples in which exactly one pair anticommutes;
    # if any exist, hamiltonian is contextual.
    for i in range(len(T)):  # WLOG, i indexes the operator that (putatively)
        # commutes with both others.
        for j in range(len(T)):
            for k in range(j + 1, len(T)):  # Ordering of j, k does not matter.
                if i!=j and i!=k and _commutes(T[i],T[j]) \
                    and _commutes(T[i],T[k]) and \
                        not _commutes(T[j],T[k]):
                    return True
    return False
