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

"""Jordan-Wigner transform on fermionic operators."""
from __future__ import absolute_import

import itertools

from openfermion.ops import (FermionOperator, InteractionOperator,
                             QubitOperator)


def jordan_wigner(operator):
    """ Apply the Jordan-Wigner transform to a FermionOperator or
    InteractionOperator to convert to a QubitOperator.

    Operators are mapped as follows:
    a_j^\dagger -> Z_0 .. Z_{j-1} (X_j - iY_j) / 2
    a_j -> Z_0 .. Z_{j-1} (X_j + iY_j) / 2

    Returns:
        transformed_operator: An instance of the QubitOperator class.

    Warning:
        The runtime of this method is exponential in the maximum locality
        of the original FermionOperator.
    """
    if isinstance(operator, InteractionOperator):
        return jordan_wigner_interaction_op(operator)
    if not isinstance(operator, FermionOperator):
        raise TypeError("operator must be a FermionOperator or "
                        "InteractionOperator.")
    transformed_operator = QubitOperator()
    for term in operator.terms:

        # Initialize identity matrix.
        transformed_term = QubitOperator((), operator.terms[term])

        # Loop through operators, transform and multiply.
        for ladder_operator in term:
            z_factors = tuple((index, 'Z') for
                              index in range(ladder_operator[0]))
            pauli_x_component = QubitOperator(
                z_factors + ((ladder_operator[0], 'X'),), 0.5)
            if ladder_operator[1]:
                pauli_y_component = QubitOperator(
                    z_factors + ((ladder_operator[0], 'Y'),), -0.5j)
            else:
                pauli_y_component = QubitOperator(
                    z_factors + ((ladder_operator[0], 'Y'),), 0.5j)
            transformed_term *= pauli_x_component + pauli_y_component
        transformed_operator += transformed_term
    return transformed_operator


def jordan_wigner_interaction_op(iop, n_qubits=None):
    """Output InteractionOperator as QubitOperator class under JW transform.

    One could accomplish this very easily by first mapping to fermions and
    then mapping to qubits. We skip the middle step for the sake of speed.

    Returns:
        qubit_operator: An instance of the QubitOperator class.
    """
    from openfermion.utils import count_qubits
    if n_qubits is None:
        n_qubits = count_qubits(iop)
    if n_qubits < count_qubits(iop):
        raise ValueError('Invalid number of qubits specified.')

    # Initialize qubit operator as constant.
    qubit_operator = QubitOperator((), iop.constant)

    # Loop through all indices.
    for p in range(n_qubits):
        for q in range(n_qubits):

            # Handle one-body terms.
            coefficient = complex(iop[(p, 1), (q, 0)])
            if coefficient and p >= q:
                qubit_operator += coefficient * jordan_wigner_one_body(p, q)

            # Keep looping for the two-body terms.
            for r in range(n_qubits):
                for s in range(n_qubits):
                    coefficient = complex(iop[(p, 1), (q, 1), (r, 0), (s, 0)])

                    # Skip zero terms.
                    if (not coefficient) or (p == q) or (r == s):
                        continue

                    # Identify and skip one of the complex conjugates.
                    if [p, q, r, s] != [s, r, q, p]:
                        if len(set([p, q, r, s])) == 4:
                            if min(r, s) < min(p, q):
                                continue
                        elif p != r and q < p:
                                continue

                    # Handle the two-body terms.
                    transformed_term = jordan_wigner_two_body(p, q, r, s)
                    transformed_term *= coefficient
                    qubit_operator += transformed_term

    return qubit_operator


def jordan_wigner_one_body(p, q):
    """Map the term a^\dagger_p a_q + a^\dagger_q a_p to QubitOperator.

    Note that the diagonal terms are divided by a factor of 2
    because they are equal to their own Hermitian conjugate.
    """
    # Handle off-diagonal terms.
    qubit_operator = QubitOperator()
    if p != q:
        a, b = sorted([p, q])
        parity_string = tuple((z, 'Z') for z in range(a + 1, b))
        for operator in ['X', 'Y']:
            operators = ((a, operator),) + parity_string + ((b, operator),)
            qubit_operator += QubitOperator(operators, .5)

    # Handle diagonal terms.
    else:
        qubit_operator += QubitOperator((), .5)
        qubit_operator += QubitOperator(((p, 'Z'),), -.5)

    return qubit_operator


def jordan_wigner_two_body(p, q, r, s):
    """Map the term a^\dagger_p a^\dagger_q a_r a_s + h.c. to QubitOperator.

    Note that the diagonal terms are divided by a factor of two
    because they are equal to their own Hermitian conjugate.
    """
    # Initialize qubit operator.
    qubit_operator = QubitOperator()

    # Return zero terms.
    if (p == q) or (r == s):
        return qubit_operator

    # Handle case of four unique indices.
    elif len(set([p, q, r, s])) == 4:

        # Loop through different operators which act on each tensor factor.
        for operator_p, operator_q, operator_r in itertools.product(
                ['X', 'Y'], repeat=3):
            if [operator_p, operator_q, operator_r].count('X') % 2:
                operator_s = 'X'
            else:
                operator_s = 'Y'

            # Sort operators.
            [(a, operator_a), (b, operator_b),
             (c, operator_c), (d, operator_d)] = sorted(
                [(p, operator_p), (q, operator_q),
                 (r, operator_r), (s, operator_s)],
                key=lambda pair: pair[0])

            # Computer operator strings.
            operators = ((a, operator_a),)
            operators += tuple((z, 'Z') for z in range(a + 1, b))
            operators += ((b, operator_b),)
            operators += ((c, operator_c),)
            operators += tuple((z, 'Z') for z in range(c + 1, d))
            operators += ((d, operator_d),)

            # Get coefficients.
            coefficient = .125
            parity_condition = bool(operator_p != operator_q or
                                    operator_p == operator_r)
            if (p > q) ^ (r > s):
                if not parity_condition:
                    coefficient *= -1.
            elif parity_condition:
                coefficient *= -1.

            # Add term.
            qubit_operator += QubitOperator(operators, coefficient)

    # Handle case of three unique indices.
    elif len(set([p, q, r, s])) == 3:

        # Identify equal tensor factors.
        if p == r:
            a, b = sorted([q, s])
            c = p
        elif p == s:
            a, b = sorted([q, r])
            c = p
        elif q == r:
            a, b = sorted([p, s])
            c = q
        elif q == s:
            a, b = sorted([p, r])
            c = q

        # Get operators.
        parity_string = tuple((z, 'Z') for z in range(a + 1, b))
        pauli_z = QubitOperator(((c, 'Z'),))
        for operator in ['X', 'Y']:
            operators = ((a, operator),) + parity_string + ((b, operator),)

            # Get coefficient.
            if (p == s) or (q == r):
                coefficient = .25
            else:
                coefficient = -.25

            # Add term.
            hopping_term = QubitOperator(operators, coefficient)
            qubit_operator -= pauli_z * hopping_term
            qubit_operator += hopping_term

    # Handle case of two unique indices.
    elif len(set([p, q, r, s])) == 2:

        # Get coefficient.
        if p == s:
            coefficient = -.25
        else:
            coefficient = .25

        # Add terms.
        qubit_operator -= QubitOperator((), coefficient)
        qubit_operator += QubitOperator(((p, 'Z'),), coefficient)
        qubit_operator += QubitOperator(((q, 'Z'),), coefficient)
        qubit_operator -= QubitOperator(((min(q, p), 'Z'), (max(q, p), 'Z')),
                                        coefficient)

    return qubit_operator
