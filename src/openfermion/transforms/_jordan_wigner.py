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
import itertools

import numpy

from openfermion.ops import (DiagonalCoulombHamiltonian, FermionOperator,
                             InteractionOperator, MajoranaOperator,
                             QubitOperator)
from openfermion.utils import count_qubits


def jordan_wigner(operator):
    r""" Apply the Jordan-Wigner transform to a FermionOperator,
    InteractionOperator, or DiagonalCoulombHamiltonian to convert
    to a QubitOperator.

    Operators are mapped as follows:
    a_j^\dagger -> Z_0 .. Z_{j-1} (X_j - iY_j) / 2
    a_j -> Z_0 .. Z_{j-1} (X_j + iY_j) / 2

    Returns:
        transformed_operator: An instance of the QubitOperator class.

    Warning:
        The runtime of this method is exponential in the maximum locality
        of the original FermionOperator.

    Raises:
        TypeError: Operator must be a FermionOperator,
            DiagonalCoulombHamiltonian, or InteractionOperator.
    """
    if isinstance(operator, FermionOperator):
        return jordan_wigner_fermion_operator(operator)
    if isinstance(operator, MajoranaOperator):
        return jordan_wigner_majorana_operator(operator)
    if isinstance(operator, DiagonalCoulombHamiltonian):
        return jordan_wigner_diagonal_coulomb_hamiltonian(operator)
    if isinstance(operator, InteractionOperator):
        return jordan_wigner_interaction_op(operator)
    raise TypeError("Operator must be a FermionOperator, "
                    "MajoranaOperator, "
                    "DiagonalCoulombHamiltonian, or "
                    "InteractionOperator.")


def jordan_wigner_fermion_operator(operator):
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


def jordan_wigner_majorana_operator(operator):
    transformed_operator = QubitOperator()
    for term, coeff in operator.terms.items():
        transformed_term = QubitOperator((), coeff)
        for majorana_index in term:
            q, b = divmod(majorana_index, 2)
            z_string = tuple((i, 'Z') for i in range(q))
            bit_flip_op = 'Y' if b else 'X'
            transformed_term *= QubitOperator(z_string + ((q, bit_flip_op),))
        transformed_operator += transformed_term
    return transformed_operator


def jordan_wigner_diagonal_coulomb_hamiltonian(operator):
    n_qubits = count_qubits(operator)
    qubit_operator = QubitOperator((), operator.constant)

    # Transform diagonal one-body terms
    for p in range(n_qubits):
        coefficient = operator.one_body[p, p] + operator.two_body[p, p]
        qubit_operator += QubitOperator(((p, 'Z'),), -.5 * coefficient)
        qubit_operator += QubitOperator((), .5 * coefficient)

    # Transform other one-body terms and two-body terms
    for p, q in itertools.combinations(range(n_qubits), 2):
        # One-body
        real_part = numpy.real(operator.one_body[p, q])
        imag_part = numpy.imag(operator.one_body[p, q])
        parity_string = [(i, 'Z') for i in range(p + 1, q)]
        qubit_operator += QubitOperator(
                [(p, 'X')] + parity_string + [(q, 'X')], .5 * real_part)
        qubit_operator += QubitOperator(
                [(p, 'Y')] + parity_string + [(q, 'Y')], .5 * real_part)
        qubit_operator += QubitOperator(
                [(p, 'Y')] + parity_string + [(q, 'X')], .5 * imag_part)
        qubit_operator += QubitOperator(
                [(p, 'X')] + parity_string + [(q, 'Y')], -.5 * imag_part)

        # Two-body
        coefficient = operator.two_body[p, q]
        qubit_operator += QubitOperator(((p, 'Z'), (q, 'Z')), .5 * coefficient)
        qubit_operator += QubitOperator((p, 'Z'), -.5 * coefficient)
        qubit_operator += QubitOperator((q, 'Z'), -.5 * coefficient)
        qubit_operator += QubitOperator((), .5 * coefficient)

    return qubit_operator


def jordan_wigner_interaction_op(iop, n_qubits=None):
    """Output InteractionOperator as QubitOperator class under JW transform.

    One could accomplish this very easily by first mapping to fermions and
    then mapping to qubits. We skip the middle step for the sake of speed.

    This only works for real InteractionOperators (no complex numbers).

    Returns:
        qubit_operator: An instance of the QubitOperator class.
    """
    if n_qubits is None:
        n_qubits = count_qubits(iop)
    if n_qubits < count_qubits(iop):
        raise ValueError('Invalid number of qubits specified.')

    # Initialize qubit operator as constant.
    qubit_operator = QubitOperator((), iop.constant)

    # Transform diagonal one-body terms
    for p in range(n_qubits):
        coefficient = iop[(p, 1), (p, 0)]
        qubit_operator += jordan_wigner_one_body(p, p, coefficient)

    # Transform other one-body terms and "diagonal" two-body terms
    for p, q in itertools.combinations(range(n_qubits), 2):
        # One-body
        coefficient = .5 * (iop[(p, 1), (q, 0)] + iop[(q, 1),
                                                      (p, 0)].conjugate())
        qubit_operator += jordan_wigner_one_body(p, q, coefficient)

        # Two-body
        coefficient = (iop[(p, 1), (q, 1), (p, 0), (q, 0)] -
                       iop[(p, 1), (q, 1), (q, 0), (p, 0)] -
                       iop[(q, 1), (p, 1), (p, 0), (q, 0)] +
                       iop[(q, 1), (p, 1), (q, 0), (p, 0)])
        qubit_operator += jordan_wigner_two_body(p, q, p, q, coefficient)

    # Transform the rest of the two-body terms
    for (p, q), (r, s) in itertools.combinations(
            itertools.combinations(range(n_qubits), 2),
            2):
        coefficient = 0.5 * (
                iop[(p, 1), (q, 1), (r, 0), (s, 0)] +
                iop[(s, 1), (r, 1), (q, 0), (p, 0)].conjugate() -
                iop[(p, 1), (q, 1), (s, 0), (r, 0)] -
                iop[(r, 1), (s, 1), (q, 0), (p, 0)].conjugate() -
                iop[(q, 1), (p, 1), (r, 0), (s, 0)] -
                iop[(s, 1), (r, 1), (p, 0), (q, 0)].conjugate() +
                iop[(q, 1), (p, 1), (s, 0), (r, 0)] +
                iop[(r, 1), (s, 1), (p, 0), (q, 0)].conjugate())
        qubit_operator += jordan_wigner_two_body(p, q, r, s, coefficient)

    return qubit_operator


def jordan_wigner_one_body(p, q, coefficient=1.):
    r"""Map the term a^\dagger_p a_q + h.c. to QubitOperator.

    Note that the diagonal terms are divided by a factor of 2
    because they are equal to their own Hermitian conjugate.
    """
    # Handle off-diagonal terms.
    qubit_operator = QubitOperator()
    if p != q:
        if p > q:
            p, q = q, p
            coefficient = coefficient.conjugate()
        parity_string = tuple((z, 'Z') for z in range(p + 1, q))
        for c, (op_a, op_b) in [(coefficient.real, 'XX'),
                                (coefficient.real, 'YY'),
                                (coefficient.imag, 'YX'),
                                (-coefficient.imag, 'XY')]:
            operators = ((p, op_a),) + parity_string + ((q, op_b),)
            qubit_operator += QubitOperator(operators, .5 * c)

    # Handle diagonal terms.
    else:
        qubit_operator += QubitOperator((), .5 * coefficient)
        qubit_operator += QubitOperator(((p, 'Z'),), -.5 * coefficient)

    return qubit_operator


def jordan_wigner_two_body(p, q, r, s, coefficient=1.):
    r"""Map the term a^\dagger_p a^\dagger_q a_r a_s + h.c. to QubitOperator.

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
        if (p > q) ^ (r > s):
            coefficient *= -1
        # Loop through different operators which act on each tensor factor.
        for ops in itertools.product('XY', repeat=4):
            # Get coefficients.
            if ops.count('X') % 2:
                coeff = .125 * coefficient.imag
                if ''.join(ops) in ['XYXX', 'YXXX', 'YYXY', 'YYYX']:
                    coeff *= -1
            else:
                coeff = .125 * coefficient.real
                if ''.join(ops) not in ['XXYY', 'YYXX']:
                    coeff *= -1
            if not coeff:
                continue

            # Sort operators.
            [(a, operator_a), (b, operator_b),
             (c, operator_c), (d, operator_d)] = sorted(zip([p, q, r, s], ops))

            # Compute operator strings.
            operators = ((a, operator_a),)
            operators += tuple((z, 'Z') for z in range(a + 1, b))
            operators += ((b, operator_b),)
            operators += ((c, operator_c),)
            operators += tuple((z, 'Z') for z in range(c + 1, d))
            operators += ((d, operator_d),)

            # Add term.
            qubit_operator += QubitOperator(operators, coeff)

    # Handle case of three unique indices.
    elif len(set([p, q, r, s])) == 3:

        # Identify equal tensor factors.
        if p == r:
            if q > s:
                a, b = s, q
                coefficient = -coefficient.conjugate()
            else:
                a, b = q, s
                coefficient = -coefficient
            c = p
        elif p == s:
            if q > r:
                a, b = r, q
                coefficient = coefficient.conjugate()
            else:
                a, b = q, r
            c = p
        elif q == r:
            if p > s:
                a, b = s, p
                coefficient = coefficient.conjugate()
            else:
                a, b = p, s
            c = q
        elif q == s:
            if p > r:
                a, b = r, p
                coefficient = -coefficient.conjugate()
            else:
                a, b = p, r
                coefficient = -coefficient
            c = q

        # Get operators.
        parity_string = tuple((z, 'Z') for z in range(a + 1, b))
        pauli_z = QubitOperator(((c, 'Z'),))
        for c, (op_a, op_b) in [(coefficient.real, 'XX'),
                                (coefficient.real, 'YY'),
                                (coefficient.imag, 'YX'),
                                (-coefficient.imag, 'XY')]:
            operators = ((a, op_a),) + parity_string + ((b, op_b),)
            if not c:
                continue

            # Add term.
            hopping_term = QubitOperator(operators, c / 4)
            qubit_operator -= pauli_z * hopping_term
            qubit_operator += hopping_term

    # Handle case of two unique indices.
    elif len(set([p, q, r, s])) == 2:

        # Get coefficient.
        if p == s:
            coeff = -.25 * coefficient
        else:
            coeff = .25 * coefficient

        # Add terms.
        qubit_operator -= QubitOperator((), coeff)
        qubit_operator += QubitOperator(((p, 'Z'),), coeff)
        qubit_operator += QubitOperator(((q, 'Z'),), coeff)
        qubit_operator -= QubitOperator(((min(q, p), 'Z'), (max(q, p), 'Z')),
                                        coeff)

    return qubit_operator
