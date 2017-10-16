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

"""Module to compute the second order Trotter error."""
from future.utils import iteritems

from math import sqrt, ceil
from scipy.linalg import expm

from openfermion.config import *
from openfermion.hamiltonians import MolecularData
from openfermion.ops import normal_ordered, QubitOperator


def commutator(op1, op2):
    return op1 * op2 - op2 * op1


def trivially_commutes(term_a, term_b):
    position_a = 0
    position_b = 0
    commutes = True

    term_op_a, = term_a.terms.keys()
    term_op_b, = term_b.terms.keys()

    while position_a < len(term_op_a) and position_b < len(term_op_b):
        qubit_a, action_a = term_op_a[position_a]
        qubit_b, action_b = term_op_b[position_b]

        if qubit_a > qubit_b:
            position_b += 1
        elif qubit_a < qubit_b:
            position_a += 1
        else:
            if action_a != action_b:
                commutes = not commutes
            position_a += 1
            position_b += 1

    return commutes


def trivially_double_commutes(term_a, term_b, term_c):
    """Check if the double commutator [term_a, [term_b, term_c]] is zero.

    Args:
        term_a, term_b, term_c: Single-term QubitOperators.

    Notes:
        If the sets of qubits which term_b and term_c act on, or if the
        intersection of term_a's qubits with (term_b's qubits U term_c's
        qubits) is empty, then the double commutator is trivially zero.
    """
    # determine the set of qubits each term acts on
    term_op_a, = term_a.terms.keys()
    term_op_b, = term_b.terms.keys()
    term_op_c, = term_c.terms.keys()

    qubits_a = set([index for index, _ in term_op_a])
    qubits_b = set([index for index, _ in term_op_b])
    qubits_c = set([index for index, _ in term_op_c])

    return (trivially_commutes(term_b, term_c) or
            not qubits_a.intersection(set(qubits_b.union(qubits_c))))


def error_operator(terms, series_order=2):
    """Determine the difference between the exact generator of unitary
    evolution and the approximate generator given by Trotter-Suzuki
    to the given order.

    Args:
        terms: a list of QubitTerms in the Hamiltonian to be simulated.
        series_order: the order at which to compute the BCH expansion.
            Only the second order formula is currently implemented
            (corresponding to Equation 9 of the paper).

    Returns:
        The difference between the true and effective generators of time
            evolution for a single Trotter step.

    Notes: follows Equation 9 of Poulin et al.'s work in "The Trotter Step
        Size Required for Accurate Quantum Simulation of Quantum Chemistry".
    """
    if series_order != 2:
        raise NotImplementedError
    error_operator = QubitOperator()
    for beta in range(len(terms)):
        for alpha in range(beta + 1):
            for alpha_prime in range(beta):
                if not trivially_double_commutes(terms[alpha], terms[beta],
                                                 terms[alpha_prime]):
                    double_com = commutator(terms[alpha],
                                            commutator(terms[beta],
                                                       terms[alpha_prime]))
                    error_operator += double_com
                    if alpha == beta:
                        error_operator -= double_com / 2.0

    return error_operator / 12.0


def error_bound(terms, tight=False):
    """
    Numerically upper bound the error in the ground state energy
    for the second order Trotter-Suzuki expansion.

    Args:
        terms: a list of single-term QubitOperators in the Hamiltonian
            to be simulated.
        tight: whether to use the triangle inequality to give a loose
            upper bound on the error (default) or to calculate the
            norm of the error operator.

    Returns:
        A float upper bound on norm of error in the ground state energy.

    Notes: follows Poulin et al.'s work in "The Trotter Step Size
           Required for Accurate Quantum Simulation of Quantum
           Chemistry". In particular, Equation 16 is used for a loose
           upper bound, and the norm of Equation 9 is calculated for
           a tighter bound using the error operator from error_operator.

           Possible extensions of this function would be to get the
           expectation value of the error operator with the Hartree-Fock
           state or CISD state, which can scalably bound the error in
           the ground state but much more accurately than the triangle
           inequality.
    """
    zero = QubitOperator()
    error = 0.0

    if tight:
        # return the 1-norm of the error operator (upper bound on error)
        error = sum(abs(coefficient)
                    for coefficient in error_operator(terms).terms.values())

    elif not tight:
        for alpha in range(len(terms)):
            term_a = terms[alpha]
            coefficient_a, = term_a.terms.values()
            if coefficient_a:
                error_a = 0.

                for beta in range(alpha + 1, len(terms)):
                    term_b = terms[beta]
                    coefficient_b, = term_b.terms.values()
                    if not (trivially_commutes(term_a, term_b) or
                            commutator(term_a, term_b).isclose(zero)):
                        error_a += abs(coefficient_b)

                error += 4.0 * abs(coefficient_a) * error_a ** 2

    return error


def trotter_steps_required(trotter_error_bound, time, energy_precision):
    """Determine the number of Trotter steps for accurate simulation.

    Args:
        trotter_error_bound (float): Upper bound on Trotter error in the
                                     state of interest.
        time (float): The total simulation time.
        energy_precision (float): Acceptable shift in state energy.

    Returns:
        The integer minimum number of Trotter steps required for
        simulation to the desired precision.

    Notes:
        The number of Trotter steps required is an upper bound on the
        true requirement, which may be lower.
    """
    return int(ceil(time * sqrt(trotter_error_bound / energy_precision)))
