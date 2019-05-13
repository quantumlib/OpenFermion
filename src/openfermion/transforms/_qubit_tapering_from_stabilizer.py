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

"""Functions to taper qubits from stabilizer in quantum codes."""
import numpy
from openfermion.ops import QubitOperator
from openfermion.utils import count_qubits


class TaperQubitError(Exception):
    """Taper qibot error class."""

    def __init__(self, message):
        """Throw custom errors after taper off qubits."""
        super().__init__(message)


def _check_commuting_stabilizers(stabilizer_list, msg, thres=1e-6):
    """
    Auxiliar function to check that stabilizers commute.

    If a two stabilizers anti-commute the product of them
    will return a imaginary coefficint.
    This function checks the list of stabilizers (QubitOperator)
    and raises and error if a complex number is found in
    any of the coefficients.
    """
    for stab in stabilizer_list:
        if abs(numpy.imag(list(stab.terms.values())[0])) >= thres:
            raise TaperQubitError(msg)


def _check_stabilizer_linearity(stabilizer_list, msg):
    """
    Auxiliar function to check that stabilizers commute.

    If two stabilizer are linearly depedent the result
    after their product will be the identity.
    This function checks the list of stabilizers (QubitOperator)
    and raises and error if the identity is found.
    """
    for stab in stabilizer_list:
        if list(stab.terms.keys())[0] == ():
            raise TaperQubitError(msg)


def _fix_single_term(term, position, fixed_op, other_op, stabilizer):
    """
    Auxiliary function for term reductions.

    Uses stabilizer to fix the action of a QubitOperator on the selected
    qubit to either the identity or a specific Pauli operator.
    """
    pauli_tuple = list(term.terms)[0]
    if (position, fixed_op) in pauli_tuple or (position,
                                               other_op) in pauli_tuple:
        return term * stabilizer
    else:
        return term


def _lookup_term(pauli_string, updated_terms_1, updated_terms_2):
    """
    Auxiliar function for reducing terms keeping length.

    This function checks the length of the original Pauli strings,
    compares it to the updated Pauli strings, and keeps the shortest operators.
    """
    pauli_op = QubitOperator(pauli_string)
    length = len(pauli_string)

    for x in numpy.arange(len(updated_terms_1)):
        if (pauli_string == updated_terms_2[x] and
                (length > len(list(updated_terms_1[x].terms)[0]))):
            pauli_op = updated_terms_1[x]
            length = len(list(updated_terms_1[x].terms)[0])
    return pauli_op


def _reduce_terms(terms, stabilizer_list, maintain_length,
                  manual_input, fixed_positions):
    """
    Perfom the term reduction by stabilizer conditions.

    Auxiliar function of reduce_number_of_terms.
    """
    # Initialize fixed_position as an empty list to avoid conflict with
    # fixed_positions.
    if manual_input is False:
        fixed_positions = []

    # We need the index of the stabilizer to connect it to the fixed qubit.
    for i, x in enumerate(stabilizer_list):
        selected_stab = list(stabilizer_list[0].terms)[0]

        if manual_input is False:
            # Finds qubit position and its Pauli.
            for qubit_pauli in selected_stab:
                if qubit_pauli[0] not in fixed_positions:
                    fixed_positions += [qubit_pauli[0]]
                    fixed_op = qubit_pauli[1]
                    break
        else:
            # Finds Pauli of the fixed qubit.
            for qubit_pauli in selected_stab:
                if qubit_pauli[0] == fixed_positions[i]:
                    fixed_op = qubit_pauli[1]
                    break

        if fixed_op in ['X', 'Z']:
            other_op = 'Y'
        else:
            other_op = 'X'

        new_terms = QubitOperator()
        for qubit_pauli in terms:
            new_terms += _fix_single_term(qubit_pauli, fixed_positions[i],
                                          fixed_op,
                                          other_op, stabilizer_list[0])
        updated_stabilizers = []
        for update_stab in stabilizer_list[1:]:
            updated_stabilizers += [_fix_single_term(update_stab,
                                                     fixed_positions[i],
                                                     fixed_op,
                                                     other_op,
                                                     stabilizer_list[0])]

        # Update terms and stabilizer list.
        terms = new_terms
        stabilizer_list = updated_stabilizers

        _check_stabilizer_linearity(stabilizer_list,
                                    msg='Linearly-dependent stabilizers.')
        _check_commuting_stabilizers(stabilizer_list,
                                     msg='Stabilizers anti-commute.')

    return terms, fixed_positions


def _reduce_terms_keep_length(terms, stabilizer_list, maintain_length,
                              manual_input, fixed_positions):
    """
    Perfom the term reduction by stabilizer conditions.

    Auxiliar funtion of reduce_number_of_terms that returns the
    Pauli strings with the same length as the starting operator.
    """
    term_list_duplicate = list(terms.terms)
    term_list = [QubitOperator(x) for x in term_list_duplicate]

    if manual_input is False:
        fixed_positions = []

    for i, x in enumerate(stabilizer_list):
        selected_stab = list(stabilizer_list[0].terms)[0]

        if manual_input is False:
            # Finds qubit position and its Pauli.
            for qubit_pauli in selected_stab:
                if qubit_pauli[0] not in fixed_positions:
                    fixed_positions += [qubit_pauli[0]]
                    fixed_op = qubit_pauli[1]
                    break
        else:
            # Finds Pauli of the fixed qubit.
            for qubit_pauli in selected_stab:
                if qubit_pauli[0] == fixed_positions[i]:
                    fixed_op = qubit_pauli[1]
                    break

        if fixed_op in ['X', 'Z']:
            other_op = 'Y'
        else:
            other_op = 'X'

        new_list = []
        updated_stabilizers = []
        for y in term_list:
            new_list += [_fix_single_term(y, fixed_positions[i], fixed_op,
                                          other_op, stabilizer_list[0])]
        for update_stab in stabilizer_list[1:]:
            updated_stabilizers += [_fix_single_term(update_stab,
                                                     fixed_positions[i],
                                                     fixed_op, other_op,
                                                     stabilizer_list[0])]
        term_list = new_list
        stabilizer_list = updated_stabilizers

        _check_stabilizer_linearity(stabilizer_list,
                                    msg='Linearly-dependent stabilizers.')
        _check_commuting_stabilizers(stabilizer_list,
                                     msg='Stabilizers anti-commute.')

    new_terms = QubitOperator()
    for x, ent in enumerate(term_list):
        new_terms += ent * terms.terms[term_list_duplicate[x]]
    for x, ent in enumerate(term_list):
        term_list_duplicate[x] = (
            QubitOperator(term_list_duplicate[x]) /
            list(ent.terms.items())[0][1])
        term_list[x] = list(ent.terms)[0]

    even_newer_terms = QubitOperator()
    for pauli_string, coefficient in new_terms.terms.items():
        even_newer_terms += coefficient * _lookup_term(
            pauli_string,
            term_list_duplicate,
            term_list)

    return even_newer_terms, fixed_positions


def reduce_number_of_terms(operator, stabilizers,
                           maintain_length=False,
                           output_fixed_positions=False,
                           manual_input=False,
                           fixed_positions=[]):
    """
    Reduce the number of Pauli strings of operator using stabilizers.

    The Pauli strings maintain their length, unless specified otherwise.
    In that case one can also pass a list with fixed indices manually.

    Args:
        operator (QubitOperator): operator from which the number of terms
                                  will be reduced.
        stabilizers (QubitOperator): stabilizer operators to be used to reduce
                                     terms of operator.
        maintain_length (Boolean): whether to keep the length of the terms
                                   equal to the starting operator or reduce it.
        output_fixed_positions (Boolean): whether to return the list of fixed
                                          qubits.
        manual_input (Boolean): whether to pass the fixed qubits manually.
        fixed_positions (list): (optional) positions of qubits to be removed,
                                only possible if manual_input is True.
    Returns:
        reduced_operator (QubitOperator): operator with reduced number of
                                          terms.

        fixed_positions (list): (optional) fixed qubits.

    Raises:
        TypeError: terms must be a QubitOperator
        TypeError: stabilizers must be a QubitOperator or array-like.
        TaperQubitError: fixed_positions and stabilizers must have the same
                         length if manual_input is True.
        TaperQubitError: fixed_positions contains the same position more
                         than once.
    """
    if not isinstance(operator, QubitOperator):
        raise TypeError('Input terms must be QubitOperator.')
    if not isinstance(stabilizers, (QubitOperator, list,
                                    tuple, numpy.ndarray)):
        raise TypeError('Input stabilizers must be QubitOperator or list.')

    stabilizer_list = list(stabilizers)

    _check_stabilizer_linearity(stabilizer_list,
                                msg='Identity is not a stabilizer.')
    _check_commuting_stabilizers(stabilizer_list,
                                 msg='No complex coefficient in stabilizers.')

    if manual_input:
        # Convert fixed_position into a list to allow any type of
        # array_like data structure.
        fixed_positions = list(fixed_positions)
        if len(fixed_positions) != len(stabilizer_list):
            raise TaperQubitError('The number of stabilizers must be equal '
                                  'to the  number of qubits manually fixed.'
                                  )
        if len(set(fixed_positions)) != len(stabilizer_list):
            raise TaperQubitError('All qubit positions must be different.')

    # should we deepcopy operator ?
    if maintain_length:
        (reduced_operator,
         fixed_positions) = _reduce_terms_keep_length(operator,
                                                      stabilizer_list,
                                                      maintain_length,
                                                      manual_input,
                                                      fixed_positions)
    else:
        (reduced_operator,
         fixed_positions) = _reduce_terms(operator,
                                          stabilizer_list,
                                          maintain_length,
                                          manual_input,
                                          fixed_positions)

    if output_fixed_positions:
        return reduced_operator, fixed_positions
    else:
        return reduced_operator


def taper_off_qubits(hamiltonian, stabilizers, manual_input=False,
                     fixed_positions=[]):
    """
    Remove qubits from the Hamiltonian.

    Qubits are removed by eliminating an eqivalent number of
    stabilizer conditions. Which qubits that are is either determined
    automatically or their positions be set manually.

    Args:
        operator (QubitOperator): operator from which the number of terms
                                  will be reduced.
        stabilizers (QubitOperator): stabilizer operators to be used to reduce
                                     terms of operator.
        manual_input (Boolean): whether to pass the fixed qubits manually.
        fixed_positions (list): (optional) positions of qubits to be removed,
                                only possible if manual_input is True.
    Returns:
        reduced_operator (QubitOperator): operator with reduced number of
                                          terms.

        fixed_positions (list): (optional) fixed qubits.

    Raises:
        TypeError: terms must be a QubitOperator
        TypeError: stabilizers must be a QubitOperator or array-like.
        ValueError: fixed_positions and stabilizers must have the same length
                    if manual_input is True.
        ValueError: fixed_positions contains the same position more than once.
        Warning: if number of qubits at the end equals number of qubits at the
                 start.

    """
    n_qbits = count_qubits(hamiltonian)
    (ham_to_update,
     qbts_to_rm) = reduce_number_of_terms(hamiltonian,
                                          stabilizers,
                                          maintain_length=False,
                                          manual_input=manual_input,
                                          fixed_positions=fixed_positions,
                                          output_fixed_positions=True)

    # Gets a list of the order of the qubits after tapering
    # putting the qubits to be removed at the end.
    qbit_order = list(numpy.arange(n_qbits - len(qbts_to_rm), dtype=int))
    qbts_to_rm.sort()
    for x in qbts_to_rm:
        qbit_order.insert(x, 'remove')

    # Remove the qubits
    tap_ham = QubitOperator()
    for term, coef in ham_to_update.terms.items():
        if term == ():
            tap_ham += QubitOperator('', coef)
            continue
        tap_tpls = []
        for p in term:
            if qbit_order[p[0]] != 'remove':
                tap_tpls.append((qbit_order[p[0]].item(), p[1]))

        tap_ham += QubitOperator(tuple(tap_tpls), coef)

    return tap_ham
