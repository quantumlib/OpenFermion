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
"""Tools to reduce the number of terms and taper off qubits
using stabilizer conditions. Based on ideas of arXiv:1701.08213. """

import numpy
from openfermion.ops import QubitOperator
from openfermion.utils import count_qubits
from openfermion.config import EQ_TOLERANCE


class StabilizerError(Exception):
    """Stabilizer error class."""

    def __init__(self, message):
        """
        Throw custom errors connected to stabilizers.

        Args:
            message(str): custome error message string.
        """
        Exception.__init__(self, message)


def check_commuting_stabilizers(stabilizer_list, msg, thres=EQ_TOLERANCE):
    """
    Auxiliary function checking that stabilizers commute.

    If two stabilizers anti-commute their product
    will have an imaginary coefficient.
    This function checks the list of stabilizers (QubitOperator)
    and raises and error if a complex number is found in
    any of the coefficients.

    Args:
        stabilizer_list (list): List of stabilizers as QubitOperators.
        msg (str): Message for the error.
        thres: Tolerance value, set to OpenFermion tolerance by default.
    """
    for stab in stabilizer_list:
        if abs(numpy.imag(list(stab.terms.values())[0])) >= thres:
            raise StabilizerError(msg)


def check_stabilizer_linearity(stabilizer_list, msg):
    """
    Auxiliary function checking that stabilizers are linearly independent.

    If two stabilizer are linearly dependent the result
    after some of their products will be the identity.
    This function checks the list of stabilizers (QubitOperator)
    and raises an error if the identity is found.

    Args:
        stabilizer_list (list): List of stabilizers (QubitOperator).
        msg (str): Message for the error.
    """
    for stab in stabilizer_list:
        if list(stab.terms.keys())[0] == ():
            raise StabilizerError(msg)


def fix_single_term(term, position, fixed_op, other_op, stabilizer):
    """
    Auxiliary function for term reductions.

    Automatically multiplies a single term with a given stabilizer if
    the Pauli operator on a given qubit is of one of two specified types.
    This fixes a certain representation of a logical operator.

    Args:
        term (QubitOperator): Single term to fix.
        position (int): Index of the qubit which is to be fixed.
        fixed_op (str): Pauli operator, which
                        will cause a multiplication by the
                        stabilizer when encountered at the fixed
                        position.
        other_op (str): Alternative Pauli operator, which will also
                        cause the multiplication by the stabilizer.
        stabilizer (QubitOperator): Stabilizer that is multiplied
                                    when necessary.

    Returns:
        term (QubitOperator): Updated term in a fiixed representation.
    """
    pauli_tuple = list(term.terms)[0]
    if (position, fixed_op) in pauli_tuple or (position,
                                               other_op) in pauli_tuple:
        return term * stabilizer
    else:
        return term


def _lookup_term(pauli_string, updated_terms_1, updated_terms_2):
    """
    Auxiliary function for reducing terms keeping length.

    This function checks the length of the original Pauli strings,
    compares it to a list of strings and returns the shortest operator
    equivalent to the original.

    Args:
        pauli_string (tuple): Original Pauli string given in the same form
                              as in the data structure of QubitOperator.
        updated_terms_1 (list): List of Pauli strings (QubitOperator),
                                which replace the original string if
                                they are shorter and they are equivalent
                                to each other.
        updated_terms_2 (list): List of Pauli strings given in the data
                                structure of QubitOperator, denoting which
                                strings the entries of the first list are
                                equivalent to.
    Returns:
        pauli_op (QubitOperator): Shortest Pauli string equivalent to the
                                  original.

    """
    pauli_op = QubitOperator(pauli_string)
    length = len(pauli_string)

    for x in numpy.arange(len(updated_terms_1)):
        if (pauli_string == updated_terms_2[x] and
            (length > len(list(updated_terms_1[x].terms)[0]))):
            pauli_op = updated_terms_1[x]
            length = len(list(updated_terms_1[x].terms)[0])
    return pauli_op


def _reduce_terms(terms, stabilizer_list, manual_input, fixed_positions):
    """
    Perform the term reduction using stabilizer conditions.

    Auxiliary function to reduce_number_of_terms.

    Args:
        terms (QubitOperator): Operator the number of terms is to be reduced.
        stabilizer_list (list): List of the stabilizers as QubitOperators.
        manual_input (Boolean): Option to pass the list of fixed qubits
                                positions manually. Set to False by default.
        fixed_positions (list): (optional) List of fixed qubit positions.
                                Passing a list is only effective if
                                manual_input is True.
    Returns:
        even_newer_terms (QubitOperator): Updated operator with reduced terms.
        fixed_positions (list): Positions of qubits to be used for the
                                term reduction.
    Raises:
        StabilizerError: Trivial stabilizer (identity).
        StabilizerError: Stabilizer with complex coefficient.
    """
    # Initialize fixed_position as an empty list to avoid conflict with
    # fixed_positions.
    if manual_input is False:
        fixed_positions = []

    # We need the index of the stabilizer to connect it to the fixed qubit.
    for i, _ in enumerate(stabilizer_list):
        selected_stab = list(stabilizer_list[0].terms)[0]

        if manual_input is False:
            # Find first position non-fixed position with non-trivial Pauli.
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
            new_terms += fix_single_term(qubit_pauli, fixed_positions[i],
                                         fixed_op, other_op, stabilizer_list[0])
        updated_stabilizers = []
        for update_stab in stabilizer_list[1:]:
            updated_stabilizers += [
                fix_single_term(update_stab, fixed_positions[i], fixed_op,
                                other_op, stabilizer_list[0])
            ]

        # Update terms and stabilizer list.
        terms = new_terms
        stabilizer_list = updated_stabilizers

        check_stabilizer_linearity(stabilizer_list,
                                   msg='Linearly dependent stabilizers.')
        check_commuting_stabilizers(stabilizer_list,
                                    msg='Stabilizers anti-commute.')

    return terms, fixed_positions


def _reduce_terms_keep_length(terms, stabilizer_list, manual_input,
                              fixed_positions):
    """
    Perform the term reduction using stabilizer conditions.

    Auxiliary function to reduce_number_of_terms that returns the
    Pauli strings with the same length as in the starting operator.

    Args:
        terms (QubitOperator): Operator from which terms are reduced.
        stabilizer_list (list): List of the stabilizers as QubitOperators.
        manual_input (Boolean): Option to pass the list of fixed qubits
                                positions manually. Set to False by default.
        fixed_positions (list): (optional) List of fixed qubit positions.
                                Passing a list is only effective if
                                manual_input is True.
    Returns:
        even_newer_terms (QubitOperator): Updated operator with reduced terms.
        fixed_positions (list): Positions of qubits to be used for the
                                term reduction.
    Raises:
        StabilizerError: Trivial stabilizer (identity).
        StabilizerError: Stabilizer with complex coefficient.
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
            new_list += [
                fix_single_term(y, fixed_positions[i], fixed_op, other_op,
                                stabilizer_list[0])
            ]
        for update_stab in stabilizer_list[1:]:
            updated_stabilizers += [
                fix_single_term(update_stab, fixed_positions[i], fixed_op,
                                other_op, stabilizer_list[0])
            ]
        term_list = new_list
        stabilizer_list = updated_stabilizers

        check_stabilizer_linearity(stabilizer_list,
                                   msg='Linearly dependent stabilizers.')
        check_commuting_stabilizers(stabilizer_list,
                                    msg='Stabilizers anti-commute.')

    new_terms = QubitOperator()
    for x, ent in enumerate(term_list):
        new_terms += ent * terms.terms[term_list_duplicate[x]]
    for x, ent in enumerate(term_list):
        term_list_duplicate[x] = (QubitOperator(term_list_duplicate[x]) /
                                  list(ent.terms.items())[0][1])
        term_list[x] = list(ent.terms)[0]

    even_newer_terms = QubitOperator()
    for pauli_string, coefficient in new_terms.terms.items():
        even_newer_terms += coefficient * _lookup_term(
            pauli_string, term_list_duplicate, term_list)

    return even_newer_terms, fixed_positions


def reduce_number_of_terms(operator,
                           stabilizers,
                           maintain_length=False,
                           output_fixed_positions=False,
                           manual_input=False,
                           fixed_positions=None):
    r"""
    Reduce the number of Pauli strings of operator using stabilizers.

    This function reduces the number of terms in a string by merging
    terms that are identical by the multiplication of stabilizers.
    The resulting Pauli strings maintain their length, unless specified
    otherwise. In the latter case, a list of indices can be passed to
    manually indicate the qubits to be fixed.

    It is possible to reduce the number of terms in a Hamiltonian by
    merging Pauli strings :math:`H_1, \, H_2` that are related by a
    stabilizer :math:`S` such that  :math:`H_1 = H_2 \cdot S`. Given
    a stabilizer generator :math:`\pm X \otimes p` this algorithm fixes the
    first qubit, such that every Pauli string in the Hamiltonian acts with
    either :math:`Z` or the identity on it. Where necessary, this is achieved
    by multiplications with :math:`\pm X \otimes p`: a string
    :math:`Y \otimes h`, for instance,  is turned into
    :math:`Z \otimes (\mp ih\cdot p)`. Qubits on which a generator acts as
    :math:`Y` (:math:`Z`) are constrained to be acted on by the Hamiltonian as
    :math:`Z` (:math:`X`) or the identity. Fixing a different qubit for every
    stabilizer generator eliminates all redundant strings. The fixed
    representations are in the end re-expressed as the shortest of the
    original strings, :math:`H_1` or :math:`H_2`.


    Args:
        operator (QubitOperator): Operator of which the number of terms
                                  will be reduced.
        stabilizers (QubitOperator): Stabilizer generators used for the
                                     reduction. Can also be passed as a list
                                     of QubitOperator.
        maintain_length (Boolean): Option deciding whether the fixed Pauli
                                   strings are re-expressed in their original
                                   form. Set to False by default.
        output_fixed_positions (Boolean): Option deciding whether to return
                                          the list of fixed qubit positions.
                                          Set to False by default.
        manual_input (Boolean): Option to pass the list of fixed qubits
                                positions manually. Set to False by default.
        fixed_positions (list): (optional) List of fixed qubit positions.
                                Passing a list is only effective if
                                manual_input is True.
    Returns:
        reduced_operator (QubitOperator): Operator with reduced number of
                                          terms.

        fixed_positions (list): (optional) Fixed qubits.

    Raises:
        TypeError: Input terms must be QubitOperator.
        TypeError: Input stabilizers must be QubitOperator or list.
        StabilizerError: Trivial stabilizer (identity).
        StabilizerError: Stabilizer with complex coefficient.
        TypeError: List of fixed qubits required if manual input True.
        StabilizerError: The number of stabilizers must be equal to the number
                         of qubits manually fixed.
        StabilizerError: All qubit positions must be different.
    """
    if not isinstance(operator, QubitOperator):
        raise TypeError('Input terms must be QubitOperator.')
    if not isinstance(stabilizers, (QubitOperator, list, tuple, numpy.ndarray)):
        raise TypeError('Input stabilizers must be QubitOperator or list.')

    stabilizer_list = list(stabilizers)

    check_stabilizer_linearity(stabilizer_list,
                               msg='Trivial stabilizer (identity).')
    check_commuting_stabilizers(stabilizer_list,
                                msg='Stabilizer with complex coefficient.')

    if manual_input:
        # Convert fixed_position into a list to allow any type of
        # array_like data structure.
        if fixed_positions is None:
            raise TypeError('List of qubit positions required.')
        fixed_positions = list(fixed_positions)
        if len(fixed_positions) != len(stabilizer_list):
            raise StabilizerError('The number of stabilizers must be equal ' +
                                  'to the number of qubits manually fixed.')
        if len(set(fixed_positions)) != len(stabilizer_list):
            raise StabilizerError('All qubit positions must be different.')

    if maintain_length:
        (reduced_operator,
         fixed_positions) = _reduce_terms_keep_length(operator, stabilizer_list,
                                                      manual_input,
                                                      fixed_positions)
    else:
        (reduced_operator,
         fixed_positions) = _reduce_terms(operator, stabilizer_list,
                                          manual_input, fixed_positions)

    if output_fixed_positions:
        return reduced_operator, fixed_positions
    else:
        return reduced_operator


def taper_off_qubits(operator,
                     stabilizers,
                     manual_input=False,
                     fixed_positions=None,
                     output_tapered_positions=False):
    r"""
    Remove qubits from given operator.

    Qubits are removed by eliminating an equivalent number of
    stabilizer conditions. Which qubits that are can either be determined
    automatically or their positions can be set manually.

    Qubits can be disregarded from the Hamiltonian when the effect of all its
    terms on them is rendered trivial. This algorithm employs a stabilizers
    like :math:`\pm X \otimes p` to fix the action of every Pauli
    string on the first qubit to :math:`Z` or the identity. A string
    :math:`X \otimes h` would for instance be multiplied with the stabilizer
    to obtain :math:`1 \otimes (\pm h\cdot p)` while a string
    :math:`Z \otimes h^\prime` would pass without correction. The first
    qubit can subsequently be removed as it must be in the computational basis
    in Hamiltonian eigenstates.
    For stabilizers acting as :math:`Y` (:math:`Z`) on selected qubits,
    the algorithm would fix the action of every Hamiltonian string to
    :math:`Z` (:math:`X`). Updating also the list of remaining stabilizer
    generators, the algorithm is run iteratively.

    Args:
        operator (QubitOperator): Operator of which qubits will be removed.
        stabilizers (QubitOperator): Stabilizer generators for the tapering.
                                     Can also be passed as a list of
                                     QubitOperator.
        manual_input (Boolean): Option to pass the list of fixed qubits
                                positions manually. Set to False by default.
        fixed_positions (list): (optional) List of fixed qubit positions.
                                Passing a list is only effective if
                                manual_input is True.
        output_tapered_positions (Boolean): Option to output the positions of
                                            qubits that have been removed.
    Returns:
        skimmed_operator (QubitOperator): Operator with fewer qubits.
        removed_positions (list): (optional) List of removed qubit positions.
                                  For the qubits to be gone in the qubit count,
                                  the remaining qubits have been moved up to
                                  those indices.
    """
    if isinstance(stabilizers, (list, tuple, numpy.ndarray)):
        n_qbits_stabs = 0
        for ent in stabilizers:
            if count_qubits(ent) > n_qbits_stabs:
                n_qbits_stabs = count_qubits(ent)
    else:
        n_qbits_stabs = count_qubits(stabilizers)

    n_qbits = max(count_qubits(operator), n_qbits_stabs)

    (ham_to_update,
     qbts_to_rm) = reduce_number_of_terms(operator,
                                          stabilizers,
                                          maintain_length=False,
                                          manual_input=manual_input,
                                          fixed_positions=fixed_positions,
                                          output_fixed_positions=True)

    # Gets a list of the order of the qubits after tapering
    qbit_order = list(numpy.arange(n_qbits - len(qbts_to_rm), dtype=int))
    # Save the original list before it gets ordered
    removed_positions = qbts_to_rm
    qbts_to_rm.sort()
    for x in qbts_to_rm:
        qbit_order.insert(x, 'remove')

    # Remove the qubits
    skimmed_operator = QubitOperator()
    for term, coef in ham_to_update.terms.items():
        if term == ():
            skimmed_operator += QubitOperator('', coef)
            continue
        tap_tpls = []
        for p in term:
            if qbit_order[p[0]] != 'remove':
                tap_tpls.append((qbit_order[p[0]].item(), p[1]))

        skimmed_operator += QubitOperator(tuple(tap_tpls), coef)

    if output_tapered_positions:
        return skimmed_operator, removed_positions
    else:
        return skimmed_operator
