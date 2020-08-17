import itertools
import numpy

from openfermion.ops.operators import FermionOperator
from openfermion.ops.representations import InteractionRDM

from openfermion.utils.operator_utils import count_qubits
from openfermion.transforms.opconversions.conversions import check_no_sympy


def get_interaction_rdm(qubit_operator, n_qubits=None):
    """Build an InteractionRDM from measured qubit operators.

    Returns: An InteractionRDM object.
    """

    check_no_sympy(qubit_operator)

    # Avoid circular import.
    from openfermion.transforms import jordan_wigner
    if n_qubits is None:
        n_qubits = count_qubits(qubit_operator)
    one_rdm = numpy.zeros((n_qubits,) * 2, dtype=complex)
    two_rdm = numpy.zeros((n_qubits,) * 4, dtype=complex)

    # One-RDM.
    for i, j in itertools.product(range(n_qubits), repeat=2):
        transformed_operator = jordan_wigner(FermionOperator(((i, 1), (j, 0))))
        for term, coefficient in transformed_operator.terms.items():
            if term in qubit_operator.terms:
                one_rdm[i, j] += coefficient * qubit_operator.terms[term]

    # Two-RDM.
    for i, j, k, l in itertools.product(range(n_qubits), repeat=4):
        transformed_operator = jordan_wigner(
            FermionOperator(((i, 1), (j, 1), (k, 0), (l, 0))))
        for term, coefficient in transformed_operator.terms.items():
            if term in qubit_operator.terms:
                two_rdm[i, j, k, l] += coefficient * qubit_operator.terms[term]

    return InteractionRDM(one_rdm, two_rdm)
