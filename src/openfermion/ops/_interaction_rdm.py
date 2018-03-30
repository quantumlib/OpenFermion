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

"""Class and functions to store reduced density matrices."""
from __future__ import absolute_import

import copy
import numpy

from openfermion.ops import (FermionOperator,
                             InteractionOperator,
                             PolynomialTensor,
                             QubitOperator,
                             normal_ordered)


class InteractionRDMError(Exception):
    pass


class InteractionRDM(PolynomialTensor):
    """Class for storing 1- and 2-body reduced density matrices.

    Attributes:
        one_body_tensor: The expectation values <a^\dagger_p a_q>.
        two_body_tensor: The expectation values
            <a^\dagger_p a^\dagger_q a_r a_s>.
    """

    def __init__(self, one_body_tensor, two_body_tensor):
        """Initialize the InteractionRDM class.

        Args:
            one_body_tensor: Expectation values <a^\dagger_p a_q>.
            two_body_tensor: Expectation values
                <a^\dagger_p a^\dagger_q a_r a_s>.
        """
        super(InteractionRDM, self).__init__(
            {(1, 0): one_body_tensor, (1, 1, 0, 0): two_body_tensor})
        self.one_body_tensor = self.n_body_tensors[1, 0]
        self.two_body_tensor = self.n_body_tensors[1, 1, 0, 0]

    def expectation(self, operator):
        """Return expectation value of an InteractionRDM with an operator.

        Args:
            operator: A QubitOperator or InteractionOperator.

        Returns:
            float: Expectation value

        Raises:
            InteractionRDMError: Invalid operator provided.
        """
        if isinstance(operator, QubitOperator):
            expectation_op = self.get_qubit_expectations(operator)
            expectation = 0.0
            for qubit_term in operator.terms:
                expectation += (operator.terms[qubit_term] *
                                expectation_op.terms[qubit_term])
        elif isinstance(operator, InteractionOperator):
            expectation = operator.constant
            expectation += numpy.sum(self.one_body_tensor *
                                     operator.one_body_tensor)
            expectation += numpy.sum(self.two_body_tensor *
                                     operator.two_body_tensor)
        else:
            raise InteractionRDMError('Invalid operator type provided.')
        return expectation

    def get_qubit_expectations(self, qubit_operator):
        """Return expectations of QubitOperator in new QubitOperator.

        Args:
            qubit_operator: QubitOperator instance to be evaluated on
                this InteractionRDM.

        Returns:
            QubitOperator: QubitOperator with coefficients
            corresponding to expectation values of those operators.

        Raises:
            InteractionRDMError: Observable not contained in 1-RDM or 2-RDM.
        """
        from openfermion.transforms import reverse_jordan_wigner
        qubit_operator_expectations = copy.deepcopy(qubit_operator)
        for qubit_term in qubit_operator_expectations.terms:
            expectation = 0.

            # Map qubits back to fermions.
            reversed_fermion_operators = reverse_jordan_wigner(
                QubitOperator(qubit_term))
            reversed_fermion_operators = normal_ordered(
                reversed_fermion_operators)

            # Loop through fermion terms.
            for fermion_term in reversed_fermion_operators.terms:
                coefficient = reversed_fermion_operators.terms[fermion_term]

                # Handle molecular term.
                if FermionOperator(fermion_term).is_molecular_term():
                    if not fermion_term:
                        expectation += coefficient
                    else:
                        indices = [operator[0] for operator in fermion_term]
                        if len(indices) == 2:
                            # One-body term
                            indices = tuple(zip(indices, (1, 0)))
                        else:
                            # Two-body term
                            indices = tuple(zip(indices, (1, 1, 0, 0)))
                        rdm_element = self[indices]
                        expectation += rdm_element * coefficient

                # Handle non-molecular terms.
                elif len(fermion_term) > 4:
                    raise InteractionRDMError('Observable not contained '
                                              'in 1-RDM or 2-RDM.')
            qubit_operator_expectations.terms[qubit_term] = expectation
        return qubit_operator_expectations
