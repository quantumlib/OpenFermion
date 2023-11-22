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
"""Class and functions to store interaction operators."""
import numpy

from openfermion.ops import QubitOperator
from openfermion.ops.representations import PolynomialTensor, get_tensors_from_integrals

COEFFICIENT_TYPES = (int, float, complex)


class DOCIHamiltonian(PolynomialTensor):
    r"""Class for storing DOCI hamiltonians which are defined to be
    restrictions of fermionic operators to doubly occupied configurations.
    As such they are by nature hard-core boson Hamiltonians, but the
    hard-core Boson algebra is identical to the Pauli algebra, which is
    why it is convenient to represent DOCI hamiltonians as QubitOperators

    Note that the operators stored in this class take the form:

        $$
            constant + \sum_{p} h^{(r1)}_{p, p}/2 (1 - \sigma^Z_p) +
            \sum_{p \neq q} h^{(r1)}_{p, q}/4
                * (\sigma^X_p \sigma^X_q + \sigma^Y_p \sigma^Y_q) +
            \sum_{p \neq q} h^{(r2)}_{p, q}/4 (1 - \sigma^Z_p -
            \sigma^Z_q + \sigma^Z_p \sigma^Z_q)
            =
            constant + \sum_{p} h_{p, p} N_p +
            \sum_{p \neq q} w_{p, p} N_p N_q +
            \sum_{p \neq q} v_{p, p} P_p^\dagger P_q,
        $$

    where

        $$
            N_p = (1 - \sigma^Z_p)/2,
            P_p = a_{i,\beta} a_{i,\alpha},
            h_p = h^{(r1)}_{p, p} = \langle p|h|p \rangle =
                2 I^{(1)}_{p, p} + I^{(2)}_{p, p, p, p},
            w_{p, q} = h^{(r2)}_{p, q} = 2 \langle pq|v|pq \rangle -
                                         \langle pq|v|qp \rangle =
                                        2 I^{(2)}_{p, q, q, p} -
                                        I^{(2)}_{p, q, p, q},
            v_{p, q} = h^{(r1)}_{p, q} = \langle pp|v|qq \rangle =
                I^{(2)}_{p, p, q, q},
        $$

    with ($I^{(1)}_{p, q}$) and ($I^{(2)}_{p, q, r, s}$) are the one
    and two body electron integrals and ($h$) and ($v$) are the
    coefficients of the corresponding InteractionOperator

        $$
            constant + \sum_{p, q} h_{p, q} a^\dagger_p a_q +
            \sum_{p, q, r, s} h_{p, q, r, s} a^\dagger_p a^\dagger_q a_r a_s.
        $$


    Attributes:
        constant: The constant offset.
        hr1: The coefficients of ($h^{r1}_{p, q}$).
            This is an n_qubits x n_qubits numpy array of floats.
        hr2: The coefficients of ($h^{r2}_{p, q}$).
            This is an n_qubits x n_qubits numpy array of floats.
    """

    def __init__(self, constant, hc, hr1, hr2):
        r"""
        Initialize the DOCIHamiltonian class.

        Args:
            constant: A constant term in the operator given as a
                float. For instance, the nuclear repulsion energy.
            hc: the coefficients of ($h^{(c)}_{p}$)
            hr1: The coefficients of ($`h^{(r1)}_{p, q}$).
               This is an n_qubits x n_qubits numpy array of floats.
            hr2: The coefficients of ($h^{(r2)}_{p, q}$).
                This is an n_qubits x n_qubits array of floats.
        """
        super(DOCIHamiltonian, self).__init__(None)

        self._n_qubits = hc.shape[0]
        self._constant = constant
        self._hr1 = hr1
        self._hr2 = hr2
        self._hc = hc

    @property
    def qubit_operator(self):
        """Return the QubitOperator representation of this DOCI Hamiltonian"""
        return self.identity_part + self.z_part + self.xy_part

    def xx_term(self, p, q):
        """Returns the XX term on a single pair of qubits as a QubitOperator
        Arguments:
            p, q [int] -- qubit indices
        Returns:
            [QubitOperator] -- XX term on the chosen qubits.
        """
        return QubitOperator("X" + str(p) + " X" + str(q), self.hr1[p, q] / 2)

    def yy_term(self, p, q):
        """Returns the YY term on a single pair of qubits as a QubitOperator
        Arguments:
            p, q [int] -- qubit indices
        Returns:
            [QubitOperator] -- YY term on the chosen qubits.
        """
        return QubitOperator("Y" + str(p) + " Y" + str(q), self.hr1[p, q] / 2)

    def z_term(self, p):
        """Returns the Z term on a single qubit as a QubitOperator
        Arguments:
            p [int] -- qubit index
        Returns:
            [QubitOperator] -- Z term on the chosen qubit.
        """
        return QubitOperator("Z" + str(p), -self.hc[p] / 2 - sum(self.hr2[:, p]) / 2)

    def zz_term(self, p, q):
        """Returns the ZZ term on a single pair of qubits as a QubitOperator
        Arguments:
            p, q [int] -- qubit indices
        Returns:
            [QubitOperator] -- ZZ term on the chosen qubits.
        """
        return QubitOperator("Z" + str(p) + " Z" + str(q), self.hr2[p, q] / 2)

    @property
    def identity_part(self):
        """Returns identity term of this operator (i.e. trace-ful term)
        in QubitOperator form.
        """
        return QubitOperator(
            (),
            self.constant
            + numpy.sum(self.hc) / 2
            + numpy.sum(self.hr2) / 4
            + numpy.sum(numpy.diag(self.hr2)) / 4,
        )

    @property
    def xx_part(self):
        """Returns the XX part of the QubitOperator representation of this
        DOCIHamiltonian
        """
        return sum(
            [self.xx_term(p, q) for p in range(self.n_qubits) for q in range(p + 1, self.n_qubits)]
        )

    @property
    def yy_part(self):
        """Returns the YY part of the QubitOperator representation of this
        DOCIHamiltonian
        """
        return sum(
            [self.yy_term(p, q) for p in range(self.n_qubits) for q in range(p + 1, self.n_qubits)]
        )

    @property
    def xy_part(self):
        """Returns the XX+YY part of the QubitOperator representation of this
        DOCIHamiltonian
        """
        return self.xx_part + self.yy_part

    @property
    def zz_part(self):
        """Returns the ZZ part of the QubitOperator representation of this
        DOCIHamiltonian
        """
        return sum(
            [self.zz_term(p, q) for p in range(self.n_qubits) for q in range(p + 1, self.n_qubits)]
        )

    @property
    def z_part(self):
        """Return the Z and ZZ part of the QubitOperator representation of this
        DOCI Hamiltonian"""
        return self.zz_part + sum([self.z_term(p) for p in range(self.n_qubits)])

    @property
    def hc(self):
        return self._hc

    @hc.setter
    def hc(self, value):
        self._hc = value

    @property
    def hr1(self):
        """The value of hr1."""
        return self._hr1

    @hr1.setter
    def hr1(self, value):
        self._hr1 = value

    @property
    def hr2(self):
        """The value of hr2."""
        return self._hr2

    @hr2.setter
    def hr2(self, value):
        self._hr2 = value

    # Override base class
    @property
    def constant(self):
        return self._constant

    @constant.setter
    def constant(self, value):
        self._constant = value

    # Override base class to generate on the fly
    @property
    def n_body_tensors(self):
        one_body_coefficients, two_body_coefficients = get_tensors_from_doci(
            self.hc, self.hr1, self.hr2
        )
        n_body_tensors = {
            (): self.constant,
            (1, 0): one_body_coefficients,
            (1, 1, 0, 0): two_body_coefficients,
        }
        return n_body_tensors

    @n_body_tensors.setter
    def n_body_tensors(self, value):
        raise TypeError(
            'Raw edits of the n_body_tensors of a DOCIHamiltonian '
            'is not allowed. Either adjust the hc/hr1/hr2 terms '
            'or cast to another PolynomialTensor class.'
        )

    def _get_onebody_term(self, key, index):
        if key[0] != 1 or key[1] != 0:
            raise IndexError(
                'DOCIHamiltonian class only contains ' 'one-body terms in the (1, 0) sector.'
            )
        if index[0] != index[1]:
            raise IndexError(
                'DOCIHamiltonian class only contains ' 'diagonal one-body electron integrals.'
            )
        return self.hc[index[0] // 2] / 2

    def _get_twobody_term(self, key, index):
        if key[0] != 1 or key[1] != 1 or key[2] != 0 or key[3] != 0:
            raise IndexError(
                'DOCIHamiltonian class only contains ' 'two-body terms in the (1, 1, 0, 0) sector.'
            )
        if index[0] == index[3] and index[1] == index[2]:
            return self.hr2[index[0] // 2, index[1] // 2] / 2
        if index[0] // 2 == index[1] // 2 and index[2] // 2 == index[3] // 2:
            return self.hr1[index[0] // 2, index[2] // 2] / 2
        raise IndexError(
            'DOCIHamiltonian class only contains '
            'two-electron integrals corresponding '
            'to a double excitation.'
        )

    # Override base class
    def __getitem__(self, args):
        """Look up matrix element.

        Args:
            args: Tuples indicating which coefficient to get. For instance,
                `my_tensor[(6, 1), (8, 1), (2, 0)]`
                returns
                `my_tensor.n_body_tensors[1, 1, 0][6, 8, 2]`
        """
        if len(args) == 0:
            return self.constant
        index = tuple([operator[0] for operator in args])
        key = tuple([operator[1] for operator in args])
        if len(index) == 2:
            return self._get_onebody_term(key, index)
        if len(index) == 4:
            return self._get_twobody_term(key, index)
        raise IndexError(
            'DOCIHamiltonian class only contains ' 'one and two-electron and constant terms.'
        )

    # Override root class
    def __setitem__(self, args, value):
        # This is not well-defined for a DOCIHamiltonian as we want to keep
        # certain tensor elements within the class the same. Better to
        # make the user update the hr1/hr2/hc terms --- if they really
        # want to play around with the n_body_tensors here they should
        # be casting this to a raw PolynomialTensor.
        raise TypeError(
            'Raw edits of the n_body_tensors of a DOCIHamiltonian '
            'is not allowed. Either adjust the hc/hr1/hr2 terms '
            'or cast to another PolynomialTensor class.'
        )

    # Override root class
    def __iadd__(self, addend):
        if isinstance(addend, COEFFICIENT_TYPES):
            self.constant += addend
            return self
        if not issubclass(type(addend), DOCIHamiltonian):
            raise TypeError('Invalid type.')
        if self.n_qubits != addend.n_qubits:
            raise TypeError('Invalid tensor shape.')
        self.hc += addend.hc
        self.hr1 += addend.hr1
        self.hr2 += addend.hr2
        self.constant += addend.constant
        return self

    def __isub__(self, subtrahend):
        if isinstance(subtrahend, COEFFICIENT_TYPES):
            self.constant -= subtrahend
            return self
        if not issubclass(type(subtrahend), DOCIHamiltonian):
            raise TypeError('Invalid type.')
        if self.n_qubits != subtrahend.n_qubits:
            raise TypeError('Invalid tensor shape.')
        self.hc -= subtrahend.hc
        self.hr1 -= subtrahend.hr1
        self.hr2 -= subtrahend.hr2
        self.constant -= subtrahend.constant
        return self

    def __imul__(self, multiplier):
        if not isinstance(multiplier, COEFFICIENT_TYPES):
            raise TypeError('Invalid type.')
        self.hc *= multiplier
        self.hr1 *= multiplier
        self.hr2 *= multiplier
        self.constant *= multiplier
        return self

    def __itruediv__(self, dividend):
        if not isinstance(dividend, COEFFICIENT_TYPES):
            raise TypeError('Invalid type.')
        self.hc /= dividend
        self.hr1 /= dividend
        self.hr2 /= dividend
        self.constant /= dividend
        return self

    @classmethod
    def from_integrals(cls, constant, one_body_integrals, two_body_integrals):
        hc, hr1, hr2 = get_doci_from_integrals(one_body_integrals, two_body_integrals)
        return cls(constant, hc, hr1, hr2)

    @classmethod
    def zero(cls, n_qubits):
        return cls(
            0,
            numpy.zeros((n_qubits,), dtype=numpy.complex128),
            numpy.zeros((n_qubits,) * 2, dtype=numpy.complex128),
            numpy.zeros((n_qubits,) * 2, dtype=numpy.complex128),
        )

    def get_projected_integrals(self):
        '''Creates the one and two body integrals that would correspond to a
        hypothetic electronic structure Hamiltonian, which would satisfy the
        given set of hc, hr1 and hr2.

        This is technically not well-defined, as hr2 is
        not generated in a one-to-one fashion. This implies that calling

        get_doci_from_integrals(
            *get_projected_integrals_from_doci(
               hc, hr1, hr2
            )
        )

        should return the same hc, hr1, and hr2, but there is no such guarantee
        for

        get_projected_integrals_from_doci(
           *get_doci_from_integrals(
              one_body_integrals, two_body_integrals
           )
        )

        but this method attempts to create integrals that conform to the
        same symmetries as a physical electronic structure Hamiltonian would,
        with inevitable loss of information due to the ambiguity above.

        Args:
           hc [numpy array]: The single-particle DOCI terms in matrix form
           hr1 [numpy array]: The off-diagonal DOCI Hamiltonian terms in matrix
               form
           hr2 [numpy array]: The diagonal DOCI Hamiltonian terms in matrix form

        Returns:
           projected_onebody_integrals [numpy array]: The corresponding one-body
               integrals for the electronic structure Hamiltonian
           projected_twobody_integrals [numpy array]: The corresponding two body
               integrals for the electronic structure Hamiltonian
        '''
        one_body_integrals, two_body_integrals = get_projected_integrals_from_doci(
            self.hc, self.hr1, self.hr2
        )
        return one_body_integrals, two_body_integrals


def get_tensors_from_doci(hc, hr1, hr2):
    '''Makes the one and two-body tensors of a fermionic "parent Hamoltonian" \
    from the DOCI Hamiltonian matrices

    Args:
        hc [numpy array]: The single-particle DOCI terms in matrix form
        hr1 [numpy array]: The off-diagonal DOCI Hamiltonian terms in matrix
            form
        hr2 [numpy array]: The diagonal DOCI Hamiltonian terms in matrix form

    Returns:
        one_body_coefficients [numpy array]: The corresponding one-body
            tensor for the electronic structure Hamiltonian
        two_body_coefficients [numpy array]: The corresponding two body
            tensor for the electronic structure Hamiltonian
    '''
    one_body_integrals, two_body_integrals = get_projected_integrals_from_doci(hc, hr1, hr2)
    one_body_coefficients, two_body_coefficients = get_tensors_from_integrals(
        one_body_integrals, two_body_integrals
    )

    two_body_coefficients = two_body_coefficients - numpy.einsum('ijlk', two_body_coefficients)

    return one_body_coefficients, two_body_coefficients


def get_projected_integrals_from_doci(hc, hr1, hr2):
    '''Generates a set of atomic integrals corresponding to this Hamiltonian

    Makes the one and two-body integrals from the DOCI projectionof the hr1,
    hr2, and hc matrices. This is technically not well-defined, as hr2 is
    not generated in a one-to-one fashion. In particular, here we assume
    hr2 corresponds entirely to a pqqp term, while this could be equally well
    added to the pqpq term. This implies that calling

    get_doci_from_integrals(*get_projected_integrals_from_doci(hc, hr1, hr2))

    should return the same hc, hr1, and hr2, but there is no such guarantee
    for

    get_projected_integrals_from_doci(*get_doci_from_integrals(
        one_body_integrals, two_body_integrals))

    Args:
        hc [numpy array]: The single-particle DOCI terms in matrix form
        hr1 [numpy array]: The off-diagonal DOCI Hamiltonian terms in matrix
            form
        hr2 [numpy array]: The diagonal DOCI Hamiltonian terms in matrix form

    Returns:
        projected_onebody_integrals [numpy array]: The corresponding one-body
            integrals for the electronic structure Hamiltonian
        projected_twobody_integrals [numpy array]: The corresponding two body
            integrals for the electronic structure Hamiltonian
    '''
    n_qubits = hr1.shape[0]
    projected_onebody_integrals = numpy.zeros((n_qubits, n_qubits), dtype=hc.dtype)
    projected_twobody_integrals = numpy.zeros(
        (n_qubits, n_qubits, n_qubits, n_qubits), dtype=hc.dtype
    )
    for p in range(n_qubits):
        projected_onebody_integrals[p, p] = hc[p] / 2
        projected_twobody_integrals[p, p, p, p] = hr2[p, p]
        for q in range(n_qubits):
            if p <= q:
                continue

            projected_twobody_integrals[p, q, q, p] = hr2[p, q] / 2 + hr1[p, q] / 2
            projected_twobody_integrals[q, p, p, q] = hr2[q, p] / 2 + hr1[p, q] / 2

            projected_twobody_integrals[p, p, q, q] += hr1[p, q]
            projected_twobody_integrals[p, q, p, q] += hr1[p, q]
            projected_twobody_integrals[q, q, p, p] += hr1[p, q]
            projected_twobody_integrals[q, p, q, p] += hr1[p, q]

    return projected_onebody_integrals, projected_twobody_integrals


def get_doci_from_integrals(one_body_integrals, two_body_integrals):
    r"""Construct a DOCI Hamiltonian from electron integrals

    Args:
        one_body_integrals [numpy array]: one-electron integrals
        two_body_integrals [numpy array]: two-electron integrals

    Returns:
        hc [numpy array]: The single-particle DOCI terms in matrix form
        hr1 [numpy array]: The off-diagonal DOCI Hamiltonian terms in matrix
            form
        hr2 [numpy array]: The diagonal DOCI Hamiltonian terms in matrix form
    """

    n_qubits = one_body_integrals.shape[0]
    hc = numpy.zeros(n_qubits)
    hr1 = numpy.zeros((n_qubits, n_qubits))
    hr2 = numpy.zeros((n_qubits, n_qubits))

    for p in range(n_qubits):
        hc[p] = 2 * one_body_integrals[p, p]
        for q in range(n_qubits):
            hr2[p, q] = 2 * two_body_integrals[p, q, q, p] - two_body_integrals[p, q, p, q]
            if p == q:
                continue
            hr1[p, q] = two_body_integrals[p, p, q, q]

    return hc, hr1, hr2
