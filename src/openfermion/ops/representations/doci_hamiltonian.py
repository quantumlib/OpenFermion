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
from openfermion.ops.representations import (PolynomialTensor,
                                             get_tensor_from_integrals)


class _HR1(object):
    """class for storing the DOCI hr1 tensor alongside an nbody tensor rep
    """
    def __init__(self, hr1, n_body_tensors):
        self._hr1 = hr1
        self._n_body_tensors = n_body_tensors

    @property
    def shape(self):
        return self._hr1.shape

    def __getitem__(self, args):
        """Look up matrix element.

        Args:
            args: i, j
        """
        if len(args) != 2:
            raise IndexError('hr1 is a two-indexed array')
        p, q = args
        return self._hr1[p, q]

    def __setitem__(self, args, value):
        """Set matrix element.

        Args:
            args: Tuples indicating which coefficient to set.
        """
        if len(args) != 2:
            raise IndexError('hr1 is a two-indexed array')
        p, q = args
        if p == q:
            raise IndexError('hr1 has no diagonal term')
        self._hr1[p, q] = value
        two_body_coefficients = self._n_body_tensors[(1, 1, 0, 0)]

        # Mixed spin
        two_body_coefficients[2 * p, 2 * p + 1, 2 * q + 1, 2 * q] = (value / 2)
        two_body_coefficients[2 * p + 1, 2 * p, 2 * q, 2 * q + 1] = (value / 2)


class _HR2(object):
    """class for storing the DOCI hr2 tensor alongside an nbody tensor rep
    """
    def __init__(self, hr2, n_body_tensors):
        self._hr2 = hr2
        self._n_body_tensors = n_body_tensors

    @property
    def shape(self):
        return self._hr2.shape

    def __getitem__(self, args):
        """Look up matrix element.

        Args:
            args: i, j
        """
        if len(args) != 2:
            raise IndexError('hr2 is a two-indexed array')
        return self._hr2[args[0], args[1]]

    def __setitem__(self, args, value):
        """Set matrix element.

        Args:
            args: Tuples indicating which coefficient to set.
        """
        if len(args) != 2:
            raise IndexError('hr2 is a two-indexed array')
        p, q = args
        self._hr2[p, q] = value
        two_body_coefficients = self._n_body_tensors[(1, 1, 0, 0)]

        # Mixed spin
        two_body_coefficients[2 * p, 2 * q + 1, 2 * q + 1, 2 * p] = (value / 2)
        two_body_coefficients[2 * p + 1, 2 * q, 2 * q, 2 * p + 1] = (value / 2)

        # Same spin
        two_body_coefficients[2 * p, 2 * q, 2 * q, 2 * p] = (value / 2)
        two_body_coefficients[2 * p + 1, 2 * q + 1, 2 * q + 1, 2 * p + 1] = (
            value / 2)


class _HC(object):
    """class for storing the DOCI hr2 tensor alongside an nbody tensor rep
    """
    def __init__(self, hc, n_body_tensors):
        self._hc = hc
        self._n_body_tensors = n_body_tensors

    @property
    def shape(self):
        return self._hc.shape

    def __getitem__(self, args):
        """Look up matrix element.

        Args:
            args: i, j
        """
        if type(args) is tuple:
            if len(args) != 1:
                raise IndexError('hc is a one-indexed array')
            p, = args
        else:
            p = args
        return self._hc[p]

    def __setitem__(self, args, value):
        """Set matrix element.

        Args:
            args: Tuples indicating which coefficient to set.
        """
        if type(args) is tuple:
            if len(args) != 1:
                raise IndexError('hc is a one-indexed array')
            p, = args
        else:
            p = args
        self._hc[p] = value
        one_body_coefficients = self._n_body_tensors[(1, 0)]
        one_body_coefficients[2 * p, 2 * p] = value
        one_body_coefficients[2 * p + 1, 2 * p + 1] = value


class DOCIHamiltonian(PolynomialTensor):
    r"""Class for storing DOCI hamiltonians which are defined to be
    restrictions of fermionic operators to doubly occupied configurations.
    As such they are by nature hard-core boson Hamiltonians, but the
    hard-core Boson algebra is identical to the Pauli algebra, which is
    why it is convenient to represent DOCI hamiltonians as QubitOperators

    Note that the operators stored in this class take the form:

        .. math::

            constant + \sum_{p} h^{(r1)}_{p, p}/2 (1 - \sigma^Z_p) +
            \sum_{p \neq q} h^{(r1)}_{p, q}/4 (\sigma^X_p \sigma^X_q + \sigma^Y_p \sigma^Y_q) +
            \sum_{p \neq q} h^{(r2)}_{p, q}/4 (1 - \sigma^Z_p -
            \sigma^Z_q + \sigma^Z_p \sigma^Z_q)
            =
            constant + \sum_{p} h_{p, p} N_p +
            \sum_{p \neq q} w_{p, p} N_p N_q +
            \sum_{p \neq q} v_{p, p} P_p^\dagger P_q,

    where

        .. math::

           N_p = (1 - \sigma^Z_p)/2,
           P_p = a_{i,\beta} a_{i,\alpha},
           h_p = h^{(r1)}_{p, p} = \langle p|h|p \rangle = 2 I^{(1)}_{p, p} + I^{(2)}_{p, p, p, p},
           w_{p, q} = h^{(r2)}_{p, q} = 2 \langle pq|v|pq \rangle - \langle pq|v|qp \rangle =
                                        2 I^{(2)}_{p, q, q, p} - I^{(2)}_{p, q, p, q},
           v_{p, q} = h^{(r1)}_{p, q} = \langle pp|v|qq \rangle = I^{(2)}_{p, p, q, q},

    with (:math:`I^{(1)}_{p, q}`) and (:math:`I^{(2)}_{p, q, r, s}`) are the one and two body
    electron integrals and (:math:`h`) and (:math:`v`) are the coefficients of the
    corresponding InteractionOperator

        .. math::

            constant + \sum_{p, q} h_{p, q} a^\dagger_p a_q +
            \sum_{p, q, r, s} h_{p, q, r, s} a^\dagger_p a^\dagger_q a_r a_s.


    Attributes:
        constant: The constant offset.
        hr1: The coefficients of (:math:`h^{r1}_{p, q}`).
            This is an n_qubits x n_qubits numpy array of floats.
        hr2: The coefficients of (:math:`h^{r2}_{p, q}`).
            This is an n_qubits x n_qubits numpy array of floats.
    """

    def __init__(self, constant, hc, hr1, hr2):
        r"""
        Initialize the DOCIHamiltonian class.

        Args:
            constant: A constant term in the operator given as a
                float. For instance, the nuclear repulsion energy.
            hr1: The coefficients of (:math:`h^{(r1)}_{p, q}`).
               This is an n_qubits x n_qubits numpy array of floats.
            hr2: The coefficients of (:math:`h^{(r2)}_{p, q}`).
                This is an n_qubits x n_qubits array of floats.
        """
        one_body_coefficients, two_body_coefficients =\
            get_tensors_from_doci(hc, hr1, hr2)
        n_body_tensors = {
            (): constant,
            (1, 0): one_body_coefficients,
            (1, 1, 0, 0): two_body_coefficients
        }
        super(DOCIHamiltonian, self).__init__(n_body_tensors)
        self._hr1 = _HR1(hr1, n_body_tensors)
        self._hr2 = _HR2(hr2, n_body_tensors)
        self._hc = _HC(hc, n_body_tensors)

    @property
    def qubit_operator(self):
        """Return the QubitOperator representation of this DOCI Hamiltonian"""
        return QubitOperator((), self.constant) + self.z_part + self.xy_part

    @property
    def xy_part(self):
        """Return the XX and YY part of the QubitOperator representation of this
        DOCI Hamiltonian"""
        qubitop = QubitOperator()
        n_qubits = self.hr1.shape[0]
        for p in range(n_qubits):
            for q in range(n_qubits):
                if p == q:
                    continue
                qubitop +=  (QubitOperator("X"+str(p)+" X"+str(q),
                                           self.hr1[p, q]/4) +
                             QubitOperator("Y"+str(p)+" Y"+str(q),
                                           self.hr1[p, q]/4))

        return qubitop

    @property
    def z_part(self):
        """Return the Z and ZZ part of the QubitOperator representation of this
        DOCI Hamiltonian"""
        qubitop = QubitOperator()
        n_qubits = self.hr1.shape[0]
        for p in range(n_qubits):
            qubitop += (QubitOperator((), self.hr1[p, p] / 2) -
                        QubitOperator("Z"+str(p), self.hr1[p, p] / 2))
            for q in range(n_qubits):
                if p == q:
                    continue
                coef = self.hr2[p, q]/4
                qubitop += (QubitOperator((), coef) +
                            QubitOperator("Z"+str(p), -coef) +
                            QubitOperator("Z"+str(q), -coef) +
                            QubitOperator("Z"+str(p)+" Z"+str(q), coef))
        return qubitop

    @property
    def hc(self):
        return self._hc

    @property
    def hr1(self):
        """The value of hr1."""
        return self._hr1

    @property
    def hr2(self):
        """The value of hr2."""
        return self._hr2

    # Override root class 
    def __getitem__(self, args):
        """Look up matrix element.

        Args:
            args: Tuples indicating which coefficient to get. For instance,
                `my_tensor[(6, 1), (8, 1), (2, 0)]`
                returns
                `my_tensor.n_body_tensors[1, 1, 0][6, 8, 2]`
        """
        if len(args) == 0:
            return self.n_body_tensors[()]
        index = tuple([operator[0] for operator in args])
        key = tuple([operator[1] for operator in args])
        return self.n_body_tensors[key][index]

    # Override root class
    def __setitem__(self, args, value):
        # This is not well-defined for a DOCIHamiltonian as we want to keep
        # certain tensor elements within the class the same. Better to
        # make the user update the hr1/hr2/hc terms --- if they really
        # want to play around with the n_body_tensors here they should
        # be castimt this to a raw PolynomialTensor.
        raise TypeError('Raw edits of the n_body_tensors of a DOCIHamiltonian '
                        'is not allowed. Either adjust the hc/hr1/hr2 terms '
                        'or cast to another PolynomialTensor class.')

    @classmethod
    def from_integrals(cls, constant, one_body_integrals, two_body_integrals):
        # TODO: discuss whether this is is an appropriate design pattern
        # to include.
        hc, hr1, hr2 = get_doci_from_integrals(one_body_integrals,
                                               two_body_integrals)
        return cls(constant, hc, hr1, hr2)

    @classmethod
    def zero(cls, n_qubits):
        return cls(0,
                   numpy.zeros((n_qubits,), dtype=numpy.complex128),
                   numpy.zeros((n_qubits,) * 2, dtype=numpy.complex128),
                   numpy.zeros((n_qubits,) * 2, dtype=numpy.complex128))


def get_tensors_from_doci(hc, hr1, hr2):
    '''Makes the one and two-body tensors from the DOCI wavefunctions

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
    one_body_integrals, two_body_integrals =\
        get_projected_integrals_from_doci(hc, hr1, hr2)
    one_body_coefficients, two_body_coefficients = get_tensor_from_integrals(
        one_body_integrals, two_body_integrals)
    return one_body_coefficients, two_body_coefficients


def get_projected_integrals_from_doci(hc, hr1, hr2):
    '''Makes the one and two-body integrals from the DOCI projection
    from the hr1 and hr2 matrices.

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
    projected_onebody_integrals = numpy.zeros((n_qubits, n_qubits))
    projected_twobody_integrals = numpy.zeros((n_qubits, n_qubits, n_qubits,
                                               n_qubits))
    for p in range(n_qubits):
        projected_onebody_integrals[p, p] = hc[p]
        for q in range(n_qubits):
            projected_twobody_integrals[p, q, q, p] = hr2[p, q] / 2
            if p == q:
                continue
            projected_twobody_integrals[p, p, q, q] = hr1[p, q]

    return projected_onebody_integrals, projected_twobody_integrals


def get_doci_from_integrals(one_body_integrals,
                            two_body_integrals):
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
        hc[p] = one_body_integrals[p, p]
        for q in range(n_qubits):
            hr2[p, q] = (2*two_body_integrals[p, q, q, p] -
                         two_body_integrals[p, q, p, q])
            if p == q:
                continue
            hr1[p, q] = two_body_integrals[p, p, q, q]

    return hc, hr1, hr2
