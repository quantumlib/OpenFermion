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


class DOCIHamiltonian(object):
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

    def __init__(self, constant, hr1, hr2):
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
        self._constant = None
        self._hr1 = None
        self._hr2 = None
        self.constant = constant
        self.hr1 = hr1
        self.hr2 = hr2

    @property
    def qubit_operator(self):
        """Return the QubitOperator representation of this DOCI Hamiltonian"""
        return QubitOperator((), self.constant) + self.z_part + self.xy_part

    @property
    def xy_part(self):
        """Return the XX and YY part of the QubitOperator representation of this DOCI Hamiltonian"""
        qubitop = QubitOperator()
        n_qubits = self.hr1.shape[0]
        for p in range(n_qubits):
            for q in range(n_qubits):
                if p == q:
                    continue
                qubitop += QubitOperator("X"+str(p)+" X"+str(q), self.hr1[p, q]/4) + QubitOperator("Y"+str(p)+" Y"+str(q), self.hr1[p, q]/4)

        return qubitop

    @property
    def z_part(self):
        """Return the Z and ZZ part of the QubitOperator representation of this DOCI Hamiltonian"""
        qubitop = QubitOperator()
        n_qubits = self.hr1.shape[0]
        for p in range(n_qubits):
            qubitop += QubitOperator((), self.hr1[p, p]/2) - QubitOperator("Z"+str(p), self.hr1[p, p]/2)
            for q in range(n_qubits):
                if p == q:
                    continue
                coef = self.hr2[p,q]/4
                qubitop += QubitOperator((), coef) + QubitOperator("Z"+str(p), -coef) + QubitOperator("Z"+str(q), -coef) + QubitOperator("Z"+str(p)+" Z"+str(q), coef)

        return qubitop

    @property
    def constant(self):
        """The value of constant."""
        return self._constant

    @constant.setter
    def constant(self, value):
        """Set the value of constant."""
        self._constant = value

    @property
    def hr1(self):
        """The value of hr1."""
        return self._hr1

    @hr1.setter
    def hr1(self, value):
        """Set the value of hr1."""
        self._hr1 = value

    @property
    def hr2(self):
        """The value of hr2."""
        return self._hr2

    @hr2.setter
    def hr2(self, value):
        """Set the value of hr2."""
        self._hr2 = value

    @classmethod
    def from_integrals(cls, constant, one_body_integrals, two_body_integrals):
        r"""Construct a DOCI Hamiltonian from electron integrals

        Args:
            constant: A constant term in the operator given as a
                float. For instance, the nuclear repulsion energy.
            one_body_integrals: Numpy array of one-electron integrals
            two_body_integrals: Numpy array of two-electron integrals
        """
        n_qubits = one_body_integrals.shape[0]
        hr1 = numpy.zeros((n_qubits, n_qubits))
        hr2 = numpy.zeros((n_qubits, n_qubits))
        for p in range(n_qubits):
            hr1[p, p] = 2*one_body_integrals[p, p] + two_body_integrals[p, p, p, p]
            for q in range(n_qubits):
                if p == q:
                    continue
                hr1[p, q] = two_body_integrals[p, p, q, q]
                hr2[p, q] = 2*two_body_integrals[p, q, q, p] - two_body_integrals[p, q, p, q]

        return cls(constant, hr1, hr2)

    @classmethod
    def zero(cls, n_qubits):
        return cls(0, numpy.zeros((n_qubits,) * 2, dtype=numpy.complex128),
                   numpy.zeros((n_qubits,) * 2, dtype=numpy.complex128))
