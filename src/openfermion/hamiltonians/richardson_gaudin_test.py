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
"""Tests for Richardson Gaudin model module."""

import pytest
import numpy as np

from openfermion.hamiltonians import RichardsonGaudin
from openfermion.ops import QubitOperator
from openfermion.transforms import get_fermion_operator
from openfermion.transforms import jordan_wigner
from openfermion.linalg import get_sparse_operator
from openfermion import get_fermion_operator, InteractionOperator, normal_ordered


@pytest.mark.parametrize(
    'g, n_qubits, expected',
    [
        (
            0.3,
            2,
            QubitOperator(
                '3.0 [] + 0.15 [X0 X1] + \
0.15 [Y0 Y1] - 1.0 [Z0] - 2.0 [Z1]'
            ),
        ),
        (
            -0.1,
            3,
            QubitOperator(
                '6.0 [] - 0.05 [X0 X1] - 0.05 [X0 X2] - \
0.05 [Y0 Y1] - 0.05 [Y0 Y2] - 1.0 [Z0] - 0.05 [X1 X2] - \
0.05 [Y1 Y2] - 2.0 [Z1] - 3.0 [Z2]'
            ),
        ),
    ],
)
def test_richardson_gaudin_hamiltonian(g, n_qubits, expected):
    rg = RichardsonGaudin(g, n_qubits)
    rg_qubit = rg.qubit_operator
    assert rg_qubit == expected

    assert np.array_equal(
        np.sort(np.unique(get_sparse_operator(rg_qubit).diagonal())),
        2 * np.array(list(range((n_qubits + 1) * n_qubits // 2 + 1))),
    )


def test_n_body_tensor_errors():
    rg = RichardsonGaudin(1.7, n_qubits=2)
    with pytest.raises(TypeError):
        rg.n_body_tensors = 0
    with pytest.raises(TypeError):
        rg.constant = 1.1


@pytest.mark.parametrize("g, n_qubits", [(0.2, 4), (-0.2, 4)])
def test_fermionic_hamiltonian_from_integrals(g, n_qubits):
    rg = RichardsonGaudin(g, n_qubits)
    # hc, hr1, hr2 = rg.hc, rg.hr1, rg.hr2
    doci = rg
    constant = doci.constant
    reference_constant = 0

    doci_qubit_op = doci.qubit_operator
    doci_mat = get_sparse_operator(doci_qubit_op).toarray()
    doci_eigvals = np.linalg.eigh(doci_mat)[0]

    tensors = doci.n_body_tensors
    one_body_tensors, two_body_tensors = tensors[(1, 0)], tensors[(1, 1, 0, 0)]

    fermion_op = get_fermion_operator(
        InteractionOperator(constant, one_body_tensors, 0.5 * two_body_tensors)
    )
    fermion_op = normal_ordered(fermion_op)
    fermion_mat = get_sparse_operator(fermion_op).toarray()
    fermion_eigvals = np.linalg.eigh(fermion_mat)[0]

    one_body_tensors2, two_body_tensors2 = rg.get_antisymmetrized_tensors()
    fermion_op2 = get_fermion_operator(
        InteractionOperator(reference_constant, one_body_tensors2, 0.5 * two_body_tensors2)
    )
    fermion_op2 = normal_ordered(fermion_op2)
    fermion_mat2 = get_sparse_operator(fermion_op2).toarray()
    fermion_eigvals2 = np.linalg.eigh(fermion_mat2)[0]

    for eigval in doci_eigvals:
        assert any(
            abs(fermion_eigvals - eigval) < 1e-6
        ), "The DOCI spectrum should have \
        been contained in the spectrum of the fermionic operator constructed via the \
        DOCIHamiltonian class"

    for eigval in doci_eigvals:
        assert any(
            abs(fermion_eigvals2 - eigval) < 1e-6
        ), "The DOCI spectrum should have \
       been contained in the spectrum of the fermionic operators constructed via the anti \
       symmetrized tensors"
