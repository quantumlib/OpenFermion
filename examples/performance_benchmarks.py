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

"""This file contains tests of code performance to reveal bottlenecks."""
import time
import logging

import numpy

from openfermion.ops import (FermionOperator,
                             QubitOperator,
                             normal_ordered)
from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermion.utils import jordan_wigner_sparse, get_linear_qubit_operator
from openfermion.utils._testing_utils import random_interaction_operator


def benchmark_mo_jordan_wigner(n_qubits):
    """Test speed with which molecular operators transform to qubit operators.

    Args:
        n_qubits: The size of the molecular operator instance. Ideally, we
            would be able to transform to a qubit operator for 50 qubit
            instances in less than a minute. We are way too slow right now.

    Returns:
        runtime: The number of seconds required to make the conversion.
    """
    # Get an instance of InteractionOperator.
    molecular_operator = random_interaction_operator(n_qubits)

    # Convert to a qubit operator.
    start = time.time()
    _ = jordan_wigner(molecular_operator)
    end = time.time()

    # Return runtime.
    runtime = end - start
    return runtime


def benchmark_fermion_math_order(n_qubits, term_length, power):
    """Benchmark both arithmetic operators and normal ordering on fermions.

    The idea is we generate two random FermionTerms, A and B, each acting
    on n_qubits with term_length operators. We then compute
    (A + B) ** power. This is costly that is the first benchmark. The second
    benchmark is in normal ordering whatever comes out.

    Args:
        n_qubits: The number of qubits on which these terms act.
        term_length: The number of operators in each term.
        power: Int, the exponent to which to raise sum of the two terms.

    Returns:
        runtime_math: The time it takes to perform (A + B) ** power
        runtime_normal_order: The time it takes to perform
            FermionOperator.normal_order()
    """
    # Generate random operator strings.
    operators_a = [(numpy.random.randint(n_qubits),
                    numpy.random.randint(2))]
    operators_b = [(numpy.random.randint(n_qubits),
                    numpy.random.randint(2))]
    for _ in range(term_length):

        # Make sure the operator is not trivially zero.
        operator_a = (numpy.random.randint(n_qubits),
                      numpy.random.randint(2))
        while operator_a == operators_a[-1]:
            operator_a = (numpy.random.randint(n_qubits),
                          numpy.random.randint(2))
        operators_a += [operator_a]

        # Do the same for the other operator.
        operator_b = (numpy.random.randint(n_qubits),
                      numpy.random.randint(2))
        while operator_b == operators_b[-1]:
            operator_b = (numpy.random.randint(n_qubits),
                          numpy.random.randint(2))
        operators_b += [operator_b]

    # Initialize FermionTerms and then sum them together.
    fermion_term_a = FermionOperator(tuple(operators_a),
                                     float(numpy.random.randn()))
    fermion_term_b = FermionOperator(tuple(operators_b),
                                     float(numpy.random.randn()))
    fermion_operator = fermion_term_a + fermion_term_b

    # Exponentiate.
    start_time = time.time()
    fermion_operator **= power
    runtime_math = time.time() - start_time

    # Normal order.
    start_time = time.time()
    normal_ordered(fermion_operator)
    runtime_normal_order = time.time() - start_time

    # Return.
    return runtime_math, runtime_normal_order


def benchmark_jordan_wigner_sparse(n_qubits):
    """Benchmark the speed at which a FermionOperator is mapped to a matrix.

    Args:
        n_qubits: The number of qubits in the example.

    Returns:
        runtime: The time in seconds that the benchmark took.
    """
    # Initialize a random FermionOperator.
    molecular_operator = random_interaction_operator(n_qubits)
    fermion_operator = get_fermion_operator(molecular_operator)

    # Map to SparseOperator class.
    start_time = time.time()
    _ = jordan_wigner_sparse(fermion_operator)
    runtime = time.time() - start_time
    return runtime


def benchmark_linear_qubit_operator(n_qubits, n_terms):
    """Test get_linear_qubit_operator() from a Qubit Operator.

    Args:
        n_qubits: The number of qubits, implying the dimension of the operator
            is 2 ** n_qubits.
        n_terms: The number of terms in a qubit operator.

    Returns:
        runtime_operator: The time it takes to get the linear operator.
        runtime_matvec: The time it takes to perform matrix multiplication.
    """
    # Generates Qubit Operator with specified number of terms.
    map_int_to_op = {
        0: 'X',
        1: 'Y',
        2: 'Z',
    }
    qubit_operator = QubitOperator.zero()
    for _ in xrange(n_terms):
        tuples = []
        for i in xrange(n_qubits):
            int_operator = numpy.random.randint(4)
            # 3 is 'I', so just skip.
            if int_operator > 2:
                continue
            tuples.append((i, map_int_to_op[int_operator]))
        if tuples:
            qubit_operator += QubitOperator(tuples, 1.00)

    # Gets an instance of LinearOperator.
    start = time.time()
    linear_operator = get_linear_qubit_operator(qubit_operator, n_qubits)
    end = time.time()
    runtime_operator = end - start

    vec = numpy.random.rand(2 ** n_qubits)
    # Performs matrix multiplication.
    start = time.time()
    _ = linear_operator * vec
    end = time.time()
    runtime_matvec = end - start
    return runtime_operator, runtime_matvec


# Sets up each benchmark run.
def run_molecular_op_jordan_wigner():
    """Run InteractionOperator.jordan_wigner_transform() benchmark."""
    n_qubits = 18
    logging.info('Starting test on InteractionOperator.jordan_wigner_transform()')
    runtime = benchmark_mo_jordan_wigner(n_qubits)
    logging.info('InteractionOperator.jordan_wigner_transform() took '
                 '%f seconds with %d qubits.\n', runtime, n_qubits)

def run_fermion_math_order():
    """Run benchmark on FermionOperator math and normal-ordering."""
    n_qubits = 20
    term_length = 10
    power = 15
    logging.info('Starting test on FermionOperator math and normal ordering.')
    runtime_math, runtime_normal = benchmark_fermion_math_order(
        n_qubits, term_length, power)
    logging.info('Math took %f seconds. Normal ordering took %f seconds.\n',
                 runtime_math, runtime_normal)

def run_jordan_wigner_sparse():
    """Run FermionOperator.jordan_wigner_sparse() benchmark."""
    n_qubits = 10
    logging.info('Starting test on FermionOperator.jordan_wigner_sparse().')
    runtime = benchmark_jordan_wigner_sparse(n_qubits)
    logging.info('Construction of SparseOperator took %f seconds.\n', runtime)

def run_get_linear_qubit_operator():
    """Run get_linear_qubit_operator() benchmark."""
    n_qubits = 20
    n_terms = 10

    logging.info('Starting test on get_linear_qubit_operator().')
    runtime_operator, runtime_matvec = benchmark_linear_qubit_operator(
        n_qubits, n_terms)
    logging.info('Linear Operator took %f seconds. '
                 'Matrix multiplication took %f seconds, '
                 'with (n_qubits, n_terms) = (%d, %d)\n',
                 runtime_operator, runtime_matvec, n_qubits, n_terms)


# Run benchmarks.
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Seed random number generator.
    numpy.random.seed(8)

    run_molecular_op_jordan_wigner()
    run_fermion_math_order()
    run_jordan_wigner_sparse()
    run_get_linear_qubit_operator()
