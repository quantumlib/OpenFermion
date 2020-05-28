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

from openfermion import (commutator,
                         FermionOperator,
                         Grid,
                         jellium_model,
                         jordan_wigner,
                         normal_ordered,
                         QubitOperator)
from openfermion.transforms import get_fermion_operator
from openfermion.utils import (
    jordan_wigner_sparse,
    LinearQubitOperator,
    LinearQubitOperatorOptions,
    ParallelLinearQubitOperator)
from openfermion.utils._testing_utils import random_interaction_operator
from openfermion.utils._commutator_diagonal_coulomb_operator import (
    commutator_ordered_diagonal_coulomb_with_two_body_operator)


def benchmark_molecular_operator_jordan_wigner(n_qubits):
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


def benchmark_fermion_math_and_normal_order(n_qubits, term_length, power):
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


def benchmark_linear_qubit_operator(n_qubits, n_terms, processes=None):
    """Test speed with getting a linear operator from a Qubit Operator.

    Args:
        n_qubits: The number of qubits, implying the dimension of the operator
            is 2 ** n_qubits.
        n_terms: The number of terms in a qubit operator.
        processes: The number of processors to use.

    Returns:
        runtime_operator: The time it takes to get the linear operator.
        runtime_matvec: The time it takes to perform matrix multiplication.
    """
    # Generates Qubit Operator with specified number of terms.
    map_int_to_operator = {
        0: 'X',
        1: 'Y',
        2: 'Z',
    }
    qubit_operator = QubitOperator.zero()
    for _ in range(n_terms):
        tuples = []
        for i in range(n_qubits):
            operator = numpy.random.randint(4)
            # 3 is 'I', so just skip.
            if operator > 2:
                continue
            tuples.append((i, map_int_to_operator[operator]))
        if tuples:
            qubit_operator += QubitOperator(tuples, 1.00)

    # Gets an instance of (Parallel)LinearQubitOperator.
    start = time.time()
    if processes is None:
        linear_operator = LinearQubitOperator(qubit_operator, n_qubits)
    else:
        linear_operator = ParallelLinearQubitOperator(
            qubit_operator, n_qubits,
            LinearQubitOperatorOptions(processes=processes))

    end = time.time()
    runtime_operator = end - start

    vec = numpy.random.rand(2 ** n_qubits)
    # Performs matrix multiplication.
    start = time.time()
    _ = linear_operator * vec
    end = time.time()
    runtime_matvec = end - start
    return runtime_operator, runtime_matvec


def benchmark_commutator_diagonal_coulomb_operators_2D_spinless_jellium(
        side_length):
    """Test speed of computing commutators using specialized functions.

    Args:
        side_length: The side length of the 2D jellium grid. There are
            side_length ** 2 qubits, and O(side_length ** 4) terms in the
            Hamiltonian.

    Returns:
        runtime_commutator: The time it takes to compute a commutator, after
            partitioning the terms and normal ordering, using the regular
            commutator function.
        runtime_diagonal_commutator: The time it takes to compute the same
            commutator using methods restricted to diagonal Coulomb operators.
    """
    hamiltonian = normal_ordered(jellium_model(Grid(2, side_length, 1.),
                                               plane_wave=False))

    part_a = FermionOperator.zero()
    part_b = FermionOperator.zero()
    add_to_a_or_b = 0  # add to a if 0; add to b if 1
    for term, coeff in hamiltonian.terms.items():
        # Partition terms in the Hamiltonian into part_a or part_b
        if add_to_a_or_b:
            part_a += FermionOperator(term, coeff)
        else:
            part_b += FermionOperator(term, coeff)
        add_to_a_or_b ^= 1

    start = time.time()
    _ = normal_ordered(commutator(part_a, part_b))
    end = time.time()
    runtime_commutator = end - start

    start = time.time()
    _ = commutator_ordered_diagonal_coulomb_with_two_body_operator(
        part_a, part_b)
    end = time.time()
    runtime_diagonal_commutator = end - start

    return runtime_commutator, runtime_diagonal_commutator


# Sets up each benchmark run.
def run_molecular_operator_jordan_wigner(n_qubits=18):
    """Run InteractionOperator.jordan_wigner_transform() benchmark."""
    logging.info('Starting test on '
                 'InteractionOperator.jordan_wigner_transform()')
    logging.info('n_qubits = %d.', n_qubits)
    runtime = benchmark_molecular_operator_jordan_wigner(n_qubits)
    logging.info('InteractionOperator.jordan_wigner_transform() takes %f '
                 'seconds.\n', runtime)

    return runtime


def run_fermion_math_and_normal_order(n_qubits=20, term_length=10, power=15):
    """Run benchmark on FermionOperator math and normal-ordering."""
    logging.info('Starting test on FermionOperator math and normal ordering.')
    logging.info('(n_qubits, term_length, power) = (%d, %d, %d).', n_qubits,
                 term_length, power)
    runtime_math, runtime_normal = benchmark_fermion_math_and_normal_order(
        n_qubits, term_length, power)
    logging.info('Math took %f seconds. Normal ordering took %f seconds.\n',
                 runtime_math, runtime_normal)

    return runtime_math, runtime_normal


def run_jordan_wigner_sparse(n_qubits=10):
    """Run FermionOperator.jordan_wigner_sparse() benchmark."""
    logging.info('Starting test on FermionOperator.jordan_wigner_sparse().')
    logging.info('n_qubits = %d.', n_qubits)
    runtime = benchmark_jordan_wigner_sparse(n_qubits)
    logging.info('Construction of SparseOperator took %f seconds.\n', runtime)

    return runtime


def run_linear_qubit_operator(n_qubits=16, n_terms=10, processes=10):
    """Run linear_qubit_operator benchmark."""

    logging.info('Starting test on linear_qubit_operator().')
    logging.info('(n_qubits, n_terms) = (%d, %d).', n_qubits, n_terms)
    _, runtime_sequential = benchmark_linear_qubit_operator(n_qubits, n_terms)
    _, runtime_parallel = benchmark_linear_qubit_operator(n_qubits, n_terms,
                                                          processes)
    logging.info('LinearQubitOperator took %f seconds, while '
                 'ParallelQubitOperator took %f seconds with %d processes, '
                 'and ratio is %.2f.\n', runtime_sequential, runtime_parallel,
                 processes, runtime_sequential / runtime_parallel)

    return runtime_sequential, runtime_parallel


def run_diagonal_commutator(side_length=4):
    """Run commutator_diagonal_coulomb_operators benchmark."""

    logging.info(
        'Starting test on '
        'commutator_ordered_diagonal_coulomb_with_two_body_operator().')
    runtime_commutator, runtime_diagonal_commutator = (
        benchmark_commutator_diagonal_coulomb_operators_2D_spinless_jellium(
            side_length=side_length))
    logging.info('Regular commutator computation took %f seconds, while '
                 'commutator_ordered_diagonal_coulomb_with_two_body_operator'
                 ' took %f seconds. Ratio is %.2f.\n', runtime_commutator,
                 runtime_diagonal_commutator,
                 runtime_commutator / runtime_diagonal_commutator)

    return runtime_commutator, runtime_diagonal_commutator


# Run benchmarks.
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Seed random number generator.
    numpy.random.seed(8)

    run_molecular_operator_jordan_wigner()
    run_fermion_math_and_normal_order()
    run_jordan_wigner_sparse()
    run_linear_qubit_operator()
    run_diagonal_commutator()
