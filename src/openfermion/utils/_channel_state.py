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

"""Module to manipulate basic models of quantum channels"""

from functools import reduce
from itertools import chain
from numpy import array, conj, dot, eye, kron, log2, sqrt


def _verify_channel_inputs(density_matrix, probability, target_qubit):
    """Verifies input parameters for channels

    Args:
        density_matrix (numpy.ndarray): Density matrix of the system
        probability (float): Probability error is applied p \in [0, 1]
        target_qubit (int): target for the channel error.

    Returns:
        new_density_matrix(numpy.ndarray): Density matrix with the channel
            applied.
    """
    n_qubits = int(log2(density_matrix.shape[0]))

    if (len(density_matrix.shape) != 2 or
            density_matrix.shape[0] != density_matrix.shape[1]):
        raise ValueError("Error in input of density matrix to channel.")
    if (probability < 0) or (probability > 1):
        raise ValueError("Channel probability must be between 0 and 1.")
    if (target_qubit < 0) or (target_qubit >= n_qubits):
        raise ValueError("Target qubits must be within number of qubits.")


def _lift_operator(operator, n_qubits, target_qubit):
    """Lift a single qubit operator into the n_qubit space by kron product

    Args:
        operator (ndarray): Single qubit operator to lift into full space
        n_qubits (int): Number of total qubits in the space
        target_qubit (int): Qubit to act on

    Return:
        new_operator(Sparse Operator): Operator representing the embedding in
            the full space.
    """
    new_operator = (
        reduce(kron,
               chain(
                   (eye(2) for i in range(0, target_qubit)),
                   [operator],
                   (eye(2) for i in range(target_qubit + 1, n_qubits)))))
    return new_operator


def amplitude_damping_channel(density_matrix, probability, target_qubit,
                              transpose=False):
    """Apply an amplitude damping channel

    Applies an amplitude damping channel with a given probability to the target
    qubit in the density_matrix.

    Args:
        density_matrix (numpy.ndarray): Density matrix of the system
        probability (float): Probability error is applied p \in [0, 1]
        target_qubit (int): target for the channel error.
        transpose (bool): Conjugate transpose channel operators, useful for
            acting on Hamiltonians in variational channel state models

    Returns:
        new_density_matrix(numpy.ndarray): Density matrix with the channel
            applied.
    """
    _verify_channel_inputs(density_matrix, probability, target_qubit)
    n_qubits = int(log2(density_matrix.shape[0]))

    E0 = _lift_operator(array([[1.0, 0.0],
                               [0.0, sqrt(1.0 - probability)]], dtype=complex),
                        n_qubits, target_qubit)

    E1 = _lift_operator(array([[0.0, sqrt(probability)],
                               [0.0, 0.0]], dtype=complex),
                        n_qubits, target_qubit)

    if transpose:
        E0 = E0.T
        E1 = E1.T

    new_density_matrix = (dot(E0, dot(density_matrix, E0.T)) +
                          dot(E1, dot(density_matrix, E1.T)))

    return new_density_matrix


def dephasing_channel(density_matrix, probability, target_qubit,
                      transpose=False):
    """Apply a dephasing channel

    Applies an amplitude damping channel with a given probability to the target
    qubit in the density_matrix.

    Args:
        density_matrix (numpy.ndarray): Density matrix of the system
        probability (float): Probability error is applied p \in [0, 1]
        target_qubit (int): target for the channel error.
        transpose (bool): Conjugate transpose channel operators, useful for
            acting on Hamiltonians in variational channel state models

    Returns:
        new_density_matrix (numpy.ndarray): Density matrix with the channel
            applied.
    """
    _verify_channel_inputs(density_matrix, probability, target_qubit)
    n_qubits = int(log2(density_matrix.shape[0]))

    E0 = _lift_operator(sqrt(1.0 - probability/2.) * eye(2),
                        n_qubits, target_qubit)
    E1 = _lift_operator(sqrt(probability/2.) *
                        array([[1.0, 0.0], [1.0, -1.0]]),
                        n_qubits, target_qubit)

    if transpose:
        E0 = E0.T
        E1 = E1.T

    new_density_matrix = (dot(E0, dot(density_matrix, E0.T)) +
                          dot(E1, dot(density_matrix, E1.T)))

    return new_density_matrix


def depolarizing_channel(density_matrix, probability, target_qubit,
                         transpose=False):
    """Apply a depolarizing channel

    Applies an amplitude damping channel with a given probability to the target
    qubit in the density_matrix.

    Args:
        density_matrix (numpy.ndarray): Density matrix of the system
        probability (float): Probability error is applied p \in [0, 1]
        target_qubit (int/str): target for the channel error, if given special
            value "all", then a total depolarizing channel is applied.
        transpose (bool): Dummy parameter to match signature of other
            channels but depolarizing channel is symmetric under
            conjugate transpose.

    Returns:
        new_density_matrix (numpy.ndarray): Density matrix with the channel
            applied.
    """
    n_qubits = int(log2(density_matrix.shape[0]))

    # Toggle depolarizing channel on all qubits
    if isinstance(target_qubit, str) and target_qubit.lower() == "all":
        dimension = density_matrix.shape[0]
        new_density_matrix = ((1.0 - probability) * density_matrix +
                              probability * eye(dimension) / float(dimension))
        return new_density_matrix

    # For any other case, depolarize only the target qubit
    _verify_channel_inputs(density_matrix, probability, target_qubit)

    E0 = _lift_operator(sqrt(1.0 - probability) * eye(2),
                        n_qubits, target_qubit)
    E1 = _lift_operator(sqrt(probability / 3.) * array([[0.0, 1.0],
                                                        [1.0, 0.0]]),
                        n_qubits, target_qubit)
    E2 = _lift_operator(sqrt(probability / 3.) * array([[0.0, -1.0j],
                                                        [1.0j, 0.0]]),
                        n_qubits, target_qubit)
    E3 = _lift_operator(sqrt(probability / 3.) * array([[1.0, 0.0],
                                                        [0.0, -1.0]]),
                        n_qubits, target_qubit)

    new_density_matrix = (dot(E0, dot(density_matrix, E0)) +
                          dot(E1, dot(density_matrix, E1)) +
                          dot(E2, dot(density_matrix, E2)) +
                          dot(E3, dot(density_matrix, E3)))

    return new_density_matrix
