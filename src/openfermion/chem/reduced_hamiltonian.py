from itertools import product
import numpy
from openfermion.ops.representations import InteractionOperator


def make_reduced_hamiltonian(
    molecular_hamiltonian: InteractionOperator, n_electrons: int
) -> InteractionOperator:
    r"""
    Construct the reduced Hamiltonian.

    This Hamiltonian is equivalent to the electronic structure Hamiltonian
    but contains only two-body terms.  To do this, the operator now depends
    on the number of particles being simulated.  We use the RDM sum rule to
    lift the 1-body terms to the two-body space.

    Derivation:
        use the fact that i^l = (1/(n -1)) sum_{jk}\delta_{jk}i^ j^ k l
                          i^l = (-1/(n -1)) sum_{jk}\delta_{jk}j^ i^ k l
                          i^l = (-1/(n -1)) sum_{jk}\delta_{jk}i^ j^ l k
                          i^l = (1/(n -1)) sum_{jk}\delta_{jk}j^ i^ l k

        Rewrite each one-body term as an even weighting of all four 2-RDM
        elements with delta functions. Then rearrange terms so that each ijkl
        term gets a sum of permuted one-body terms multiplied by delta
        function. One should notice that this results in the same formula
        if one was to apply the wedge product!

    Args:
        molecular_hamiltonian: operator to write reduced hamiltonian for
        n_electrons: number of electrons in the system
    Returns:
        InteractionOperator with a zero one-body component.
    """
    constant = molecular_hamiltonian.constant
    h1 = molecular_hamiltonian.one_body_tensor
    h2 = molecular_hamiltonian.two_body_tensor

    delta = numpy.eye(h1.shape[0])
    k2 = numpy.zeros_like(h2)
    normalization = 1 / (4 * (n_electrons - 1))
    for i, j, k, l in product(range(h1.shape[0]), repeat=4):
        k2[i, j, k, l] = (
            normalization
            * (
                h1[i, l] * delta[j, k]
                + h1[j, k] * delta[i, l]
                - h1[i, k] * delta[j, l]
                - h1[j, l] * delta[i, k]
            )
            + h2[i, j, k, l]
        )

    return InteractionOperator(constant, numpy.zeros_like(h1), k2)
