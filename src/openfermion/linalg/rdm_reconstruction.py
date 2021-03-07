import numpy as np
from openfermion.linalg.wedge_product import wedge


def valdemoro_reconstruction(tpdm, n_electrons):
    """
    Build a 3-RDM by cumulant expansion and setting 3rd cumulant to zero

    d3 approx = D ^ D ^ D + 3 (2C) ^ D

    tpdm has normalization (n choose 2) where n is the number of electrons

    Args:
        tpdm (np.ndarray): four-tensor representing the two-RDM
        n_electrons (int): number of electrons in the system
    Returns:
        six-tensor reprsenting the three-RDM
    """
    opdm = (2 / (n_electrons - 1)) * np.einsum('ijjk', tpdm)
    unconnected_tpdm = wedge(opdm, opdm, (1, 1), (1, 1))
    unconnected_d3 = wedge(opdm, unconnected_tpdm, (1, 1), (2, 2))
    return 3 * wedge(tpdm, opdm, (2, 2), (1, 1)) - 2 * unconnected_d3
