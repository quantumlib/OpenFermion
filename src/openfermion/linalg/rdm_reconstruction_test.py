import numpy as np
from openfermion.linalg.rdm_reconstruction import valdemoro_reconstruction
from openfermion.linalg.wedge_product import wedge


def test_valdemoro_correct_size():
    m = 4
    opdm = np.random.random((m, m))
    tpdm = wedge(opdm, opdm, (1, 1), (1, 1))
    d3v = valdemoro_reconstruction(tpdm, 4)
    assert d3v.shape == (m, m, m, m, m, m)


def test_correct_trace():
    opdm = np.diag([1] * 2 + [0] * 2)
    tpdm = wedge(opdm, opdm, (1, 1), (1, 1))
    assert np.isclose(np.einsum('ijji', tpdm), 2 * (2 - 1) / 2)
    d3v = valdemoro_reconstruction(tpdm, 2)
    # because only 2 particles in this system
    assert np.isclose(np.einsum('ijkkji', d3v), 0)

    opdm = np.diag([1] * 5 + [0] * 3)
    tpdm = wedge(opdm, opdm, (1, 1), (1, 1))
    assert np.isclose(np.einsum('ijji', tpdm), 5 * (5 - 1) / 2)
    d3v = valdemoro_reconstruction(tpdm, 5)
    assert np.isclose(np.einsum('ijkkji', d3v), 5 * (5 - 1) * (5 - 2) / 6)