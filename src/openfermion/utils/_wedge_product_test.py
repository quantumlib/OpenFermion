from itertools import product
import pytest
import numpy

from openfermion.utils._wedge_product import (wedge,
                                              generate_parity_permutations)


def test_upper_lower_index_size_check():
    n_tensor = numpy.zeros((2, 2, 2))
    m_tensor = numpy.zeros((2, 2))
    with pytest.raises(IndexError):
        wedge(n_tensor, m_tensor, (3, 1), (1, 1))

    with pytest.raises(IndexError):
        wedge(n_tensor, m_tensor, (4, 0), (1, 2))

    with pytest.raises(IndexError):
        wedge(n_tensor, m_tensor, (2, 1), (1, 2))


def test_eye_wedge():
    """
    Test wedge product code through identity wedge

    (p, s)(q, r) - (q, s)(p, r) - (p, r)(q, s) + (q, r)(p, s)
    """
    n_tensor = numpy.eye(2)
    m_tensor = numpy.eye(2)
    true_eye2 = numpy.zeros((2, 2, 2, 2))
    for p, q, r, s in product(range(2), repeat=4):
        true_eye2[p, q, r, s] += (n_tensor[p, s] * m_tensor[q, r] -
                                  n_tensor[q, s] * m_tensor[p, r] -
                                  n_tensor[p, r] * m_tensor[q, s] +
                                  n_tensor[q, r] * m_tensor[p, s])
    true_eye2 /= 2 ** 2
    test_eye2 = wedge(n_tensor, m_tensor, (1, 1), (1, 1))
    assert numpy.allclose(test_eye2, true_eye2)


def test_random_wedge():
    """
    Test wedge product code through identity wedge

    (p, s)(q, r) - (q, s)(p, r) - (p, r)(q, s) + (q, r)(p, s)
    """
    dim = 4
    n_tensor = numpy.random.random((dim, dim))
    m_tensor = numpy.random.random((dim, dim))
    true_eye2 = numpy.zeros(tuple([dim] * 4))
    for p, q, r, s in product(range(dim), repeat=4):
        true_eye2[p, q, r, s] += (n_tensor[p, s] * m_tensor[q, r] -
                                  n_tensor[q, s] * m_tensor[p, r] -
                                  n_tensor[p, r] * m_tensor[q, s] +
                                  n_tensor[q, r] * m_tensor[p, s])
    true_eye2 /= 2 ** 2
    test_eye2 = wedge(n_tensor, m_tensor, (1, 1), (1, 1))
    assert numpy.allclose(test_eye2, true_eye2)


def test_random_two_wedge():
    """
    Test wedge product code with random 4-tensor and a random 2-tensor

    (a, b, c, d) wedge (e, f) -> (a, b, e, f, c, d)

    """
    dim = 3
    n_tensor = numpy.random.random((dim, dim, dim, dim))
    m_tensor = numpy.random.random((dim, dim))
    true_tensor = numpy.zeros(tuple([dim] * 6))
    for a, b, c, d, e, f in product(range(dim), repeat=6):
        for u_perm, u_phase in generate_parity_permutations([a, b, c]):
            for l_perm, l_phase in generate_parity_permutations([f, e, d]):
                true_tensor[a, b, c, d, e, f] += u_phase * l_phase * n_tensor[
                    u_perm[0], u_perm[1], l_perm[1], l_perm[0]] * m_tensor[
                                                     u_perm[2], l_perm[2]]

    true_tensor /= 6 ** 2
    test_tensor = wedge(n_tensor, m_tensor, (2, 2), (1, 1))
    assert numpy.allclose(test_tensor, true_tensor)


def test_random_two_wedge_two():
    """
    Test wedge product code with a random 4-tensor and a random 4-tensor

    (a, b, h, g) wedge (c, d, f, e) -> (a, b, c, d, e, f, g, h)

    """
    dim = 3
    n_tensor = numpy.random.random((dim, dim, dim, dim))
    m_tensor = numpy.random.random((dim, dim, dim, dim))
    true_tensor = numpy.zeros(tuple([dim] * 8))
    for a, b, c, d, e, f, g, h in product(range(dim), repeat=8):
        for u_perm, u_phase in generate_parity_permutations([a, b, c, d]):
            for l_perm, l_phase in generate_parity_permutations([h, g, f, e]):
                # D_{hgfe}^{abcd} = C [D_{hg}^{ab}D_{fe}^{cd} +
                # -1 * D_{hg}^{ab}D_{ef}^{cd} + ...]
                true_tensor[a, b, c, d, e, f, g, h] += u_phase * l_phase * \
                                                       n_tensor[
                                                           u_perm[0], u_perm[1],
                                                           l_perm[1], l_perm[
                                                               0]] * m_tensor[
                                                           u_perm[2], u_perm[3],
                                                           l_perm[3], l_perm[2]]

    true_tensor /= (4 * 3 * 2) ** 2
    test_tensor = wedge(n_tensor, m_tensor, (2, 2), (2, 2))
    assert numpy.allclose(test_tensor, true_tensor)


def test_random_1_2_wedge_1_1():
    dim = 3
    n_tensor = numpy.random.random((dim, dim, dim))
    m_tensor = numpy.random.random((dim, dim))
    true_tensor = numpy.zeros(tuple([dim] * 5))
    for a, b, c, d, e in product(range(dim), repeat=5):
        for u_perm, u_phase in generate_parity_permutations([a, b]):
            for l_perm, l_phase in generate_parity_permutations([e, d, c]):
                # D_{edc}^{ab} = C[D_{ed}^{a}D_{c}^{b} - D_{ed}^{b}D_{c}^{a} -
                # D_{ec}^{a}D_{d}^{b} +...]
                true_tensor[a, b, c, d, e] += u_phase * l_phase * n_tensor[
                    u_perm[0], l_perm[2], l_perm[1]] * m_tensor[
                                                  u_perm[1], l_perm[0]]

    true_tensor /= (2 * 3 * 2)
    test_tensor = wedge(n_tensor, m_tensor, (1, 2), (1, 1))
    assert numpy.allclose(test_tensor, true_tensor)


def test_random_2_1_wedge_1_1():
    dim = 3
    n_tensor = numpy.random.random((dim, dim, dim))
    m_tensor = numpy.random.random((dim, dim))
    true_tensor = numpy.zeros(tuple([dim] * 5))
    for a, b, c, d, e in product(range(dim), repeat=5):
        for u_perm, u_phase in generate_parity_permutations([a, b, c]):
            for l_perm, l_phase in generate_parity_permutations([e, d]):
                true_tensor[a, b, c, d, e] += u_phase * l_phase * n_tensor[
                    u_perm[0], u_perm[1], l_perm[0]] * m_tensor[
                                                  u_perm[2], l_perm[1]]

    true_tensor /= (2 * 3 * 2)
    test_tensor = wedge(n_tensor, m_tensor, (2, 1), (1, 1))
    assert numpy.allclose(test_tensor, true_tensor)
