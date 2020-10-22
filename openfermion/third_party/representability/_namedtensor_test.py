from itertools import product
import numpy as np
import pytest
from openfermion.third_party.representability._bijections import Bijection, \
    index_index_basis, index_tuple_basis
from openfermion.third_party.representability._namedtensor import Tensor


def test_namedtensor_initialization():
    td = Tensor()
    assert isinstance(td, Tensor)

    data = np.arange(16).reshape((4, 4))
    td = Tensor(name='opdm', tensor=data, basis=index_index_basis(4))
    assert td.name == 'opdm'
    for i in range(4):
        assert td.basis.fwd(i) == i
    assert np.allclose(td.data, data)

    with pytest.raises(TypeError):
        _ = Tensor(name='opdm',
                   tensor=data,
                   basis=dict(zip(range(4), range(4))))

    td = Tensor(name='opdm', tensor=data)
    assert isinstance(td.basis, Bijection)


def test_namedtensor_getdata():
    data = np.arange(16).reshape((4, 4))
    td = Tensor(name='opdm', tensor=data, basis=index_index_basis(4))
    for i, j in product(range(4), repeat=2):
        assert np.isclose(td[i, j], data[i, j])
    blank_td = Tensor(name='opdm')
    with pytest.raises(TypeError):
        _ = blank_td[0, 0]


def test_namedtensor_call():
    """
    make a matrix that has a different basis than indexing
    """
    n = 4
    dim = int(n * (n - 1) / 2)
    geminals = []
    bas = {}
    cnt = 0
    for i in range(4):
        for j in range(i + 1, 4):
            bas[cnt] = (i, j)
            geminals.append((i, j))
            cnt += 1
    rev_bas = dict(zip(bas.values(), bas.keys()))

    rand_mat = np.random.random((dim, dim))
    basis_bijection = index_tuple_basis(geminals)
    test_tensor = Tensor(tensor=rand_mat, basis=basis_bijection)
    assert test_tensor.basis.fwd(0) == (0, 1)
    assert test_tensor.basis.fwd(2) == (0, 3)
    assert test_tensor.basis.rev(test_tensor.basis.fwd(5)) == 5
    assert test_tensor.ndim == 2
    assert test_tensor.dim == dim
    # index into data directly
    assert test_tensor[2, 3] == rand_mat[2, 3]
    # index into data via basis indexing
    assert test_tensor(0, 1, 0, 1) == rand_mat[0, 0]
    assert test_tensor(0, 1, 0, 1) == rand_mat[0, 0]
    assert test_tensor(1, 2, 0, 1) == rand_mat[rev_bas[(1, 2)], rev_bas[(0, 1)]]
    assert test_tensor.index_vectorized(1, 2, 0, 1) == rev_bas[(1, 2)] * dim + \
           rev_bas[(0, 1)]

    # testing iteration over the upper triangle
    for iter_vals in test_tensor.utri_iterator():
        val, [i, j] = iter_vals
        assert val == rand_mat[test_tensor.basis.rev(i),
                               test_tensor.basis.rev(j)]


def test_tensor_index():
    """
    Testing the index_vectorized which is mapped through the basis
    """
    a = np.arange(16).reshape((4, 4), order='C')
    basis = [(0, 0), (0, 1), (1, 0), (1, 1)]
    basis = index_tuple_basis(basis)
    tt = Tensor(tensor=a, basis=basis)
    assert np.allclose(tt.data, a)
    assert tt.size == 16
    assert isinstance(tt.basis, Bijection)
    assert np.isclose(tt.index_vectorized(0, 0, 0, 0), 0)
    assert np.isclose(tt.index_vectorized(0, 0, 0, 1), 1)
    assert np.isclose(tt.index_vectorized(0, 0, 1, 0), 2)
    assert np.isclose(tt.index_vectorized(0, 0, 1, 1), 3)
    assert np.isclose(tt.index_vectorized(0, 1, 0, 0), 4)
    assert np.isclose(tt.index_vectorized(0, 1, 0, 1), 5)
    assert np.isclose(tt.index_vectorized(0, 1, 1, 0), 6)
    assert np.isclose(tt.index_vectorized(0, 1, 1, 1), 7)
    assert np.isclose(tt.index_vectorized(1, 0, 0, 0), 8)
    # etc...

    a = np.arange(16).reshape((4, 4), order='C')
    tt = Tensor(tensor=a)  # the canonical basis
    assert np.isclose(tt.index_vectorized(0, 0), 0)
    assert np.isclose(tt.index_vectorized(0, 1), 1)
    assert np.isclose(tt.index_vectorized(0, 2), 2)
    assert np.isclose(tt.index_vectorized(0, 3), 3)
    assert np.isclose(tt.index_vectorized(1, 0), 4)
    assert np.isclose(tt.index_vectorized(1, 1), 5)
    assert np.isclose(tt.index_vectorized(1, 2), 6)
    assert np.isclose(tt.index_vectorized(1, 3), 7)


def test_tensor_iterator():
    a = np.arange(16).reshape((4, 4))
    test_tensor = Tensor(tensor=a)
    assert np.allclose(test_tensor.data, a)
    assert test_tensor.size == 16
    assert isinstance(test_tensor.basis, Bijection)

    a_triu = a[np.triu_indices_from(a)]
    a_tril = a[np.tril_indices_from(a)]

    counter = 0
    for val, idx in test_tensor.utri_iterator():
        assert val == a[tuple(idx)]
        assert val == a_triu[counter]
        counter += 1
    assert counter == 4 * (4 + 1) / 2

    counter = 0
    for val, idx in test_tensor.ltri_iterator():
        assert val == a[tuple(idx)]
        assert val == a_tril[counter]
        counter += 1
    assert counter == 4 * (4 + 1) / 2

    counter = 0
    for val, idx in test_tensor.all_iterator():
        assert val == a[tuple(idx)]
        counter += 1

    assert np.allclose(test_tensor.vectorize(), a.reshape((-1, 1), order='C'))

    with pytest.raises(TypeError):
        list(test_tensor._iterator('blah'))


def test_get_obj_size():
    assert Tensor.get_obj_size(1) == 1
    assert Tensor.get_obj_size([1, 1]) == 2
    with pytest.raises(TypeError):
        Tensor.get_obj_size('a')


def test_index_bijection():
    with pytest.raises(TypeError):
        Tensor.index_bijection((1, 1, 1, 1), 1, 2)
