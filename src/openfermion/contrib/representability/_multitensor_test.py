import numpy as np
import pytest
import scipy as sp
from openfermion.contrib.representability._namedtensor import Tensor
from openfermion.contrib.representability._multitensor import MultiTensor, TMap
from openfermion.contrib.representability._dualbasis import DualBasis, DualBasisElement


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_tmap(rng: np.random.Generator) -> None:
    a = rng.random((5, 5))
    b = rng.random((4, 4))
    c = rng.random((3, 3))
    at = Tensor(tensor=a, name='a')
    bt = Tensor(tensor=b, name='b')
    ct = Tensor(tensor=c, name='c')
    ttmap = TMap(tensors=[at, bt, ct])
    assert np.allclose(ttmap['a'].data, a)
    assert np.allclose(ttmap['b'].data, b)
    assert np.allclose(ttmap['c'].data, c)

    tmp_tensors = [a, b, c]
    for idx, iterated_tensor in enumerate(ttmap):
        assert np.allclose(iterated_tensor.data, tmp_tensors[idx])


def test_multitensor_init(rng: np.random.Generator) -> None:
    """
    Testing the generation of a multitensor object with random tensors
    """
    a = rng.random((5, 5))
    b = rng.random((4, 4))
    c = rng.random((3, 3))
    at = Tensor(tensor=a, name='a')
    bt = Tensor(tensor=b, name='b')
    ct = Tensor(tensor=c, name='c')
    mt = MultiTensor([at, bt, ct])

    with pytest.raises(TypeError):
        _ = MultiTensor((at, bt))

    assert len(mt.dual_basis) == 0

    assert np.isclose(mt.vec_dim, 5**2 + 4**2 + 3**2)


def test_multitensor_offsetmap(rng: np.random.Generator) -> None:
    a = rng.random((5, 5, 5, 5))
    b = rng.random((4, 4, 4))
    c = rng.random((3, 3))
    at = Tensor(tensor=a, name='a')
    bt = Tensor(tensor=b, name='b')
    ct = Tensor(tensor=c, name='c')
    mt = MultiTensor([at, bt, ct])

    assert mt.off_set_map == {'a': 0, 'b': 5**4, 'c': 5**4 + 4**3}


def test_vectorize_test(rng: np.random.Generator) -> None:
    a = rng.random((5, 5))
    b = rng.random((4, 4))
    c = rng.random((3, 3))
    at = Tensor(tensor=a, name='a')
    bt = Tensor(tensor=b, name='b')
    ct = Tensor(tensor=c, name='c')
    mt = MultiTensor([at, bt, ct])
    vec = np.vstack((at.vectorize(), bt.vectorize()))
    vec = np.vstack((vec, ct.vectorize()))
    assert np.allclose(vec, mt.vectorize_tensors())

    a = rng.random((5, 5, 5, 5))
    b = rng.random((4, 4, 4))
    c = rng.random((3, 3))
    at = Tensor(tensor=a, name='a')
    bt = Tensor(tensor=b, name='b')
    ct = Tensor(tensor=c, name='c')
    mt = MultiTensor([at, bt, ct])
    vec = np.vstack((at.vectorize(), bt.vectorize()))
    vec = np.vstack((vec, ct.vectorize()))
    assert np.allclose(vec, mt.vectorize_tensors())


def test_add_dualelement(rng: np.random.Generator) -> None:
    a = rng.random((5, 5, 5, 5))
    b = rng.random((4, 4, 4))
    c = rng.random((3, 3))
    at = Tensor(tensor=a, name='a')
    bt = Tensor(tensor=b, name='b')
    ct = Tensor(tensor=c, name='c')
    mt = MultiTensor([at, bt, ct])
    assert isinstance(mt.dual_basis, DualBasis)

    dbe = DualBasisElement()
    dbe.add_element('a', (0, 1, 2, 3), 4)
    mt.add_dual_elements(dbe)
    assert len(mt.dual_basis) == 1

    dbe2 = DualBasisElement()
    dbe2.add_element('b', (0, 1, 2), 5)
    mt.add_dual_elements(dbe2)
    assert len(mt.dual_basis) == 2

    A, bias, scalar = mt.synthesize_dual_basis()
    assert A.shape[0] == 2
    # Verify that the elements are correctly added as DualBasisElements
    # and not as their internal tuples (which would cause synthesis to fail).
    assert isinstance(mt.dual_basis[0], DualBasisElement)
    assert isinstance(mt.dual_basis[1], DualBasisElement)


def test_multitensor_init_isolation(rng: np.random.Generator) -> None:
    # Test that different MultiTensor instances don't share the same dual_basis.
    a = rng.random((2, 2))
    at = Tensor(tensor=a, name='a')
    mt1 = MultiTensor([at])
    mt2 = MultiTensor([at])

    dbe = DualBasisElement()
    dbe.add_element('a', (0, 0), 1.0)
    mt1.add_dual_elements(dbe)

    assert len(mt1.dual_basis) == 1
    assert len(mt2.dual_basis) == 0


def test_synthesis_element(rng: np.random.Generator) -> None:
    a = rng.random((5, 5))
    b = rng.random((4, 4))
    c = rng.random((3, 3))
    at = Tensor(tensor=a, name='a')
    bt = Tensor(tensor=b, name='b')
    ct = Tensor(tensor=c, name='c')
    mt = MultiTensor([at, bt, ct])

    dbe = DualBasisElement()
    dbe.add_element('a', (0, 1), 4)
    dbe.add_element('a', (1, 0), 4)
    with pytest.raises(TypeError):
        dbe.add_element(5)  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        mt.add_dual_elements(5)

    mt.add_dual_elements(dbe)
    colidx, data_vals = mt.synthesize_element(dbe)
    assert data_vals == [4, 4]
    assert colidx == [1, 5]
    assert at.data is not None
    assert [at.data[0, 1], at.data[1, 0]] == [at(0, 1), at(1, 0)]


def test_synthesis_dualbasis(rng: np.random.Generator) -> None:
    a = rng.random((5, 5))
    b = rng.random((4, 4))
    c = rng.random((3, 3))
    at = Tensor(tensor=a, name='a')
    bt = Tensor(tensor=b, name='b')
    ct = Tensor(tensor=c, name='c')
    dbe = DualBasisElement()
    dbe.add_element('a', (0, 1), 4)
    dbe.add_element('a', (1, 0), 4)
    mt = MultiTensor([at, bt, ct], DualBasis(elements=[dbe]))

    A, c, b = mt.synthesize_dual_basis()
    assert isinstance(A, sp.sparse.csr_matrix)
    assert isinstance(c, sp.sparse.csr_matrix)
    assert isinstance(b, sp.sparse.csr_matrix)
    assert A.shape == (1, 50)
    assert b.shape == (1, 1)
    assert c.shape == (1, 1)


def test_dual_basis_element(rng: np.random.Generator) -> None:
    de = DualBasisElement()
    de_2 = DualBasisElement()
    db_0 = de + de_2
    assert isinstance(db_0, DualBasis)
    db_1 = db_0 + db_0
    assert isinstance(db_1, DualBasis)

    dim = 2
    opdm_data = rng.random((dim, dim))
    opdm_data = (opdm_data.T + opdm_data) / 2
    opdm = Tensor(tensor=opdm_data, name='opdm')
    rdm = MultiTensor([opdm])

    def generate_dual_basis_element(i: int, j: int) -> DualBasisElement:
        element = DualBasisElement(
            tensor_names=["opdm"],
            tensor_elements=[(i, j)],
            tensor_coeffs=[-1.0],
            bias=1 if i == j else 0,
            scalar=0,
        )
        return element

    opdm_to_oqdm_map = DualBasis()
    for _, idx in opdm.all_iterator():
        i, j = idx
        opdm_to_oqdm_map += generate_dual_basis_element(i, j)

    rdm.dual_basis = opdm_to_oqdm_map
    A, b, _ = rdm.synthesize_dual_basis()
    Adense = A.todense()
    opdm_flat = opdm_data.reshape((-1, 1))
    oqdm = Adense.dot(opdm_flat)
    test_oqdm = oqdm + b.todense()
    assert np.allclose(test_oqdm.reshape((dim, dim)), np.eye(dim) - opdm_data)


def test_cover_make_offset_dict(rng: np.random.Generator) -> None:
    a = rng.random((5, 5))
    b = rng.random((4, 4))
    c = rng.random((3, 3))
    with pytest.raises(TypeError):
        _ = MultiTensor.make_offset_dict([a, b, c])
