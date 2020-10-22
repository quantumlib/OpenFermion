import pytest
import numpy as np
from openfermion.third_party.representability._dualbasis import \
    DualBasisElement, DualBasis


def test_dualbasis_element_init():
    dbe = DualBasisElement()
    assert isinstance(dbe, DualBasisElement)

    dbe = DualBasisElement(tensor_names=['A'] * 5,
                           tensor_coeffs=[1] * 5,
                           tensor_elements=[(i, i) for i in range(5)])

    assert dbe.primal_tensors_names == ['A'] * 5
    assert dbe.primal_coeffs == [1] * 5
    assert dbe.primal_elements == [(i, i) for i in range(5)]
    assert dbe.constant_bias == 0
    assert dbe.dual_scalar == 0


def test_daulbasis_init():
    db = DualBasis()
    assert isinstance(db, DualBasis)
    dbe = DualBasisElement(tensor_names=['A'] * 5,
                           tensor_coeffs=[1] * 5,
                           tensor_elements=[(i, i) for i in range(5)])
    db = DualBasis(elements=[dbe])
    assert db[0] == dbe
    assert len(db) == 1
    for tdbe in db:
        assert tdbe == dbe

    db2 = db + dbe
    assert isinstance(db2, DualBasis)
    assert len(db2) == 2

    tdbe = DualBasisElement(tensor_names=['B'] * 5,
                            tensor_coeffs=[1] * 5,
                            tensor_elements=[(i, i) for i in range(5)])
    tdb = DualBasis(elements=[tdbe])
    db3 = db + tdb
    assert isinstance(db3, DualBasis)
    assert len(db3) == 2

    db4 = tdbe + db
    assert len(db4) == 2

    with pytest.raises(TypeError):
        _ = DualBasis(elements=[tdbe, 4])

    with pytest.raises(TypeError):
        db = DualBasis(elements=[tdbe])
        _ = db + 4


def test_dbe_element_add():
    de = DualBasisElement()
    de_2 = DualBasisElement()
    db_0 = de + de_2
    assert isinstance(db_0, DualBasis)
    db_0 = de_2 + de
    assert isinstance(db_0, DualBasis)
    db_1 = db_0 + db_0
    assert isinstance(db_1, DualBasis)

    with pytest.raises(TypeError):
        _ = de + 2


def test_dbe_string():
    dbe = DualBasisElement(tensor_names=['A'] * 5,
                           tensor_coeffs=[1] * 5,
                           tensor_elements=[(i, i) for i in range(5)])

    assert dbe.id() == "A(0,0)	A(1,1)	A(2,2)	A(3,3)	A(4,4)\t"


def test_dbe_iterator():
    dbe = DualBasisElement(tensor_names=['A'] * 5,
                           tensor_coeffs=[1] * 5,
                           tensor_elements=[(i, i) for i in range(5)])
    for idx, (t, v, c) in enumerate(dbe):
        assert t == 'A'
        assert v == (idx, idx)
        assert c == 1


def test_add_element():
    with pytest.raises(TypeError):
        dbe = DualBasisElement()
        dbe.add_element('cckk', [1, 2, 3, 4], 0.5)

    with pytest.raises(TypeError):
        dbe = DualBasisElement()
        dbe.add_element(5, (1, 2, 3, 4), 0.5)

    with pytest.raises(TypeError):
        dbe = DualBasisElement()
        dbe.add_element('a', (1, 2, 3, 4), 'a')

    dbe = DualBasisElement()
    dbe.add_element('cckk', (1, 2, 3, 4), 0.5)
    dbe.constant_bias = 0.33

    assert dbe.primal_coeffs == [0.5]
    assert dbe.primal_tensors_names == ['cckk']
    assert dbe.primal_elements == [(1, 2, 3, 4)]

    dbe.add_element('ck', (0, 1), 1)
    assert dbe.primal_coeffs == [0.5, 1]
    assert dbe.primal_tensors_names == ['cckk', 'ck']
    assert dbe.primal_elements == [(1, 2, 3, 4), (0, 1)]

    dbe2 = DualBasisElement()
    with pytest.raises(TypeError):
        dbe2.join_elements((1, 1))

    dbe2.constant_bias = 0.25
    dbe2.add_element('cckk', (1, 2, 3, 4), 0.5)
    dbe3 = dbe2.join_elements(dbe)
    assert np.isclose(dbe3.constant_bias, 0.58)

    assert set(dbe3.primal_elements) == {(0, 1), (1, 2, 3, 4)}
    assert np.allclose(dbe3.primal_coeffs, [1, 1])
    assert set(dbe3.primal_tensors_names) == {'ck', 'cckk'}


def test_simplify():
    i, j, k, l = 0, 1, 2, 3
    names = ['opdm'] * 3 + ['oqdm']
    elements = [(i, j), (i, j), (i, l), (l, k)]
    coeffs = [1, 1, 0.25, 0.3]
    dbe = DualBasisElement(tensor_names=names,
                           tensor_elements=elements,
                           tensor_coeffs=coeffs)
    dbe.simplify()
    assert len(dbe.primal_tensors_names) == 3
    assert set(dbe.primal_coeffs) == {2, 0.25, 0.3}
    assert set(dbe.primal_tensors_names) == {'opdm', 'oqdm'}
    assert set(dbe.primal_elements) == {(0, 1), (0, 3), (3, 2)}
