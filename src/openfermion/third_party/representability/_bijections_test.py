from itertools import product
from openfermion.third_party.representability._bijections import Bijection, \
    index_index_basis, index_tuple_basis


def test_bijection():
    """
    Testing the basis bijection
    """
    b = Bijection(lambda x: x + 1, lambda y: y - 1, lambda: (1, 1))
    assert isinstance(b, Bijection)
    assert b.fwd(2) == 3
    assert b.rev(2) == 1
    assert b.fwd(b.rev(5)) == 5
    assert b.domain_element_sizes() == (1, 1)


def test_index_basis():
    b = index_index_basis(5)
    assert b.fwd(4) == 4
    assert b.rev(4) == 4
    assert b.domain_element_sizes() == (1, 1)


def test_geminal_basis():
    gems = list(product(range(5), repeat=2))
    b = index_tuple_basis(gems)
    assert b.fwd(4) == (0, 4)
    assert b.rev((0, 4)) == 4
    assert b.rev(b.fwd(4)) == 4
    assert b.domain_element_sizes() == (1, 2)