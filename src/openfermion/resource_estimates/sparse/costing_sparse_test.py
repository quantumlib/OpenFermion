#coverage:ignore
"""Test cases for costing_sparse.py
"""
from openfermion.resource_estimates.sparse.costing_sparse import cost_sparse


def test_reiher_sparse():
    """ Reproduce Reiher et al orbital sparse FT costs from paper """
    DE = 0.001
    CHI = 10

    # Reiher et al orbitals
    N = 108
    LAM = 2135.3
    D = 705831

    # Here we're using an initial calculation with a very rough estimate of the
    # number of steps to give a more accurate number of steps. Then we input
    # that into the function again.
    output = cost_sparse(N, LAM, D, DE, CHI, stps=20000)
    stps1 = output[0]
    output = cost_sparse(N, LAM, D, DE, CHI, stps1)
    assert output == (26347, 88371052334, 2190)


def test_li_sparse():
    """ Reproduce Li et al orbital sparse FT costs from paper """
    DE = 0.001
    CHI = 10

    # Li et al orbitals
    N = 152
    LAM = 1547.3
    D = 440501

    # Here we're using an initial calculation with a very rough estimate of the
    # number of steps to give a more accurate number of steps. Then we input
    # that into the function again.
    output = cost_sparse(N, LAM, D, DE, CHI, stps=20000)
    stps2 = output[0]
    output = cost_sparse(N, LAM, D, DE, CHI, stps2)
    assert output == (18143, 44096452642, 2489)
