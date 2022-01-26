#coverage:ignore
"""Test cases for costing_sf.py
"""
from openfermion.resource_estimates import sf


def test_reiher_sf():
    """ Reproduce Reiher et al orbital SF FT costs from paper """
    DE = 0.001
    CHI = 10

    # Reiher et al orbitals
    N = 108
    LAM = 4258.0
    L = 200

    # Here we're using an initial calculation with a very rough estimate of the
    # number of steps to give a more accurate number of steps.
    # Then we input that into the function again.
    output = sf.compute_cost(N, LAM, DE, L, CHI, stps=20000)
    stps1 = output[0]
    output = sf.compute_cost(N, LAM, DE, L, CHI, stps1)
    assert output == (14184, 94868988984, 3320)


def test_li_sf():
    """ Reproduce Li et al orbital SF FT costs from paper """
    DE = 0.001
    CHI = 10

    # Li et al orbitals
    N = 152
    LAM = 3071.8
    L = 275
    # Here we're using an initial calculation with a very rough estimate of the
    # number of steps to give a more accurate number of steps.
    # Then we input that into the function again.
    output = sf.compute_cost(N, LAM, DE, L, CHI, stps=20000)
    stps2 = output[0]
    output = sf.compute_cost(N, LAM, DE, L, CHI, stps2)
    assert output == (24396, 117714920508, 3628)
