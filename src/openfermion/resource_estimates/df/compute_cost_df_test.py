# coverage:ignore
"""Test cases for costing_df.py
"""
import pytest

from openfermion.resource_estimates import HAVE_DEPS_FOR_RESOURCE_ESTIMATES

if HAVE_DEPS_FOR_RESOURCE_ESTIMATES:
    from openfermion.resource_estimates import df


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
def test_reiher_df():
    """Reproduce Reiher et al orbital DF FT costs from paper"""
    DE = 0.001
    CHI = 10

    # Reiher et al orbitals
    N = 108
    LAM = 294.8
    L = 360
    LXI = 13031
    BETA = 16

    # Here we're using an initial calculation with a very rough estimate of the
    # number of steps to give a more accurate number of steps.
    # Then we input that into the function again.
    output = df.compute_cost(N, LAM, DE, L, LXI, CHI, BETA, stps=20000)
    stps1 = output[0]
    output = df.compute_cost(N, LAM, DE, L, LXI, CHI, BETA, stps1)
    assert output == (21753, 10073183463, 3725)


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
def test_li_df():
    """Reproduce Li et al orbital DF FT costs from paper"""
    DE = 0.001
    CHI = 10

    # Li et al orbitals
    N = 152
    LAM = 1171.2
    L = 394
    LXI = 20115
    BETA = 20

    # Here we're using an initial calculation with a very rough estimate of the
    # number of steps to give a more accurate number of steps.
    # Then we input that into the function again.
    output = df.compute_cost(N, LAM, DE, L, LXI, CHI, BETA, stps=20000)
    stps2 = output[0]
    output = df.compute_cost(N, LAM, DE, L, LXI, CHI, BETA, stps2)
    assert output == (35008, 64404812736, 6404)
