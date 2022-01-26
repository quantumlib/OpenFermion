#coverage:ignore
"""Test cases for compute_lambda_sf.py
"""
from os import path
import numpy as np
from openfermion.resource_estimates import sf
from openfermion.resource_estimates.molecule import load_casfile_to_pyscf


def test_reiher_sf_lambda():
    """ Reproduce Reiher et al orbital SF lambda from paper """

    RANK = 200

    NAME = path.join(path.dirname(__file__), '../integrals/eri_reiher.h5')
    _, reiher_mf = load_casfile_to_pyscf(NAME, num_alpha=27, num_beta=27)
    eri_rr, sf_factors = sf.factorize(reiher_mf._eri, RANK)
    lambda_tot = sf.compute_lambda(reiher_mf, sf_factors)
    assert eri_rr.shape[0] * 2 == 108
    assert np.isclose(lambda_tot, 4258.0)
