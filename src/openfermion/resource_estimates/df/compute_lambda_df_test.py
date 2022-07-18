#coverage:ignore
"""Test cases for compute_lambda_df.py
"""
from os import path
import numpy as np
from openfermion.resource_estimates import df
from openfermion.resource_estimates.molecule import load_casfile_to_pyscf


def test_reiher_df_lambda():
    """ Reproduce Reiher et al orbital DF lambda from paper """

    THRESH = 0.00125
    NAME = path.join(path.dirname(__file__), '../integrals/eri_reiher.h5')
    _, mf = load_casfile_to_pyscf(NAME, num_alpha=27, num_beta=27)
    eri_rr, LR, L, Lxi = df.factorize(mf._eri, thresh=THRESH)
    total_lambda = df.compute_lambda(mf, LR)
    assert eri_rr.shape[0] * 2 == 108
    assert L == 360
    assert Lxi == 13031
    assert np.isclose(np.round(total_lambda, decimals=1), 294.8)
