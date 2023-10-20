# coverage:ignore
"""Test cases for pyscf_utils.py
"""
import unittest
from os import path

import numpy as np
import pytest

from openfermion.resource_estimates import HAVE_DEPS_FOR_RESOURCE_ESTIMATES

if HAVE_DEPS_FOR_RESOURCE_ESTIMATES:
    from pyscf import cc, gto, scf

    from openfermion.resource_estimates import df, sf
    from openfermion.resource_estimates.molecule import (
        ccsd_t,
        factorized_ccsd_t,
        load_casfile_to_pyscf,
        open_shell_t1_d1,
        pyscf_to_cas,
        stability,
    )


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
@pytest.mark.slow
class OpenFermionPyscfUtilsTest(unittest.TestCase):
    def test_full_ccsd_t(self):
        """Test resource_estimates full CCSD(T) from h1/eri/ecore tensors
        matches regular PySCF CCSD(T)
        """

        for scf_type in ['rhf', 'rohf']:
            mol = gto.Mole()
            mol.atom = 'H 0 0 0; F 0 0 1.1'
            mol.charge = 0
            if scf_type == 'rhf':
                mol.spin = 0
            elif scf_type == 'rohf':
                mol.spin = 2
            mol.basis = 'ccpvtz'
            mol.symmetry = False
            mol.build()

            if scf_type == 'rhf':
                mf = scf.RHF(mol)
            elif scf_type == 'rohf':
                mf = scf.ROHF(mol)

            mf.init_guess = 'mindo'
            mf.conv_tol = 1e-10
            mf.kernel()
            mf = stability(mf)

            # Do PySCF CCSD(T)
            mycc = cc.CCSD(mf)
            mycc.max_cycle = 500
            mycc.conv_tol = 1e-9
            mycc.conv_tol_normt = 1e-5
            mycc.diis_space = 24
            mycc.diis_start_cycle = 4
            mycc.kernel()
            et = mycc.ccsd_t()

            pyscf_escf = mf.e_tot
            pyscf_ecor = mycc.e_corr + et
            pyscf_etot = pyscf_escf + pyscf_ecor
            pyscf_results = np.array([pyscf_escf, pyscf_ecor, pyscf_etot])

            n_elec = mol.nelectron
            n_orb = mf.mo_coeff[0].shape[-1]

            resource_estimates_results = ccsd_t(*pyscf_to_cas(mf, n_orb, n_elec))
            resource_estimates_results = np.asarray(resource_estimates_results)

            # ignore relative tolerance, we just want absolute tolerance
            assert np.allclose(pyscf_results, resource_estimates_results, rtol=1e-14)

    def test_reduced_ccsd_t(self):
        """Test resource_estimates reduced (2e space) CCSD(T) from tensors
        matches PySCF CAS(2e,No)
        """

        for scf_type in ['rhf', 'rohf']:
            mol = gto.Mole()
            mol.atom = 'H 0 0 0; F 0 0 1.1'
            mol.charge = 0
            if scf_type == 'rhf':
                mol.spin = 0
            elif scf_type == 'rohf':
                mol.spin = 2
            mol.basis = 'ccpvtz'
            mol.symmetry = False
            mol.build()

            if scf_type == 'rhf':
                mf = scf.RHF(mol)
            elif scf_type == 'rohf':
                mf = scf.ROHF(mol)

            mf.init_guess = 'mindo'
            mf.conv_tol = 1e-10
            mf.kernel()
            mf = stability(mf)

            # PySCF CAS(No,2e) for 2 electrons CCSD (and so CCSD(T)) is exact
            n_elec = 2  # electrons
            n_orb = mf.mo_coeff[0].shape[-1] - mf.mol.nelectron - n_elec
            mycas = mf.CASCI(n_orb, n_elec).run()

            pyscf_etot = mycas.e_tot

            # Don't do triples (it's zero anyway for 2e) b/c div by zero w/ ROHF
            _, _, resource_estimates_etot = ccsd_t(
                *pyscf_to_cas(mf, n_orb, n_elec), no_triples=True
            )

            # ignore relative tolerance, we just want absolute tolerance
            assert np.isclose(pyscf_etot, resource_estimates_etot, rtol=1e-14)

    def test_reiher_sf_ccsd_t(self):
        """Reproduce Reiher et al FeMoco SF CCSD(T) errors from paper"""

        NAME = path.join(path.dirname(__file__), '../integrals/eri_reiher.h5')
        _, mf = load_casfile_to_pyscf(NAME, num_alpha=27, num_beta=27)
        _, ecorr, _ = factorized_ccsd_t(mf, eri_rr=None)  # use full (local) ERIs for 2-body
        exact_energy = ecorr
        rank = 100
        eri_rr, _ = sf.factorize(mf._eri, rank)
        _, ecorr, _ = factorized_ccsd_t(mf, eri_rr)
        appx_energy = ecorr

        error = (appx_energy - exact_energy) * 1e3  # mEh

        assert np.isclose(np.round(error, decimals=2), 1.55)

    def test_reiher_df_ccsd_t(self):
        """Reproduce Reiher et al FeMoco DF CCSD(T) errors from paper"""

        NAME = path.join(path.dirname(__file__), '../integrals/eri_reiher.h5')
        _, mf = load_casfile_to_pyscf(NAME, num_alpha=27, num_beta=27)
        _, ecorr, _ = factorized_ccsd_t(mf, eri_rr=None)  # use full (local) ERIs for 2-body
        exact_energy = ecorr
        appx_energy = []
        THRESH = 0.00125
        eri_rr, _, _, _ = df.factorize(mf._eri, thresh=THRESH)
        _, ecorr, _ = factorized_ccsd_t(mf, eri_rr)
        appx_energy = ecorr

        error = (appx_energy - exact_energy) * 1e3  # mEh

        assert np.isclose(np.round(error, decimals=2), 0.44)

    def test_t1_d1_openshell(self):
        """Test open shell t1-diagnostic by reducing back to closed shell"""
        mol = gto.M()
        mol.atom = 'N 0 0 0; N 0 0 1.4'
        mol.basis = 'cc-pvtz'
        mol.spin = 0
        mol.build()

        mf = scf.RHF(mol)
        mf.kernel()

        mycc = cc.CCSD(mf)
        mycc.kernel()

        true_t1d, true_d1d = mycc.get_t1_diagnostic(), mycc.get_d1_diagnostic()

        uhf_mf = scf.convert_to_uhf(mf)
        mycc_uhf = cc.CCSD(uhf_mf)
        mycc_uhf.kernel()
        t1a, t1b = mycc_uhf.t1
        test_t1d, test_d1d = open_shell_t1_d1(
            t1a, t1b, uhf_mf.mo_occ[0] + uhf_mf.mo_occ[1], uhf_mf.nelec[0], uhf_mf.nelec[1]
        )

        assert np.isclose(test_t1d, true_t1d)
        assert np.isclose(test_d1d, true_d1d)
        assert np.sqrt(2) * test_t1d <= test_d1d

    def test_t1_d1_oxygen(self):
        """Test open shell t1-diagnostic on O2 molecule

        Compare with output from Psi4

        * Input:

        molecule oxygen {
          0 3
          O 0.0 0.0 0.0
          O 0.0 0.0 1.1
          no_reorient
          symmetry c1
        }

        set {
          reference rohf
          basis cc-pvtz
        }

        energy('CCSD')

        * Output
        @ROHF Final Energy:  -149.65170765644311

                       Solving CC Amplitude Equations
                        ------------------------------
        Iter     Energy           RMS        T1Diag      D1Diag    New D1Diag
        ----     ----------    ---------   ----------  ----------  ----------
        ...
        10        -0.464506    1.907e-07    0.004390    0.009077    0.009077
        11        -0.464506    5.104e-08    0.004390    0.009077    0.009077

        Iterations converged.
        """

        mol = gto.M()
        mol.atom = 'O 0 0 0; O 0 0 1.1'
        mol.basis = 'cc-pvtz'
        mol.spin = 2
        mol.build()

        mf = scf.ROHF(mol)
        mf.kernel()

        uhf_mf = scf.convert_to_uhf(mf)
        mycc_uhf = cc.CCSD(mf)
        mycc_uhf.kernel()

        t1a, t1b = mycc_uhf.t1
        test_t1d, test_d1d = open_shell_t1_d1(
            t1a, t1b, uhf_mf.mo_occ[0] + uhf_mf.mo_occ[1], uhf_mf.nelec[0], uhf_mf.nelec[1]
        )

        assert np.isclose(mf.e_tot, -149.651708, atol=1e-6)
        assert np.isclose(mycc_uhf.e_corr, -0.464507, atol=1e-6)
        assert np.isclose(test_t1d, 0.004390, atol=1e-4)
        assert np.isclose(test_d1d, 0.009077, atol=1e-4)

    def test_t1_d1_bound(self):
        """sqrt(2) * t1 <= d1"""
        mol = gto.M()
        mol.atom = 'O 0 0 0; O 0 0 1.4'
        mol.basis = 'cc-pvtz'
        mol.spin = 2
        mol.build()
        mf = scf.ROHF(mol)
        mf.kernel()
        mycc = cc.CCSD(mf)
        mycc.kernel()
        uhf_mf = scf.convert_to_uhf(mf)
        mycc_uhf = cc.CCSD(uhf_mf)
        mycc_uhf.kernel()
        t1a, t1b = mycc_uhf.t1
        test_t1d, test_d1d = open_shell_t1_d1(
            t1a, t1b, uhf_mf.mo_occ[0] + uhf_mf.mo_occ[1], uhf_mf.nelec[0], uhf_mf.nelec[1]
        )
        assert np.sqrt(2) * test_t1d <= test_d1d
