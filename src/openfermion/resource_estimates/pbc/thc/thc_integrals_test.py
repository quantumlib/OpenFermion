# coverage: ignore
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
import numpy as np
import pytest

from openfermion.resource_estimates import HAVE_DEPS_FOR_RESOURCE_ESTIMATES

if HAVE_DEPS_FOR_RESOURCE_ESTIMATES:
    from pyscf.pbc import gto, scf, cc

    from openfermion.resource_estimates.pbc.thc.factorizations.isdf import solve_kmeans_kpisdf
    from openfermion.resource_estimates.pbc.thc.thc_integrals import (
        KPTHCDoubleTranslation,
        KPTHCSingleTranslation,
    )
    from openfermion.resource_estimates.pbc.hamiltonian.cc_extensions import build_approximate_eris


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
@pytest.mark.slow
def test_thc_helper():
    cell = gto.Cell()
    cell.atom = """
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391
    """
    cell.basis = "gth-szv"
    cell.pseudo = "gth-hf-rev"
    cell.a = """
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000"""
    cell.unit = "B"
    cell.verbose = 0
    cell.mesh = [11] * 3
    cell.build()

    kmesh = [1, 1, 3]
    kpts = cell.make_kpts(kmesh)
    mf = scf.KRHF(cell, kpts)
    mf.kernel()

    exact_cc = cc.KRCCSD(mf)
    eris = exact_cc.ao2mo()
    exact_emp2, _, _ = exact_cc.init_amps(eris)

    approx_cc = cc.KRCCSD(mf)
    approx_cc.verbose = 0
    kpt_thc = solve_kmeans_kpisdf(
        mf,
        np.prod(cell.mesh),  # Use the whole grid to avoid any precision issues
        single_translation=False,
        use_density_guess=True,
        verbose=False,
    )

    helper = KPTHCDoubleTranslation(kpt_thc.chi, kpt_thc.zeta, mf)

    num_kpts = len(mf.kpts)
    for iq in range(num_kpts):
        for ik in range(num_kpts):
            ik_minus_q = helper.k_transfer_map[iq, ik]
            for ik_prime in range(num_kpts):
                ik_prime_minus_q = helper.k_transfer_map[iq, ik_prime]
                eri_thc = helper.get_eri([ik, ik_minus_q, ik_prime_minus_q, ik_prime])
                eri_exact = helper.get_eri_exact([ik, ik_minus_q, ik_prime_minus_q, ik_prime])
                assert np.allclose(eri_thc, eri_exact)

    eris_approx = build_approximate_eris(approx_cc, helper)
    emp2, _, _ = approx_cc.init_amps(eris_approx)
    assert np.isclose(emp2, exact_emp2)
    kpt_thc = solve_kmeans_kpisdf(mf, np.prod(cell.mesh), single_translation=True, verbose=False)
    helper = KPTHCSingleTranslation(kpt_thc.chi, kpt_thc.zeta, mf)
    for iq in range(num_kpts):
        for ik in range(num_kpts):
            ik_minus_q = helper.k_transfer_map[iq, ik]
            for ik_prime in range(num_kpts):
                ik_prime_minus_q = helper.k_transfer_map[iq, ik_prime]
                eri_thc = helper.get_eri([ik, ik_minus_q, ik_prime_minus_q, ik_prime])
                eri_exact = helper.get_eri_exact([ik, ik_minus_q, ik_prime_minus_q, ik_prime])
                assert np.allclose(eri_thc, eri_exact)

    eris_approx = build_approximate_eris(approx_cc, helper)
    emp2, _, _ = approx_cc.init_amps(eris_approx)
    assert np.isclose(emp2, exact_emp2)
