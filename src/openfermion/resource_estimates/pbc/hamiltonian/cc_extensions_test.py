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
import os

import pytest
import numpy as np

from openfermion.resource_estimates import HAVE_DEPS_FOR_RESOURCE_ESTIMATES

if HAVE_DEPS_FOR_RESOURCE_ESTIMATES:
    from pyscf.pbc import gto, scf, mp
    from pyscf.pbc.cc import KRCCSD
    from pyscf.lib import chkfile
    from openfermion.resource_estimates.pbc.hamiltonian import cholesky_from_df_ints
    from openfermion.resource_estimates.pbc.hamiltonian.cc_extensions import (
        build_approximate_eris_rohf,
        build_approximate_eris,
    )
    from openfermion.resource_estimates.pbc.sf.sf_integrals import SingleFactorization

_TEST_CHK = os.path.join(os.path.dirname(__file__), "../test_data/scf.chk")


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
@pytest.mark.slow
def test_cc_extension_rohf():
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
    cell.build()

    kmesh = [1, 1, 3]
    kpts = cell.make_kpts(kmesh)
    mf = scf.KROHF(cell, kpts).rs_density_fit()
    mf.with_df.mesh = cell.mesh
    mf.kernel()

    # Only ROHF integrals are supported for resource estimates but only UCCSD
    # available, so convert MOs to UHF format first.
    u_from_ro = scf.addons.convert_to_uhf(mf)
    mymp = mp.KMP2(mf)

    Luv = cholesky_from_df_ints(mymp)  # [kpt, kpt, naux, nao, nao]
    from pyscf.pbc.cc.kccsd_uhf import _make_df_eris
    from pyscf.pbc.cc import KUCCSD

    cc_inst = KUCCSD(u_from_ro)
    eris = _make_df_eris(cc_inst)
    emp2_ref, _, _ = cc_inst.init_amps(eris)
    ref_eris = _make_df_eris(cc_inst)

    helper = SingleFactorization(cholesky_factor=Luv, kmf=mf)
    test_eris = build_approximate_eris_rohf(cc_inst, helper)
    eri_blocks = [
        "OOOO",
        "OOOV",
        "OOVV",
        "OVOV",
        "VOOV",
        "VOVV",
        "oooo",
        "ooov",
        "oovv",
        "ovov",
        "voov",
        "vovv",
        "OOov",
        "OOvv",
        "OVov",
        "VOov",
        "VOvv",
    ]
    for block in eri_blocks:
        assert np.allclose(test_eris.__dict__[block][:], ref_eris.__dict__[block][:])
    # Test MP2 energy is the correct
    emp2_approx, _, _ = cc_inst.init_amps(test_eris)
    assert abs(emp2_approx - emp2_ref) < 1e-12

    # Test MP2 energy is the different when truncated
    helper = SingleFactorization(cholesky_factor=Luv, kmf=mf, naux=10)
    test_eris_approx = build_approximate_eris_rohf(cc_inst, helper)
    for block in eri_blocks:
        assert not np.allclose(test_eris_approx.__dict__[block][:], ref_eris.__dict__[block][:])
    emp2_approx, _, _ = cc_inst.init_amps(test_eris_approx)
    # MP2 energy should be pretty bad
    assert abs(emp2_approx - emp2_ref) > 1e-12
    # Test CCSD
    cc_exact = KUCCSD(u_from_ro)
    ecc_exact, _, _ = cc_exact.kernel()
    cc_approx = KUCCSD(u_from_ro)
    helper = SingleFactorization(cholesky_factor=Luv, kmf=mf)
    test_eris = build_approximate_eris_rohf(cc_approx, helper)
    cc_approx.ao2mo = lambda mo_coeff=None: test_eris
    emp2_approx, _, _ = cc_approx.init_amps(test_eris)
    ecc_approx, _, _ = cc_approx.kernel()
    assert abs(ecc_exact - ecc_approx) < 1e-12
    # Should be different
    helper = SingleFactorization(cholesky_factor=Luv, kmf=mf, naux=10)
    test_eris = build_approximate_eris_rohf(cc_approx, helper)
    for block in eri_blocks:
        assert not np.allclose(test_eris.__dict__[block][:], ref_eris.__dict__[block][:])
    # Want to avoid ccsd helper using density-fitted integrals which will be
    # "exact"
    assert test_eris.Lpv is None
    assert test_eris.LPV is None
    cc_approx = KUCCSD(u_from_ro)
    # overwrite ao2mo object required as function does not check if eris exists.
    cc_approx.ao2mo = lambda mo_coeff=None: test_eris
    ecc_approx, _, _ = cc_approx.kernel()
    assert abs(ecc_exact - ecc_approx) > 1e-12


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
@pytest.mark.slow
def test_cc_helper_rhf():
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
    cell.build()

    kmesh = [1, 1, 3]
    kpts = cell.make_kpts(kmesh)
    mf = scf.KRHF(cell, kpts).rs_density_fit()
    scf_dict = chkfile.load(_TEST_CHK, "scf")
    mf.__dict__.update(scf_dict)
    mf.with_df._cderi = _TEST_CHK
    dm0 = mf.make_rdm1()
    mf.with_df.mesh = cell.mesh
    mf.kernel(dm0=dm0)

    # Only ROHF integrals are supported for resource estimates but only UCCSD
    # available, so convert MOs to UHF format first.
    mymp = mp.KMP2(mf)

    Luv = cholesky_from_df_ints(mymp)  # [kpt, kpt, naux, nao, nao]

    cc_inst = KRCCSD(mf)
    ref_eris = cc_inst.ao2mo()
    emp2_ref, _, _ = cc_inst.init_amps(ref_eris)

    helper = SingleFactorization(cholesky_factor=Luv, kmf=mf)
    test_eris = build_approximate_eris(cc_inst, helper)
    eri_blocks = ["oooo", "ooov", "oovv", "ovov", "voov", "vovv"]
    for block in eri_blocks:
        assert np.allclose(test_eris.__dict__[block][:], ref_eris.__dict__[block][:])
    # Test MP2 energy is the correct
    emp2_approx, _, _ = cc_inst.init_amps(test_eris)
    assert abs(emp2_approx - emp2_ref) < 1e-12

    # Test MP2 energy is the different when truncated
    helper = SingleFactorization(cholesky_factor=Luv, kmf=mf, naux=10)
    test_eris_approx = build_approximate_eris(cc_inst, helper)
    for block in eri_blocks:
        assert not np.allclose(test_eris_approx.__dict__[block][:], ref_eris.__dict__[block][:])
    emp2_approx, _, _ = cc_inst.init_amps(test_eris_approx)
    # MP2 energy should be pretty bad
    assert abs(emp2_approx - emp2_ref) > 1e-12
    # Test CCSD
    cc_exact = KRCCSD(mf)
    ecc_exact, _, _ = cc_exact.kernel()
    cc_approx = KRCCSD(mf)
    helper = SingleFactorization(cholesky_factor=Luv, kmf=mf)
    test_eris = build_approximate_eris(cc_approx, helper)
    cc_approx.ao2mo = lambda mo_coeff=None: test_eris
    emp2_approx, _, _ = cc_approx.init_amps(test_eris)
    ecc_approx, _, _ = cc_approx.kernel()
    assert abs(ecc_exact - ecc_approx) < 1e-12
    # Should be different
    helper = SingleFactorization(cholesky_factor=Luv, kmf=mf, naux=10)
    test_eris = build_approximate_eris(cc_approx, helper)
    for block in eri_blocks:
        assert not np.allclose(test_eris.__dict__[block][:], ref_eris.__dict__[block][:])
    # Want to avoid ccsd helper using density-fitted integrals which will be
    # "exact"
    cc_approx = KRCCSD(mf)
    # overwrite ao2mo object required as function does not check if eris exists.
    cc_approx.ao2mo = lambda mo_coeff=None: test_eris
    ecc_approx, _, _ = cc_approx.kernel()
    assert abs(ecc_exact - ecc_approx) > 1e-12
