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
"""Methods for overwriting coupled cluster single and doubles (CCSD) eris."""

from pyscf.lib import logger
from pyscf.pbc.lib.kpts_helper import loop_kkk
from pyscf.pbc.cc.kccsd_uhf import _make_eris_incore
from pyscf.pbc.cc.kccsd_rhf import _ERIS
from pyscf.pbc import cc, scf


def build_cc_inst(pyscf_mf):
    """Build PBC CC instance.

    If ROHF build KUCCSD object

    Args:
        pyscf_mf: pyscf mean field object (RHF or ROHF).

    Returns:
        cc_inst: Coupled cluster instance (RHF->KRCCSD, ROHF->KUCCSD).
    """
    if pyscf_mf.cell.spin == 0:
        cc_inst = cc.KRCCSD(pyscf_mf)
    else:
        u_from_ro = scf.addons.convert_to_uhf(pyscf_mf)
        cc_inst = cc.KUCCSD(u_from_ro)
    return cc_inst


def build_approximate_eris(krcc_inst, eri_helper, eris=None):
    """Update coupled cluster eris object with approximate integrals.

    Arguments:
        cc: pyscf PBC KRCCSD object.
        eri_helper: Approximate ERIs helper function which defines MO integrals.
        eris: pyscf _ERIS object. Optional, if present overwrite this eris
            object rather than build from scratch.

    Returns:
        eris: pyscf _ERIS object updated to hold approximate eris
            defined by eri_helper.
    """
    log = logger.Logger(krcc_inst.stdout, krcc_inst.verbose)
    kconserv = krcc_inst.khelper.kconserv
    khelper = krcc_inst.khelper
    nocc = krcc_inst.nocc
    nkpts = krcc_inst.nkpts
    dtype = krcc_inst.mo_coeff[0].dtype
    if eris is not None:
        log.info("Modifying coupled cluster _ERIS object inplace using " f"{eri_helper.__class__}.")
        out_eris = eris
    else:
        log.info(f"Rebuilding coupled cluster _ERIS object using " " {eri_helper.__class__}.")
        out_eris = _ERIS(krcc_inst)
    for ikp, ikq, ikr in khelper.symm_map.keys():
        iks = kconserv[ikp, ikq, ikr]
        kpts = [ikp, ikq, ikr, iks]
        eri_kpt = eri_helper.get_eri(kpts) / nkpts
        if dtype == float:
            eri_kpt = eri_kpt.real
        eri_kpt = eri_kpt
        for kp, kq, kr in khelper.symm_map[(ikp, ikq, ikr)]:
            eri_kpt_symm = khelper.transform_symm(eri_kpt, kp, kq, kr).transpose(0, 2, 1, 3)
            out_eris.oooo[kp, kr, kq] = eri_kpt_symm[:nocc, :nocc, :nocc, :nocc]
            out_eris.ooov[kp, kr, kq] = eri_kpt_symm[:nocc, :nocc, :nocc, nocc:]
            out_eris.oovv[kp, kr, kq] = eri_kpt_symm[:nocc, :nocc, nocc:, nocc:]
            out_eris.ovov[kp, kr, kq] = eri_kpt_symm[:nocc, nocc:, :nocc, nocc:]
            out_eris.voov[kp, kr, kq] = eri_kpt_symm[nocc:, :nocc, :nocc, nocc:]
            out_eris.vovv[kp, kr, kq] = eri_kpt_symm[nocc:, :nocc, nocc:, nocc:]
            out_eris.vvvv[kp, kr, kq] = eri_kpt_symm[nocc:, nocc:, nocc:, nocc:]
    return out_eris


def build_approximate_eris_rohf(kucc_inst, eri_helper, eris=None):
    """Update unrestricted coupled cluster eris object with approximate ERIs.

    KROCCSD is run through KUCCSD object, but we expect (and build) RO
    integrals only.

    Arguments:
        kucc_inst: pyscf PBC KUCCSD object. Only ROHF integrals are supported.
        eri_helper: Approximate ERIs helper function which defines MO integrals.
        eris: pyscf _ERIS object. Optional, if present overwrite this eris
            object rather than build from scratch.

    Returns:
        eris: pyscf _ERIS object updated to hold approximate eris defined by
            eri_helper.
    """
    log = logger.Logger(kucc_inst.stdout, kucc_inst.verbose)
    kconserv = kucc_inst.khelper.kconserv
    nocca, noccb = kucc_inst.nocc
    nkpts = kucc_inst.nkpts
    if eris is not None:
        log.info("Modifying coupled cluster _ERIS object inplace using " f"{eri_helper.__class__}.")
        out_eris = eris
    else:
        log.info("Rebuilding coupled cluster _ERIS object using " f"{eri_helper.__class__}.")
        out_eris = _make_eris_incore(kucc_inst)
    for kp, kq, kr in loop_kkk(nkpts):
        ks = kconserv[kp, kq, kr]
        kpts = [kp, kq, kr, ks]
        tmp = eri_helper.get_eri(kpts) / nkpts
        out_eris.oooo[kp, kq, kr] = tmp[:nocca, :nocca, :nocca, :nocca]
        out_eris.ooov[kp, kq, kr] = tmp[:nocca, :nocca, :nocca, nocca:]
        out_eris.oovv[kp, kq, kr] = tmp[:nocca, :nocca, nocca:, nocca:]
        out_eris.ovov[kp, kq, kr] = tmp[:nocca, nocca:, :nocca, nocca:]
        out_eris.voov[kq, kp, ks] = tmp[:nocca, nocca:, nocca:, :nocca].conj().transpose(1, 0, 3, 2)
        out_eris.vovv[kq, kp, ks] = tmp[:nocca, nocca:, nocca:, nocca:].conj().transpose(1, 0, 3, 2)

    for kp, kq, kr in loop_kkk(nkpts):
        ks = kconserv[kp, kq, kr]
        kpts = [kp, kq, kr, ks]
        tmp = eri_helper.get_eri(kpts) / nkpts
        out_eris.OOOO[kp, kq, kr] = tmp[:noccb, :noccb, :noccb, :noccb]
        out_eris.OOOV[kp, kq, kr] = tmp[:noccb, :noccb, :noccb, noccb:]
        out_eris.OOVV[kp, kq, kr] = tmp[:noccb, :noccb, noccb:, noccb:]
        out_eris.OVOV[kp, kq, kr] = tmp[:noccb, noccb:, :noccb, noccb:]
        out_eris.VOOV[kq, kp, ks] = tmp[:noccb, noccb:, noccb:, :noccb].conj().transpose(1, 0, 3, 2)
        out_eris.VOVV[kq, kp, ks] = tmp[:noccb, noccb:, noccb:, noccb:].conj().transpose(1, 0, 3, 2)

    for kp, kq, kr in loop_kkk(nkpts):
        ks = kconserv[kp, kq, kr]
        kpts = [kp, kq, kr, ks]
        tmp = eri_helper.get_eri(kpts) / nkpts
        out_eris.ooOO[kp, kq, kr] = tmp[:nocca, :nocca, :noccb, :noccb]
        out_eris.ooOV[kp, kq, kr] = tmp[:nocca, :nocca, :noccb, noccb:]
        out_eris.ooVV[kp, kq, kr] = tmp[:nocca, :nocca, noccb:, noccb:]
        out_eris.ovOV[kp, kq, kr] = tmp[:nocca, nocca:, :noccb, noccb:]
        out_eris.voOV[kq, kp, ks] = tmp[:nocca, nocca:, noccb:, :noccb].conj().transpose(1, 0, 3, 2)
        out_eris.voVV[kq, kp, ks] = tmp[:nocca, nocca:, noccb:, noccb:].conj().transpose(1, 0, 3, 2)

    for kp, kq, kr in loop_kkk(nkpts):
        ks = kconserv[kp, kq, kr]
        kpts = [kp, kq, kr, ks]
        tmp = eri_helper.get_eri(kpts) / nkpts
        # out_eris.OOoo[kp,kq,kr] = tmp[:noccb,:noccb,:nocca,:nocca]
        out_eris.OOov[kp, kq, kr] = tmp[:noccb, :noccb, :nocca, nocca:]
        out_eris.OOvv[kp, kq, kr] = tmp[:noccb, :noccb, nocca:, nocca:]
        out_eris.OVov[kp, kq, kr] = tmp[:noccb, noccb:, :nocca, nocca:]
        out_eris.VOov[kq, kp, ks] = tmp[:noccb, noccb:, nocca:, :nocca].conj().transpose(1, 0, 3, 2)
        out_eris.VOvv[kq, kp, ks] = tmp[:noccb, noccb:, nocca:, nocca:].conj().transpose(1, 0, 3, 2)
    # Force CCSD to use eri tensors.
    out_eris.Lpv = None
    out_eris.LPV = None

    return out_eris


def compute_emp2_approx(mf, intgl_helper) -> float:
    """Compute MP2 energy given an integral helper

    Args:
        mf: pyscf MF object (RHF or ROHF).
        ingl_helper: Integral helper (sparse, SF, DF, or THC)

    Returns:
        emp: MP2 total energy.
    """
    cc_inst = build_cc_inst(mf)
    approx_eris = build_approximate_eris(cc_inst, intgl_helper)
    emp2_approx, _, _ = cc_inst.init_amps(approx_eris)
    emp2_approx += mf.e_tot
    return emp2_approx
