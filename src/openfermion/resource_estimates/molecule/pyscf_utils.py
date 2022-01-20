#coverage:ignore
""" Drivers for various PySCF electronic structure routines """
from typing import Tuple, Optional
import sys
import h5py
import numpy as np
from pyscf import gto, scf, ao2mo, mcscf, lo, tools, cc
from pyscf.mcscf import avas


def stability(pyscf_mf):
    """
    Test wave function stability and re-optimize SCF.

    Args:
        pyscf_mf: PySCF mean field object (e.g. `scf.RHF()`)

    Returns:
        pyscf_mf: Updated PySCF mean field object
    """
    new_orbitals = pyscf_mf.stability()[0]
    new_1rdm = pyscf_mf.make_rdm1(new_orbitals, pyscf_mf.mo_occ)
    pyscf_mf = pyscf_mf.run(new_1rdm)

    return pyscf_mf


def localize(pyscf_mf, loc_type='pm', verbose=0):
    """ Localize orbitals given a PySCF mean-field object

    Args:
        pyscf_mf:  PySCF mean field object
        loc_type (str): localization type;
            Pipek-Mezey ('pm') or Edmiston-Rudenberg ('er')
        verbose (int): print level during localization

    Returns:
        pyscf_mf:  Updated PySCF mean field object with localized orbitals
    """
    # Note: After loading with `load_casfile_to_pyscf()` you can quiet message
    # by resetting mf.mol, i.e., mf.mol = gto.M(...)
    # but this assumes you have the *exact* molecular specification on hand.
    # I've gotten acceptable results by restoring mf.mol this way (usually
    # followed by calling mf.kernel()). But consistent localization is not a
    # given (not unique) despite restoring data this way, hence the message.
    if len(pyscf_mf.mol.atom) == 0:
        sys.exit("`localize()` requires atom loc. and atomic basis to be" + \
                 " defined.\n  " + \
                 "It also can be sensitive to the initial guess and MO" + \
                 " coefficients.\n  " + \
                 "Best to try re-creating the PySCF molecule and doing the" + \
                 " SCF, rather than\n  " + \
                 "try to load the mean-field object with" + \
                 " `load_casfile_to_pyscf()`. You can \n " + \
                 "try to provide the missing information, but consistency" + \
                 " cannot be guaranteed!")

    # Split-localize (localize DOCC, SOCC, and virtual separately)
    docc_idx = np.where(np.isclose(pyscf_mf.mo_occ, 2.))[0]
    socc_idx = np.where(np.isclose(pyscf_mf.mo_occ, 1.))[0]
    virt_idx = np.where(np.isclose(pyscf_mf.mo_occ, 0.))[0]

    # Pipek-Mezey
    if loc_type.lower() == 'pm':
        print("Localizing doubly occupied ... ", end="")
        loc_docc_mo = lo.PM(
            pyscf_mf.mol,
            pyscf_mf.mo_coeff[:, docc_idx]).kernel(verbose=verbose)
        print("singly occupied ... ", end="")
        loc_socc_mo = lo.PM(
            pyscf_mf.mol,
            pyscf_mf.mo_coeff[:, socc_idx]).kernel(verbose=verbose)
        print("virtual ... ", end="")
        loc_virt_mo = lo.PM(
            pyscf_mf.mol,
            pyscf_mf.mo_coeff[:, virt_idx]).kernel(verbose=verbose)
        print("DONE")

    # Edmiston-Rudenberg
    elif loc_type.lower() == 'er':
        print("Localizing doubly occupied ... ", end="")
        loc_docc_mo = lo.ER(
            pyscf_mf.mol,
            pyscf_mf.mo_coeff[:, docc_idx]).kernel(verbose=verbose)
        print("singly occupied ... ", end="")
        loc_socc_mo = lo.ER(
            pyscf_mf.mol,
            pyscf_mf.mo_coeff[:, socc_idx]).kernel(verbose=verbose)
        print("virtual ... ", end="")
        loc_virt_mo = lo.ER(
            pyscf_mf.mol,
            pyscf_mf.mo_coeff[:, virt_idx]).kernel(verbose=verbose)
        print("DONE")

    # overwrite orbitals with localized orbitals
    pyscf_mf.mo_coeff[:, docc_idx] = loc_docc_mo.copy()
    pyscf_mf.mo_coeff[:, socc_idx] = loc_socc_mo.copy()
    pyscf_mf.mo_coeff[:, virt_idx] = loc_virt_mo.copy()

    return pyscf_mf


def avas_active_space(pyscf_mf,
                      ao_list=None,
                      molden_fname='avas_localized_orbitals',
                      **kwargs):
    """ Return AVAS active space as PySCF molecule and mean-field object

    Args:
        pyscf_mf:  PySCF mean field object

    Kwargs:
        ao_list: list of strings of AOs (print mol.ao_labels() to see options)
                 Example: ao_list = ['H 1s', 'O 2p', 'O 2s'] for water
        verbose (bool): do additional print
        molden_fname (str): MOLDEN filename to save AVAS active space orbitals.
                            Default is to save
                            to 'avas_localized_orbitals.molden'
        **kwargs: other keyworded arguments to pass into avas.avas()

    Returns:
        pyscf_active_space_mol: Updated PySCF molecule object from
                                AVAS-selected active space
        pyscf_active_space_mf:  Updated PySCF mean field object from
                                AVAS-selected active space
    """

    # Note: requires openshell_option = 3 for this to work, which keeps all
    #     singly occupied in CAS
    # we also require canonicalize = False so that we don't destroy local orbs
    avas_output = avas.avas(pyscf_mf,
                            ao_list,
                            canonicalize=False,
                            openshell_option=3,
                            **kwargs)
    active_norb, active_ne, reordered_orbitals = avas_output

    active_alpha, _ = get_num_active_alpha_beta(pyscf_mf, active_ne)

    if molden_fname is not None:
        # save set of localized orbitals for active space
        if isinstance(pyscf_mf, scf.rohf.ROHF):
            frozen_alpha = pyscf_mf.nelec[0] - active_alpha
            assert frozen_alpha >= 0
        else:
            frozen_alpha = pyscf_mf.mol.nelectron // 2 - active_alpha
            assert frozen_alpha >= 0

        active_space_idx = slice(frozen_alpha, frozen_alpha + active_norb)
        active_mos = reordered_orbitals[:, active_space_idx]
        tools.molden.from_mo(pyscf_mf.mol,
                             molden_fname + '.molden',
                             mo_coeff=active_mos)

    # Choosing an active space changes the molecule ("freezing" electrons,
    # for example), so we
    # form the active space tensors first, then re-form the PySCF objects to
    # ensure consistency
    pyscf_active_space_mol, pyscf_active_space_mf = cas_to_pyscf(
        *pyscf_to_cas(pyscf_mf,
                      cas_orbitals=active_norb,
                      cas_electrons=active_ne,
                      avas_orbs=reordered_orbitals))

    return pyscf_active_space_mol, pyscf_active_space_mf


def cas_to_pyscf(h1, eri, ecore, num_alpha, num_beta):
    """ Return a PySCF molecule and mean-field object from pre-computed CAS Ham

    Args:
        h1 (ndarray) - 2D matrix containing one-body terms (MO basis)
        eri (ndarray) - 4D tensor containing two-body terms (MO basis)
        ecore (float) - frozen core electronic energy + nuclear repulsion energy
        num_alpha (int) - number of spin up electrons in CAS space
        num_beta (int) - number of spin down electrons in CAS space

    Returns:
        pyscf_mol: PySCF molecule object
        pyscf_mf:  PySCF mean field object
    """

    n_orb = len(h1)  # number orbitals
    assert [n_orb] * 4 == [*eri.shape]  # check dims are consistent

    pyscf_mol = gto.M()
    pyscf_mol.nelectron = num_alpha + num_beta
    n_orb = h1.shape[0]
    alpha_diag = [1] * num_alpha + [0] * (n_orb - num_alpha)
    beta_diag = [1] * num_beta + [0] * (n_orb - num_beta)

    # Assumes Hamiltonian is either RHF or ROHF ... should be OK since UHF will
    # have two h1s, etc.
    if num_alpha == num_beta:
        pyscf_mf = scf.RHF(pyscf_mol)
        scf_energy = ecore + \
                     2*np.einsum('ii',  h1[:num_alpha,:num_alpha]) + \
                     2*np.einsum('iijj',
                           eri[:num_alpha,:num_alpha,:num_alpha,:num_alpha]) - \
                       np.einsum('ijji',
                           eri[:num_alpha,:num_alpha,:num_alpha,:num_alpha])

    else:
        pyscf_mf = scf.ROHF(pyscf_mol)
        pyscf_mf.nelec = (num_alpha, num_beta)
        # grab singly and doubly occupied orbitals (assume high-spin open shell)
        docc = slice(None, min(num_alpha, num_beta))
        socc = slice(min(num_alpha, num_beta), max(num_alpha, num_beta))
        scf_energy = ecore + \
                     2.0*np.einsum('ii',h1[docc, docc]) + \
                         np.einsum('ii',h1[socc, socc]) + \
                     2.0*np.einsum('iijj',eri[docc, docc, docc, docc]) - \
                         np.einsum('ijji',eri[docc, docc, docc, docc]) + \
                         np.einsum('iijj',eri[socc, socc, docc, docc]) - \
                     0.5*np.einsum('ijji',eri[socc, docc, docc, socc]) + \
                         np.einsum('iijj',eri[docc, docc, socc, socc]) - \
                     0.5*np.einsum('ijji',eri[docc, socc, socc, docc]) + \
                     0.5*np.einsum('iijj',eri[socc, socc, socc, socc]) - \
                     0.5*np.einsum('ijji',eri[socc, socc, socc, socc])

    pyscf_mf.get_hcore = lambda *args: np.asarray(h1)
    pyscf_mf.get_ovlp = lambda *args: np.eye(h1.shape[0])
    pyscf_mf.energy_nuc = lambda *args: ecore
    pyscf_mf._eri = eri  # ao2mo.restore('8', np.zeros((8, 8, 8, 8)), 8)
    pyscf_mf.e_tot = scf_energy

    pyscf_mf.init_guess = '1e'
    pyscf_mf.mo_coeff = np.eye(n_orb)
    pyscf_mf.mo_occ = np.array(alpha_diag) + np.array(beta_diag)
    pyscf_mf.mo_energy, _ = np.linalg.eigh(pyscf_mf.get_fock())

    return pyscf_mol, pyscf_mf


def pyscf_to_cas(pyscf_mf,
                 cas_orbitals: Optional[int] = None,
                 cas_electrons: Optional[int] = None,
                 avas_orbs=None):
    """ Return CAS Hamiltonian tensors from a PySCF mean-field object

    Args:
        pyscf_mf: PySCF mean field object
        cas_orbitals (int, optional):  number of orbitals in CAS space,
                                       default all orbitals
        cas_electrons (int, optional): number of electrons in CAS space,
                                       default all electrons
        avas_orbs (ndarray, optional): orbitals selected by AVAS in PySCF

    Returns:
        h1 (ndarray) - 2D matrix containing one-body terms (MO basis)
        eri (ndarray) - 4D tensor containing two-body terms (MO basis)
        ecore (float) - frozen core electronic energy + nuclear repulsion energy
        num_alpha (int) - number of spin up electrons in CAS space
        num_beta (int) - number of spin down electrons in CAS space
    """

    # Only RHF or ROHF possible with mcscf.CASCI
    assert isinstance(pyscf_mf, scf.rhf.RHF)  # ROHF is child of RHF class

    if cas_orbitals is None:
        cas_orbitals = len(pyscf_mf.mo_coeff)
    if cas_electrons is None:
        cas_electrons = pyscf_mf.mol.nelectron

    cas = mcscf.CASCI(pyscf_mf, ncas=cas_orbitals, nelecas=cas_electrons)
    h1, ecore = cas.get_h1eff(mo_coeff=avas_orbs)
    eri = cas.get_h2cas(mo_coeff=avas_orbs)
    eri = ao2mo.restore('s1', eri, h1.shape[0])  # chemist convention (11|22)
    ecore = float(ecore)

    num_alpha, num_beta = get_num_active_alpha_beta(pyscf_mf, cas_electrons)

    return h1, eri, ecore, num_alpha, num_beta


def get_num_active_alpha_beta(pyscf_mf, cas_electrons):
    """ Return number of alpha and beta electrons in the active space given
        number of CAS electrons
        This assumes that all the unpaired electrons are in the active space

    Args:
        pyscf_mf: PySCF mean field object
        cas_orbitals (int):  number of electrons in CAS space,

    Returns:
        num_alpha (int): number of alpha (spin-up) electrons in active space
        num_beta (int):  number of beta (spin-down) electrons in active space
    """
    # Sanity checks and active space info
    total_electrons = pyscf_mf.mol.nelectron
    frozen_electrons = total_electrons - cas_electrons
    assert frozen_electrons % 2 == 0

    # ROHF == RHF but RHF != ROHF, and we only do either RHF or ROHF
    if isinstance(pyscf_mf, scf.rohf.ROHF):
        frozen_alpha = frozen_electrons // 2
        frozen_beta = frozen_electrons // 2
        num_alpha = pyscf_mf.nelec[0] - frozen_alpha
        num_beta = pyscf_mf.nelec[1] - frozen_beta
        assert np.isclose(num_beta + num_alpha, cas_electrons)

    else:
        assert cas_electrons % 2 == 0
        num_alpha = cas_electrons // 2
        num_beta = cas_electrons // 2

    return num_alpha, num_beta


def load_casfile_to_pyscf(fname,
                          num_alpha: Optional[int] = None,
                          num_beta: Optional[int] = None):
    """ Load CAS Hamiltonian from pre-computed HD5 file into a PySCF molecule
        and mean-field object

    Args:
        fname (str): path to hd5 file to be created containing CAS one and two
                     body terms
        num_alpha (int, optional): number of spin up electrons in CAS space
        num_beta (int, optional):  number of spin down electrons in CAS space

    Returns:
        pyscf_mol: PySCF molecule object
        pyscf_mf:  PySCF mean field object
    """

    with h5py.File(fname, "r") as f:
        eri = np.asarray(f['eri'][()])
        # h1 one body elements are sometimes called different things. Try a few.
        try:
            h1 = np.asarray(f['h0'][()])
        except KeyError:
            try:
                h1 = np.asarray(f['hcore'][()])
            except KeyError:
                try:
                    h1 = np.asarray(f['h1'][()])
                except KeyError:
                    raise KeyError("Could not find 1-electron Hamiltonian")
        # ecore sometimes exists, and sometimes as enuc (no frozen electrons)
        try:
            ecore = float(f['ecore'][()])
        except KeyError:
            try:
                ecore = float(f['enuc'][()])
            except KeyError:
                ecore = 0.0
        # read the number of spin up and spin down electrons if not input
        if (num_alpha is None) or (num_beta is None):
            try:
                num_alpha = int(f['active_nalpha'][()])
            except KeyError:
                sys.exit("In `load_casfile_to_pyscf()`: \n" + \
                         " No values found on file for num_alpha " + \
                         "(key: 'active_nalpha' in h5). " + \
                         " Try passing in a value for num_alpha, or" + \
                         " re-check integral file.")
            try:
                num_beta = int(f['active_nbeta'][()])
            except KeyError:
                sys.exit("In `load_casfile_to_pyscf()`: \n" + \
                         " No values found on file for num_beta " + \
                         "(key: 'active_nbeta' in h5). " + \
                         " Try passing in a value for num_beta, or" + \
                         " re-check integral file.")

    pyscf_mol, pyscf_mf = cas_to_pyscf(h1, eri, ecore, num_alpha, num_beta)

    return pyscf_mol, pyscf_mf


def save_pyscf_to_casfile(fname,
                          pyscf_mf,
                          cas_orbitals: Optional[int] = None,
                          cas_electrons: Optional[int] = None,
                          avas_orbs=None):
    """ Save CAS Hamiltonian from a PySCF mean-field object to an HD5 file

    Args:
        fname (str): path to hd5 file to be created containing CAS terms
        pyscf_mf: PySCF mean field object
        cas_orbitals (int, optional):  number of orb in CAS space, default all
        cas_electrons (int, optional): number of elec in CAS, default all elec
        avas_orbs (ndarray, optional): orbitals selected by AVAS in PySCF
    """
    h1, eri, ecore, num_alpha, num_beta = \
        pyscf_to_cas(pyscf_mf, cas_orbitals, cas_electrons, avas_orbs)

    with h5py.File(fname, 'w') as fid:
        fid.create_dataset('ecore', data=float(ecore), dtype=float)
        fid.create_dataset(
            'h0',
            data=h1)  # note the name change to be consistent with THC paper
        fid.create_dataset('eri', data=eri)
        fid.create_dataset('active_nalpha', data=int(num_alpha), dtype=int)
        fid.create_dataset('active_nbeta', data=int(num_beta), dtype=int)


def factorized_ccsd_t(pyscf_mf, eri_rr = None, use_kernel = True,\
    no_triples=False) -> Tuple[float, float, float]:
    """ Compute CCSD(T) energy using rank-reduced ERIs

    Args:
        pyscf_mf - PySCF mean field object
        eri_rr (ndarray) - rank-reduced ERIs, or use full ERIs from pyscf_mf
        use_kernel (bool) - re-do SCF, using canonical orbitals for one-body?
        no_triples (bool) - skip the perturbative triples correction? (CCSD)

    Returns:
        e_scf (float) - SCF energy
        e_cor (float) - Correlation energy from CCSD(T)
        e_tot (float) - Total energy; i.e. SCF + Corr energy from CCSD(T)
    """
    h1, eri_full, ecore, num_alpha, num_beta = pyscf_to_cas(pyscf_mf)

    # If no rank-reduced ERIs, use the full (possibly local) ERIs from pyscf_mf
    if eri_rr is None:
        eri_rr = eri_full

    e_scf, e_cor, e_tot = ccsd_t(h1, eri_rr, ecore, num_alpha, num_beta,\
        eri_full, use_kernel, no_triples)

    return e_scf, e_cor, e_tot


def ccsd_t(h1, eri, ecore, num_alpha: int, num_beta: int, eri_full = None,\
    use_kernel=True, no_triples=False) -> Tuple[float, float, float]:
    """ Helper function to do CCSD(T) on set of one- and two-body Hamil elems

    Args:
        h1 (ndarray) - 2D matrix containing one-body terms (MO basis)
        eri (ndarray) - 4D tensor containing two-body terms (MO basis)
                        may be from integral factorization (e.g. SF/DF/THC)
        ecore (float) - frozen core electronic energy + nuclear repulsion energy
        num_alpha (int) - number of spin alpha electrons in Hamiltonian
        num_beta (int) - number of spin beta electrons in Hamiltonian
        eri_full (ndarray) - optional 4D tensor containing full two-body
            terms (MO basis) for the SCF procedure only
        use_kernel (bool) - re-run SCF prior to doing CCSD(T)?
        no_triples (bool) - skip the perturbative triples correction? (CCSD)

    Returns:
        e_scf (float) - SCF energy
        e_cor (float) - Correlation energy from CCSD(T)
        e_tot (float) - Total energy; i.e. SCF + Corr energy from CCSD(T)
    """

    mol = gto.M()
    mol.nelectron = num_alpha + num_beta
    n_orb = h1.shape[0]
    alpha_diag = [1] * num_alpha + [0] * (n_orb - num_alpha)
    beta_diag = [1] * num_beta + [0] * (n_orb - num_beta)

    # If eri_full not provided, use (possibly rank-reduced) ERIs for check
    if eri_full is None:
        eri_full = eri

    # either RHF or ROHF ... should be OK since UHF will have two h1s, etc.
    if num_alpha == num_beta:
        mf = scf.RHF(mol)
        scf_energy = ecore + \
                     2*np.einsum('ii',h1[:num_alpha,:num_alpha]) + \
                     2*np.einsum('iijj',eri_full[:num_alpha,\
                                                 :num_alpha,\
                                                 :num_alpha,\
                                                 :num_alpha]) - \
                       np.einsum('ijji',eri_full[:num_alpha,\
                                                 :num_alpha,\
                                                 :num_alpha,\
                                                 :num_alpha])

    else:
        mf = scf.ROHF(mol)
        mf.nelec = (num_alpha, num_beta)
        # grab singly and doubly occupied orbitals (assume high-spin open shell)
        docc = slice(None, min(num_alpha, num_beta))
        socc = slice(min(num_alpha, num_beta), max(num_alpha, num_beta))
        scf_energy = ecore + \
                     2.0*np.einsum('ii',h1[docc, docc]) + \
                         np.einsum('ii',h1[socc, socc]) + \
                     2.0*np.einsum('iijj',eri_full[docc, docc, docc, docc]) - \
                         np.einsum('ijji',eri_full[docc, docc, docc, docc]) + \
                         np.einsum('iijj',eri_full[socc, socc, docc, docc]) - \
                     0.5*np.einsum('ijji',eri_full[socc, docc, docc, socc]) + \
                         np.einsum('iijj',eri_full[docc, docc, socc, socc]) - \
                     0.5*np.einsum('ijji',eri_full[docc, socc, socc, docc]) + \
                     0.5*np.einsum('iijj',eri_full[socc, socc, socc, socc]) - \
                     0.5*np.einsum('ijji',eri_full[socc, socc, socc, socc])

    mf.get_hcore = lambda *args: np.asarray(h1)
    mf.get_ovlp = lambda *args: np.eye(h1.shape[0])
    mf.energy_nuc = lambda *args: ecore
    mf._eri = eri_full  # ao2mo.restore('8', np.zeros((8, 8, 8, 8)), 8)

    mf.init_guess = '1e'
    mf.mo_coeff = np.eye(n_orb)
    mf.mo_occ = np.array(alpha_diag) + np.array(beta_diag)
    w, _ = np.linalg.eigh(mf.get_fock())
    mf.mo_energy = w

    # Rotate the interaction tensors into the canonical basis.
    # Reiher and Li tensors, for example, are read-in in the local MO basis,
    # which is not optimal for the CCSD(T) calculation (canonical gives better
    # energy estimate whereas QPE is invariant to choice of basis)
    if use_kernel:
        mf.conv_tol = 1e-7
        mf.init_guess = '1e'
        mf.verbose = 4
        mf.diis_space = 24
        mf.level_shift = 0.5
        mf.conv_check = False
        mf.max_cycle = 800
        mf.kernel(mf.make_rdm1(mf.mo_coeff,
                               mf.mo_occ))  # use MO info to generate guess
        mf = stability(mf)
        mf = stability(mf)
        mf = stability(mf)

        # Check if SCF has changed by doing restart, and print warning if so
        try:
            assert np.isclose(scf_energy, mf.e_tot, rtol=1e-14)
        except AssertionError:
            print(
                "WARNING: E(SCF) from input integrals does not match E(SCF)" + \
                " from mf.kernel()")
            print("  Will use E(SCF) = {:12.6f} from mf.kernel going forward.".
                  format(mf.e_tot))
        print("E(SCF, ints) = {:12.6f} whereas E(SCF) = {:12.6f}".format(
            scf_energy, mf.e_tot))

        # New SCF energy and orbitals for CCSD(T)
        scf_energy = mf.e_tot

    # Now re-set the eri's to the (possibly rank-reduced) ERIs
    mf._eri = eri
    mf.mol.incore_anyway = True

    mycc = cc.CCSD(mf)
    mycc.max_cycle = 800
    mycc.conv_tol = 1E-8
    mycc.conv_tol_normt = 1E-4
    mycc.diis_space = 24
    mycc.verbose = 4
    mycc.kernel()

    if no_triples:
        et = 0.0
    else:
        et = mycc.ccsd_t()

    e_scf = scf_energy  # may be read-in value or 'fresh' SCF value
    e_cor = mycc.e_corr + et
    e_tot = e_scf + e_cor

    print("E(SCF):       ", e_scf)
    print("E(cor):       ", e_cor)
    print("Total energy: ", e_tot)
    return e_scf, e_cor, e_tot


def open_shell_t1_d1(t1a, t1b, mo_occ, nalpha, nbeta):
    """
    T1-diagnostic for open-shell is defined w.r.t Sx eigenfunction of T1
        where reference is ROHF.

    given i double occ, c unoccupied, x is single occuplied The T1 amps
      (high spin) in Sz basis are:
    T1 = t_{ia}^{ca}(ca^ ia) + t_{ib}^{cb}(cb^ ib)
       + t_{xa}^{ca}(ca^ xa) + t_{ib}^{xb}(xb^ ib)
    T1 in the Sx basis are
    T1 = f_{i}^{c}E_{ci} + v_{i}^{c}A_{ci}
       + sqrt(2)f_{x}^{c}(ca^ xa) + sqrt(2)f_{i}^{x}(xb^ ib)

    where E_{ci} = ca^ ia + cb^ ib and A_{ci} = ca^ ia - cb^ ib.

    See:  The Journal of Chemical Physics 98, 9734 (1993);
              doi: 10.1063/1.464352
          Chemical Physics Letters 372 (2003) 362–367;
              doi:10.1016/S0009-2614(03)00435-4

    based on these and two papers from Lee the T1-openshell diagnostic is

    sqrt(sum_{ia}(f_{ia})^2 + 2sum_{xa}(t_{xa}^{ca})^2
        + 2 sum_{ix}(t_{ib}^{xb})^2) / 2 sqrt{N}

    To get this relate eqs 3-7 from Chemical Physics Letters 372 (2003) 362–367
    to Eqs. 45, 46, and 51 from Journal of Chemical Physics 98, 9734 (1993);
    doi: 10.1063/1.464352.
    """
    # compute t1-diagnostic
    docc_idx = np.where(np.isclose(mo_occ, 2.))[0]
    socc_idx = np.where(np.isclose(mo_occ, 1.))[0]
    virt_idx = np.where(np.isclose(mo_occ, 0.))[0]
    t1a_docc = t1a[docc_idx, :]  # double occ-> virtual
    t1b_docc = t1b[docc_idx, :][:, -len(virt_idx):]  # double occ-> virtual
    if len(socc_idx) > 0:
        t1_xa = t1a[socc_idx, :]  # single occ -> virtual
        t1_ix = t1b[docc_idx, :][:, :len(socc_idx)]  # double occ -> single occ
    else:
        t1_xa = np.array(())
        t1_ix = np.array(())

    if nalpha - nbeta + len(virt_idx) != t1b.shape[1]:
        raise ValueError(
            "Inconsistent shapes na {}, nb {}, t1b.shape {},{}".format(
                nalpha, nbeta, t1b.shape[0], t1b.shape[1]))

    if t1a_docc.shape != (len(docc_idx), len(virt_idx)):
        raise ValueError("T1a_ia does not have the right shape")
    if t1b_docc.shape != (len(docc_idx), len(virt_idx)):
        raise ValueError("T1b_ia does not have the right shape")
    if len(socc_idx) > 0:
        if t1_ix.shape != (len(docc_idx), len(socc_idx)):
            raise ValueError("T1_ix does not have the right shape")
        if t1_xa.shape != (len(socc_idx), len(virt_idx)):
            raise ValueError("T1_xa does not have the right shape")

    t1_diagnostic = np.sqrt(
        np.sum((t1a_docc + t1b_docc)**2) + 2 * np.sum(t1_xa**2) +
        2 * np.sum(t1_ix**2)) / (2 * np.sqrt(nalpha + nbeta))
    # compute D1-diagnostic
    f_ia = 0.5 * (t1a_docc + t1b_docc)
    s_f_ia_2, _ = np.linalg.eigh(f_ia @ f_ia.T)
    s_f_ia_2_norm = np.sqrt(np.max(s_f_ia_2, initial=0))

    if len(socc_idx) > 0:
        f_xa = np.sqrt(1 / 2) * t1_xa
        f_ix = np.sqrt(1 / 2) * t1_ix
        s_f_xa_2, _ = np.linalg.eigh(f_xa @ f_xa.T)
        s_f_ix_2, _ = np.linalg.eigh(f_ix @ f_ix.T)
    else:
        s_f_xa_2 = np.array(())
        s_f_ix_2 = np.array(())
    s_f_xa_2_norm = np.sqrt(np.max(s_f_xa_2, initial=0))
    s_f_ix_2_norm = np.sqrt(np.max(s_f_ix_2, initial=0))

    d1_diagnostic = np.max(
        np.array([s_f_ia_2_norm, s_f_xa_2_norm, s_f_ix_2_norm]))

    return t1_diagnostic, d1_diagnostic
