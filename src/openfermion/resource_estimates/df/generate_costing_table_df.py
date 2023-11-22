# coverage:ignore
""" Pretty-print a table comparing DF vector thresh vs accuracy and cost """
import numpy as np
from pyscf import scf
from openfermion.resource_estimates import df
from openfermion.resource_estimates.molecule import factorized_ccsd_t, cas_to_pyscf, pyscf_to_cas


def generate_costing_table(
    pyscf_mf,
    name='molecule',
    thresh_range=None,
    dE=0.001,
    chi=10,
    beta=20,
    use_kernel=True,
    no_triples=False,
):
    """Print a table to file for testing how various DF thresholds impact cost,
        accuracy, etc.

    Args:
        pyscf_mf - PySCF mean field object
        name (str) - file will be saved to 'double_factorization_<name>.txt'
        thresh_range (list of floats) - list of thresholds to try for DF alg
        dE (float) - max allowable phase error (default: 0.001)
        chi (int) - number of bits for representation of coefficients
                    (default: 10)
        beta (int) - number of bits for rotations (default: 20)
        use_kernel (bool) - re-do SCF prior to estimating CCSD(T) error?
            Will use canonical orbitals and full ERIs for the one-body
            contributions, using DF reconstructed ERIs for two-body
        no_triples (bool) - if True, skip the (T) correction, doing only CCSD

    Returns:
       None
    """

    if thresh_range is None:
        thresh_range = [0.0001]

    DE = dE  # max allowable phase error
    CHI = chi  # number of bits for representation of coefficients
    BETA = beta  # number of bits for rotations

    if isinstance(pyscf_mf, scf.rohf.ROHF):
        num_alpha, num_beta = pyscf_mf.nelec
        assert num_alpha + num_beta == pyscf_mf.mol.nelectron
    else:
        assert pyscf_mf.mol.nelectron % 2 == 0
        num_alpha = pyscf_mf.mol.nelectron // 2
        num_beta = num_alpha

    num_orb = len(pyscf_mf.mo_coeff)
    num_spinorb = num_orb * 2

    cas_info = "CAS((%sa, %sb), %so)" % (num_alpha, num_beta, num_orb)

    try:
        assert num_orb**4 == len(pyscf_mf._eri.flatten())
    except AssertionError:
        # ERIs are not in correct form in pyscf_mf._eri, so this is a quick prep
        _, pyscf_mf = cas_to_pyscf(*pyscf_to_cas(pyscf_mf))

    # Reference calculation (eri_rr= None is full rank / exact ERIs)
    escf, ecor, etot = factorized_ccsd_t(
        pyscf_mf, eri_rr=None, use_kernel=use_kernel, no_triples=no_triples
    )

    # exact_ecor = ecor
    exact_etot = etot

    filename = 'double_factorization_' + name + '.txt'

    with open(filename, 'w') as f:
        print("\n Double low rank factorization data for '" + name + "'.", file=f)
        print("    [*] using " + cas_info, file=f)
        print("        [+]                      E(SCF): %18.8f" % escf, file=f)
        if no_triples:
            print("        [+]    Active space CCSD E(cor): %18.8f" % ecor, file=f)
            print("        [+]    Active space CCSD E(tot): %18.8f" % etot, file=f)
        else:
            print("        [+] Active space CCSD(T) E(cor): %18.8f" % ecor, file=f)
            print("        [+] Active space CCSD(T) E(tot): %18.8f" % etot, file=f)
        print("{}".format('=' * 139), file=f)
        if no_triples:
            print(
                "{:^12} {:^18} {:^12} {:^12} {:^12} {:^24} {:^20} {:^20}".format(
                    'threshold',
                    '||ERI - DF||',
                    'L',
                    'eigenvectors',
                    'lambda',
                    'CCSD error (mEh)',
                    'logical qubits',
                    'Toffoli count',
                ),
                file=f,
            )
        else:
            print(
                "{:^12} {:^18} {:^12} {:^12} {:^12} {:^24} {:^20} {:^20}".format(
                    'threshold',
                    '||ERI - DF||',
                    'L',
                    'eigenvectors',
                    'lambda',
                    'CCSD(T) error (mEh)',
                    'logical qubits',
                    'Toffoli count',
                ),
                file=f,
            )
        print("{}".format('-' * 139), file=f)
    for thresh in thresh_range:
        # First, up: lambda and CCSD(T)
        eri_rr, LR, L, Lxi = df.factorize(pyscf_mf._eri, thresh=thresh)
        lam = df.compute_lambda(pyscf_mf, LR)
        escf, ecor, etot = factorized_ccsd_t(
            pyscf_mf, eri_rr, use_kernel=use_kernel, no_triples=no_triples
        )
        error = (etot - exact_etot) * 1e3  # to mEh
        l2_norm_error_eri = np.linalg.norm(eri_rr - pyscf_mf._eri)  # ERI reconstruction error

        # now do costing
        stps1 = df.compute_cost(num_spinorb, lam, DE, L=L, Lxi=Lxi, chi=CHI, beta=BETA, stps=20000)[
            0
        ]
        _, df_total_cost, df_logical_qubits = df.compute_cost(
            num_spinorb, lam, DE, L=L, Lxi=Lxi, chi=CHI, beta=BETA, stps=stps1
        )

        with open(filename, 'a') as f:
            print(
                "{:^12.6f} {:^18.4e} {:^12} {:^12} {:^12.1f} {:^24.2f} {:^20} \
                 {:^20.1e}".format(
                    thresh, l2_norm_error_eri, L, Lxi, lam, error, df_logical_qubits, df_total_cost
                ),
                file=f,
            )
    with open(filename, 'a') as f:
        print("{}".format('=' * 139), file=f)
