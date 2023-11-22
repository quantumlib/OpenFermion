# coverage:ignore
""" Pretty-print a table comparing number of SF vectors versus acc and cost """
import numpy as np
import pytest

from openfermion.resource_estimates import HAVE_DEPS_FOR_RESOURCE_ESTIMATES, sf

if HAVE_DEPS_FOR_RESOURCE_ESTIMATES:
    from pyscf import scf

    from openfermion.resource_estimates.molecule import (
        cas_to_pyscf,
        factorized_ccsd_t,
        pyscf_to_cas,
    )


@pytest.mark.skipif(not HAVE_DEPS_FOR_RESOURCE_ESTIMATES, reason='pyscf and/or jax not installed.')
def generate_costing_table(
    pyscf_mf,
    name='molecule',
    rank_range=range(50, 401, 25),
    chi=10,
    dE=0.001,
    use_kernel=True,
    no_triples=False,
):
    """Print a table to file for how various SF ranks impact cost, acc., etc.

    Args:
        pyscf_mf - PySCF mean field object
        name (str) - file will be saved to 'single_factorization_<name>.txt'
        rank_range (list of ints) - list number of vectors to retain in SF alg
        dE (float) - max allowable phase error (default: 0.001)
        chi (int) - number of bits for representation of coefficients
           (default: 10)
        use_kernel (bool) - re-do SCF prior to estimating CCSD(T) error?
            Will use canonical orbitals and full ERIs for the one-body
            contributions, using rank-reduced ERIs for two-body
        no_triples (bool) - if True, skip the (T) correction, doing only CCSD

    Returns:
       None
    """

    DE = dE  # max allowable phase error
    CHI = chi  # number of bits for representation of coefficients

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

    # Reference calculation (eri_rr = None is full rank / exact ERIs)
    escf, ecor, etot = factorized_ccsd_t(
        pyscf_mf, eri_rr=None, use_kernel=use_kernel, no_triples=no_triples
    )

    exact_etot = etot

    filename = 'single_factorization_' + name + '.txt'

    with open(filename, 'w') as f:
        print("\n Single low rank factorization data for '" + name + "'.", file=f)
        print("    [*] using " + cas_info, file=f)
        print("        [+]                      E(SCF): %18.8f" % escf, file=f)
        if no_triples:
            print("        [+]    Active space CCSD E(cor): %18.8f" % ecor, file=f)
            print("        [+]    Active space CCSD E(tot): %18.8f" % etot, file=f)
        else:
            print("        [+] Active space CCSD(T) E(cor): %18.8f" % ecor, file=f)
            print("        [+] Active space CCSD(T) E(tot): %18.8f" % etot, file=f)
        print("{}".format('=' * 108), file=f)
        if no_triples:
            print(
                "{:^12} {:^18} {:^12} {:^24} {:^20} {:^20}".format(
                    'L',
                    '||ERI - SF||',
                    'lambda',
                    'CCSD error (mEh)',
                    'logical qubits',
                    'Toffoli count',
                ),
                file=f,
            )
        else:
            print(
                "{:^12} {:^18} {:^12} {:^24} {:^20} {:^20}".format(
                    'L',
                    '||ERI - SF||',
                    'lambda',
                    'CCSD(T) error (mEh)',
                    'logical qubits',
                    'Toffoli count',
                ),
                file=f,
            )
        print("{}".format('-' * 108), file=f)
    for rank in rank_range:
        # First, up: lambda and CCSD(T)
        eri_rr, LR = sf.factorize(pyscf_mf._eri, rank)
        lam = sf.compute_lambda(pyscf_mf, LR)
        escf, ecor, etot = factorized_ccsd_t(
            pyscf_mf, eri_rr, use_kernel=use_kernel, no_triples=no_triples
        )
        error = (etot - exact_etot) * 1e3  # to mEh
        l2_norm_error_eri = np.linalg.norm(eri_rr - pyscf_mf._eri)  # eri reconstruction error

        # now do costing
        stps1 = sf.compute_cost(num_spinorb, lam, DE, L=rank, chi=CHI, stps=20000)[0]

        _, sf_total_cost, sf_logical_qubits = sf.compute_cost(
            num_spinorb, lam, DE, L=rank, chi=CHI, stps=stps1
        )

        with open(filename, 'a') as f:
            print(
                "{:^12} {:^18.4e} {:^12.1f} {:^24.2f} {:^20} {:^20.1e}".format(
                    rank, l2_norm_error_eri, lam, error, sf_logical_qubits, sf_total_cost
                ),
                file=f,
            )
    with open(filename, 'a') as f:
        print("{}".format('=' * 108), file=f)
