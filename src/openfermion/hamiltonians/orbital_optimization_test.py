import itertools

import numpy as np
import pytest
import scipy as sp

from openfermion.chem import MolecularData
from openfermion.config import DATA_DIRECTORY
from openfermion.hamiltonians.hartree_fock import generate_hamiltonian, rhf_params_to_matrix
from openfermion.hamiltonians.orbital_optimization import optimize_orbitals
from openfermion.linalg import expectation, get_ground_state, get_sparse_operator
from openfermion.ops.operators import FermionOperator
from openfermion.ops.representations import general_basis_change


def _fci_ground_state_rdms(hamiltonian, n_qubits):
    """1-/2-RDM of the exact ground state, computed directly from the state
    vector.

    Same construction `measurements.get_interaction_rdm` uses, but starting
    from `linalg.get_ground_state` instead of a real measurement -- not via
    a measured qubit operator, appropriate for a known-answer test.
    """
    sparse_h = get_sparse_operator(hamiltonian, n_qubits=n_qubits)
    energy, state = get_ground_state(sparse_h)

    one_rdm = np.zeros((n_qubits, n_qubits))
    for p, q in itertools.product(range(n_qubits), repeat=2):
        op = get_sparse_operator(FermionOperator(((p, 1), (q, 0))), n_qubits=n_qubits)
        one_rdm[p, q] = expectation(op, state).real

    two_rdm = np.zeros((n_qubits,) * 4)
    for p, q, r, s in itertools.product(range(n_qubits), repeat=4):
        op = get_sparse_operator(
            FermionOperator(((p, 1), (q, 1), (r, 0), (s, 0))), n_qubits=n_qubits
        )
        two_rdm[p, q, r, s] = expectation(op, state).real

    return energy, one_rdm, two_rdm


def _load_h2(bond_length='0.7414', basis_suffix='sto-3g'):
    m = MolecularData(filename=f"{DATA_DIRECTORY}/H2_{basis_suffix}_singlet_{bond_length}.hdf5")
    m.load()
    return m


def test_optimize_orbitals_at_identity_reproduces_full_ci_energy():
    """A necessary correctness check.

    With the RDM taken from a full (untruncated) FCI calculation in the
    reference orbitals, evaluating the objective at kappa=0 (the identity
    rotation) must reproduce the FCI energy exactly -- this just re-checks
    that energy() is wired up to the same accounting
    `InteractionRDM.expectation` uses, nothing about optimization yet.
    """
    m = _load_h2()
    hamiltonian = m.get_molecular_hamiltonian()
    fci_energy, one_rdm, two_rdm = _fci_ground_state_rdms(hamiltonian, m.n_qubits)

    result = optimize_orbitals(
        m.one_body_integrals,
        m.two_body_integrals,
        one_rdm,
        two_rdm,
        m.n_electrons,
        nuclear_repulsion=m.nuclear_repulsion,
        initial_guess=np.zeros((m.n_electrons // 2) * (m.n_orbitals - m.n_electrons // 2)),
        method='Nelder-Mead',
        verbose=False,
        sp_options={'maxiter': 1},  # don't actually move -- just evaluate near kappa=0
    )
    assert np.isclose(result.fun, fci_energy, atol=1e-6)


def test_full_space_fci_rdm_is_never_beaten_by_any_rotation():
    """Physical correctness check, not a code-behavior tautology.

    When the RDM comes from a full-space FCI calculation, no orbital
    rotation can produce a state with LOWER energy than the FCI value,
    because a rotation among all M orbitals stays inside the same complete
    N-electron Fock space that full CI already minimizes over exactly.
    kappa=0 must therefore be a global minimum of energy(kappa) -- the
    optimizer, started away from kappa=0, must converge back down to (not
    below) the FCI energy, and any explicit nonzero kappa must score
    >= the FCI energy.
    """
    m = _load_h2()
    hamiltonian = m.get_molecular_hamiltonian()
    fci_energy, one_rdm, two_rdm = _fci_ground_state_rdms(hamiltonian, m.n_qubits)

    # explicit nonzero rotations must not beat the FCI floor
    n_orbitals = m.n_orbitals
    nocc = m.n_electrons // 2
    occ = list(range(nocc))
    virt = list(range(nocc, n_orbitals))
    for scale in (0.3, -0.7, 1.2):
        params = np.full(nocc * (n_orbitals - nocc), scale)
        kappa = rhf_params_to_matrix(params, n_orbitals, occ, virt)
        rotation = sp.linalg.expm(kappa)
        rotated_obi = general_basis_change(m.one_body_integrals, rotation, (1, 0), transpose=False)
        rotated_tbi = general_basis_change(
            m.two_body_integrals, rotation, (1, 1, 0, 0), transpose=False
        )
        rotated_hamiltonian = generate_hamiltonian(rotated_obi, rotated_tbi, m.nuclear_repulsion)
        energy = rotated_hamiltonian.constant
        energy += np.sum(one_rdm * rotated_hamiltonian.one_body_tensor).real
        energy += np.sum(two_rdm * rotated_hamiltonian.two_body_tensor).real
        assert energy >= fci_energy - 1e-8, (
            f"rotation with params={scale} scored below the FCI floor -- "
            f"got {energy}, floor is {fci_energy}"
        )

    # the optimizer, started away from kappa=0, must converge back to the floor
    rng = np.random.default_rng(1234)
    init = rng.normal(scale=0.4, size=nocc * (n_orbitals - nocc))
    result = optimize_orbitals(
        m.one_body_integrals,
        m.two_body_integrals,
        one_rdm,
        two_rdm,
        m.n_electrons,
        nuclear_repulsion=m.nuclear_repulsion,
        initial_guess=init,
        verbose=False,
    )
    assert np.isclose(result.fun, fci_energy, atol=1e-5)
    assert result.fun >= fci_energy - 1e-6


def test_optimize_orbitals_improves_a_truncated_active_space():
    """The realistic use case.

    A CASCI-style active-space-truncated RDM (computed via canonical/
    reference orbitals, which are not generally CASSCF-optimal) should
    either be improved by orbital rotation or, at worst, left unchanged --
    never made worse than the untruncated (kappa=0) starting point.
    """
    m = _load_h2(bond_length='0.75', basis_suffix='6-31g')
    # 4 spatial orbitals total; restrict the active CI space to the lowest 2
    # (drop the top 2 virtuals from the CI problem, but keep them in the
    # one-/two-body integral tensors that optimize_orbitals rotates over --
    # this is exactly the "orbital rotation between active and excluded
    # space is not a symmetry" scenario orbital optimization targets).
    active_indices = [0, 1]
    active_hamiltonian = m.get_molecular_hamiltonian(active_indices=active_indices)
    n_active_qubits = 2 * len(active_indices)
    active_energy, active_one_rdm, active_two_rdm = _fci_ground_state_rdms(
        active_hamiltonian, n_active_qubits
    )

    # Pad the active-space RDM back out to the full 4-orbital (8 spin-orbital)
    # tensor shape optimize_orbitals expects, with the excluded orbitals'
    # entries left at zero (unoccupied in this trial density).
    n_orbitals = m.n_orbitals
    n_spin_orbitals = 2 * n_orbitals
    n_active_spin = n_active_qubits
    one_rdm = np.zeros((n_spin_orbitals, n_spin_orbitals))
    one_rdm[:n_active_spin, :n_active_spin] = active_one_rdm
    two_rdm = np.zeros((n_spin_orbitals,) * 4)
    two_rdm[:n_active_spin, :n_active_spin, :n_active_spin, :n_active_spin] = active_two_rdm

    baseline = optimize_orbitals(
        m.one_body_integrals,
        m.two_body_integrals,
        one_rdm,
        two_rdm,
        n_electrons=2 * len(active_indices),
        nuclear_repulsion=m.nuclear_repulsion,
        initial_guess=np.zeros((len(active_indices)) * (n_orbitals - len(active_indices))),
        method='Nelder-Mead',
        verbose=False,
        sp_options={'maxiter': 1},
    )
    optimized = optimize_orbitals(
        m.one_body_integrals,
        m.two_body_integrals,
        one_rdm,
        two_rdm,
        n_electrons=2 * len(active_indices),
        nuclear_repulsion=m.nuclear_repulsion,
        verbose=False,
    )
    assert np.isclose(baseline.fun, active_energy, atol=1e-6)
    assert optimized.fun <= baseline.fun + 1e-8


def test_optimize_orbitals_rejects_degenerate_orbital_split():
    """No occupied or no virtual spatial orbitals -- nothing to rotate."""
    obi = np.zeros((2, 2))
    tbi = np.zeros((2, 2, 2, 2))
    one_rdm = np.zeros((4, 4))
    two_rdm = np.zeros((4, 4, 4, 4))
    with pytest.raises(ValueError):
        optimize_orbitals(obi, tbi, one_rdm, two_rdm, n_electrons=4)  # all occupied, no virtuals
    with pytest.raises(ValueError):
        optimize_orbitals(obi, tbi, one_rdm, two_rdm, n_electrons=0)  # all virtual, no occupied


def test_optimize_orbitals_rejects_mismatched_shapes():
    """No shape validation caught early (found by automated code review).

    A mismatched one_body_integrals/two_body_integrals/one_rdm/two_rdm/
    n_electrons combination would fail deep inside energy() with a
    confusing broadcast/index error instead of a clear message at the
    call boundary.
    """
    n_orbitals = 3
    obi = np.zeros((n_orbitals, n_orbitals))
    tbi = np.zeros((n_orbitals,) * 4)
    one_rdm = np.zeros((2 * n_orbitals, 2 * n_orbitals))
    two_rdm = np.zeros((2 * n_orbitals,) * 4)

    with pytest.raises(ValueError, match="one_body_integrals"):
        optimize_orbitals(np.zeros((n_orbitals, n_orbitals + 1)), tbi, one_rdm, two_rdm, 4)
    with pytest.raises(ValueError, match="two_body_integrals"):
        optimize_orbitals(obi, np.zeros((n_orbitals + 1,) * 4), one_rdm, two_rdm, 4)
    with pytest.raises(ValueError, match="one_rdm"):
        optimize_orbitals(obi, tbi, np.zeros((2 * n_orbitals + 1,) * 2), two_rdm, 4)
    with pytest.raises(ValueError, match="two_rdm"):
        optimize_orbitals(obi, tbi, one_rdm, np.zeros((2 * n_orbitals + 1,) * 4), 4)


def test_optimize_orbitals_rejects_odd_electron_count():
    """optimize_orbitals is restricted (closed-shell).

    n_electrons // 2 silently rounds an odd count down, which would
    optimize the wrong number of occupied orbitals without any warning.
    """
    n_orbitals = 3
    obi = np.zeros((n_orbitals, n_orbitals))
    tbi = np.zeros((n_orbitals,) * 4)
    one_rdm = np.zeros((2 * n_orbitals, 2 * n_orbitals))
    two_rdm = np.zeros((2 * n_orbitals,) * 4)
    with pytest.raises(ValueError, match="even"):
        optimize_orbitals(obi, tbi, one_rdm, two_rdm, n_electrons=3)


def test_optimize_orbitals_rejects_mismatched_initial_guess():
    """A caller-supplied initial_guess of the wrong length.

    Would otherwise hit an IndexError deep inside rhf_params_to_matrix
    instead of a clear message naming the expected parameter count.
    """
    m = _load_h2()
    hamiltonian = m.get_molecular_hamiltonian()
    _, one_rdm, two_rdm = _fci_ground_state_rdms(hamiltonian, m.n_qubits)
    with pytest.raises(ValueError, match="initial_guess"):
        optimize_orbitals(
            m.one_body_integrals,
            m.two_body_integrals,
            one_rdm,
            two_rdm,
            m.n_electrons,
            initial_guess=np.zeros(5),  # wrong size for this molecule (H2/sto-3g needs 1)
            verbose=False,
        )
