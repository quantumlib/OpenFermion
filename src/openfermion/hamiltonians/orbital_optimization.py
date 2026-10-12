"""
Basic orbital optimization for a fixed 1- and 2-particle reduced density
matrix (RDM), restricted (closed-shell) case.

Motivation: a common post-processing step for a correlated calculation
(CASCI/FCI in a truncated active space, or a 1-/2-RDM recovered from a
quantum device via sample-based diagonalization) is to ask whether a
different choice of one-particle orbitals -- expressed as a rotation of the
orbitals the RDM was computed in -- would lower the total energy, holding the
RDM itself fixed. This module reuses the existing restricted-orbital
generator parametrization from `hartree_fock.py` (`rhf_params_to_matrix`,
an anti-Hermitian kappa matrix restricted to occupied/virtual blocks,
exponentiated into a unitary) and asks SciPy to minimize the resulting
energy over the rotation parameters.

This is deliberately "basic" (per the issue's own wording): the gradient
used is SciPy's numerical one, not an analytic one. Each trial rotation is
scored by rotating the Hamiltonian integrals into the trial basis (via
`general_basis_change`, the same utility `HartreeFockFunctional.__init__`
already uses to change basis) and evaluating that rotated Hamiltonian
against the *fixed* given RDM -- i.e., holding the CI (configuration
interaction) wavefunction's expansion coefficients fixed while asking what
energy those same coefficients would give if they described occupations of
a different, rotated one-particle basis instead. This is a physically real
question with a nontrivial answer, not a change of labels: reusing a
wavefunction's coefficients under rotated orbitals is generally a different
state, and its energy is generally different from (and, importantly, never
lower than -- see below) the state the RDM actually came from.

Correctness/scope note: for a *full* active space (all molecular orbitals
included, RDM from an untruncated FCI calculation), the identity rotation
(kappa=0) is provably the *global minimum* of this objective -- any
rotation keeps the trial state inside the same complete N-electron Fock
space that full CI already minimizes over exactly, so no rotation can
score below the FCI energy, and the optimizer started away from kappa=0
must converge back down to it (not below). The routine's actual use case
is the active-space-truncated case, where the RDM comes from a CI
diagonalization over a strict subset of orbitals -- there, orbital
rotation between the active and excluded space is not a symmetry of the
truncated problem, and rotating orbitals can genuinely recover some of the
energy lost to the truncation (this is exactly the orbital-rotation step
of CASSCF-style methods).
"""

from typing import Optional

import numpy as np
import scipy as sp
from scipy.optimize import OptimizeResult

from openfermion.hamiltonians.hartree_fock import generate_hamiltonian, rhf_params_to_matrix
from openfermion.ops.representations import general_basis_change


def _energy_from_rdms(
    hamiltonian_one_body: np.ndarray,
    hamiltonian_two_body: np.ndarray,
    constant: float,
    one_rdm: np.ndarray,
    two_rdm: np.ndarray,
) -> float:
    r"""⟨H⟩ for a fixed Hamiltonian and a fixed (possibly rotated) RDM pair.

    Uses the same elementwise-sum-product convention as
    `InteractionRDM.expectation()` (both tensors are assumed to already be
    expressed in the same orbital-index basis).
    """
    energy = constant
    energy += np.sum(one_rdm * hamiltonian_one_body).real
    energy += np.sum(two_rdm * hamiltonian_two_body).real
    return energy


def optimize_orbitals(
    one_body_integrals: np.ndarray,
    two_body_integrals: np.ndarray,
    one_rdm: np.ndarray,
    two_rdm: np.ndarray,
    n_electrons: int,
    *,
    nuclear_repulsion: float = 0.0,
    initial_guess: Optional[np.ndarray] = None,
    method: str = 'BFGS',
    verbose: bool = True,
    sp_options: Optional[dict] = None,
) -> OptimizeResult:
    r"""Restricted orbital-rotation optimization for a fixed 1-/2-RDM.

    Finds the anti-Hermitian generator kappa (parametrized exactly as in
    `hartree_fock.rhf_params_to_matrix` -- a rotation restricted to
    occupied-virtual blocks, using `n_electrons // 2` occupied spatial
    orbitals) that minimizes

    $$E(\kappa) = \sum_{pq} h_{pq}(\kappa) D_{qp}
                + \sum_{pqrs} V_{pqrs}(\kappa) \Gamma_{qpsr}$$

    where h(kappa)/V(kappa) are `one_body_integrals`/`two_body_integrals`
    rotated into the trial orbital basis U(kappa) = expm(kappa), and D/Gamma
    are the *fixed* given `one_rdm`/`two_rdm` (i.e. the CI wavefunction's
    expansion coefficients are held fixed while the orbitals they refer to
    are rotated, not re-solved at every step).

    Args:
        one_body_integrals: spatial-orbital one-body integrals, shape
            (n_orbitals, n_orbitals), in the same reference basis the RDMs
            were computed in.
        two_body_integrals: spatial-orbital two-body integrals, shape
            (n_orbitals,) * 4, chemist ordering matching
            `hartree_fock.generate_hamiltonian`.
        one_rdm: fixed spin-orbital 1-RDM, $\langle a_p^\dagger a_q
            \rangle$, shape (2 * n_orbitals,) * 2, in the same reference
            basis.
        two_rdm: fixed spin-orbital 2-RDM, $\langle a_p^\dagger a_q^\dagger
            a_r a_s \rangle$, shape (2 * n_orbitals,) * 4, in the same
            reference basis.
        n_electrons: total electron count (used only to split occupied vs.
            virtual spatial orbitals for the restricted parametrization;
            the RDM's actual trace need not equal this exactly, e.g. for an
            active-space RDM computed with frozen core orbitals excluded
            from `one_body_integrals`/`two_body_integrals` -- pass the
            electron count for *this* integral set).
        nuclear_repulsion: constant energy offset added to every evaluation.
        initial_guess: starting kappa parameter vector. Defaults to zero
            (start from the reference orbitals, i.e. no rotation).
        method: scipy.optimize.minimize method. Gradient-free by default
            (numerical differentiation) -- see module docstring.
        verbose: passed through as SciPy's 'disp' option.
        sp_options: extra options merged into the SciPy optimizer options.

    Returns:
        scipy.optimize.OptimizeResult. `result.x` is the optimal kappa
        parameter vector; `result.fun` is the optimized energy.
    """
    if one_body_integrals.ndim != 2 or one_body_integrals.shape[0] != one_body_integrals.shape[1]:
        raise ValueError(
            f"one_body_integrals must be a square 2D array, got shape "
            f"{one_body_integrals.shape}"
        )
    n_orbitals = one_body_integrals.shape[0]
    if two_body_integrals.shape != (n_orbitals,) * 4:
        raise ValueError(
            f"two_body_integrals must have shape {(n_orbitals,) * 4} to match "
            f"one_body_integrals (n_orbitals={n_orbitals}), got "
            f"{two_body_integrals.shape}"
        )
    n_spin_orbitals = 2 * n_orbitals
    if one_rdm.shape != (n_spin_orbitals,) * 2:
        raise ValueError(
            f"one_rdm must have shape {(n_spin_orbitals,) * 2} (spin-orbital "
            f"basis, 2 * n_orbitals with n_orbitals={n_orbitals}), got "
            f"{one_rdm.shape}"
        )
    if two_rdm.shape != (n_spin_orbitals,) * 4:
        raise ValueError(
            f"two_rdm must have shape {(n_spin_orbitals,) * 4} (spin-orbital "
            f"basis, 2 * n_orbitals with n_orbitals={n_orbitals}), got "
            f"{two_rdm.shape}"
        )
    if n_electrons % 2 != 0:
        raise ValueError(
            f"optimize_orbitals is restricted (closed-shell) -- n_electrons "
            f"must be even, got {n_electrons}"
        )
    nocc = n_electrons // 2
    nvirt = n_orbitals - nocc
    if nocc <= 0 or nvirt <= 0:
        raise ValueError(
            f"optimize_orbitals needs at least one occupied and one virtual "
            f"spatial orbital (got n_orbitals={n_orbitals}, n_electrons={n_electrons})"
        )
    occ = list(range(nocc))
    virt = list(range(nocc, n_orbitals))

    def energy(params: np.ndarray) -> float:
        kappa = rhf_params_to_matrix(params, n_orbitals, occ, virt)
        rotation = sp.linalg.expm(kappa)
        rotated_obi = general_basis_change(one_body_integrals, rotation, (1, 0), transpose=False)
        rotated_tbi = general_basis_change(
            two_body_integrals, rotation, (1, 1, 0, 0), transpose=False
        )
        hamiltonian = generate_hamiltonian(rotated_obi, rotated_tbi, nuclear_repulsion)
        return _energy_from_rdms(
            hamiltonian.one_body_tensor,
            hamiltonian.two_body_tensor,
            hamiltonian.constant,
            one_rdm,
            two_rdm,
        )

    if initial_guess is None:
        init_params = np.zeros(nocc * nvirt)
    else:
        init_params = np.asarray(initial_guess).flatten()
        if init_params.size != nocc * nvirt:
            raise ValueError(
                f"initial_guess has {init_params.size} parameters, expected "
                f"nocc * nvirt = {nocc} * {nvirt} = {nocc * nvirt}"
            )

    sp_optimizer_options = {'disp': verbose}
    if sp_options is not None:
        sp_optimizer_options.update(sp_options)

    return sp.optimize.minimize(energy, init_params, method=method, options=sp_optimizer_options)
