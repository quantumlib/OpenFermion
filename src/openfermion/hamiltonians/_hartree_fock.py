"""
Module performs gradient based RHF, [WIP] UHF, [WIP] GHF

Module needs AO integrals
"""
# pylint: disable=C
from typing import Callable, Dict, List, Optional, Tuple, Union
from itertools import product
import numpy as np
import scipy as sp
from scipy.optimize.optimize import OptimizeResult
from openfermion.ops import InteractionOperator, InteractionRDM, general_basis_change
from openfermion.utils import wedge


def get_matrix_of_eigs(w: np.ndarray) -> np.ndarray:
    """
    Transform the eigenvalues into a matrix corresponding
    to summing the adjoint rep.

    Args:
        w: eigenvalues of C-matrix

    Returns: new array of transformed eigenvalues
    """
    transform_eigs = np.zeros((w.shape[0], w.shape[0]), dtype=np.complex128)
    for i, j in product(range(w.shape[0]), repeat=2):
        if np.isclose(abs(w[i] - w[j]), 0):
            transform_eigs[i, j] = 1
        else:
            transform_eigs[i, j] = (np.exp(1j *
                                           (w[i] - w[j])) - 1) / (1j *
                                                                  (w[i] - w[j]))
    return transform_eigs


class InputError(Exception):
    pass


class HartreeFockFunctional():
    """
    Implementation of the objective function code for Restricted Hartree-Fock

    The object transforms a variety of input types into the appropriate output.
    It does this by analyzing the type and size of the input based on its
    knowledge of each type.
    """

    def __init__(self,
                 *,
                 one_body_integrals: np.ndarray,
                 two_body_integrals: np.ndarray,
                 overlap: np.ndarray,
                 n_electrons: int,
                 model='rhf',
                 nuclear_repulsion: Optional[float] = 0.,
                 initial_orbitals: Optional[Union[None, Callable]] = None):
        """
        Initialize functional

        Args:
            one_body_integrals: integrals in the atomic orbital basis for the
                                one-body potential.
            two_body_integrals: integrals in the  atomic obrital basis for the
                                two-body potential ordered according to
                                phi_{p}(r1)^{*}phi_{q}^{*}(r2) x
                                phi_{r}(r2)phi_{s}(r1)
            overlap:  overlap integrals in the atomic orbital basis
            n_electrons:  number of electrons total
            model: Optional flag for performing restricted-, unrestricted-,
                   or generalized- hartree-fock.
            nuclear_repulsion: Optional nuclear repulsion term.  Energy is
                               shifted by this amount. default is 0.
            initial_orbitals:  Method for producing the initial orbitals from
                               the atomic orbitals. Default is defining the
                               core orbitals.
        """
        if model not in ['rhf', 'uhf', 'ghf']:
            raise InputError("{} is not rhf, uhf, or ghf".format(model))
        self.model = model
        self.obi = one_body_integrals
        self.tbi = two_body_integrals
        self.overlap = overlap
        self.num_orbitals = one_body_integrals.shape[0]
        self.num_electrons = n_electrons
        self.constant_offset = nuclear_repulsion
        self.hamiltonian = None

        self.nocc = None
        self.nvirt = None
        self.occ = None
        self.virt = None
        if model == 'rhf':
            self.nocc = self.num_electrons // 2
            self.nvirt = self.num_orbitals - self.nocc
            self.occ = list(range(self.nocc))
            self.virt = list(range(self.nocc, self.nocc + self.nvirt))
        elif model == 'uhf' or model == 'ghf':
            self.nocc = self.num_electrons
            self.nvirt = 2 * self.num_orbitals - self.nocc
            self.occ = list(range(self.nocc))
            self.virt = list(range(self.nocc, self.nocc + self.nvirt))

        if initial_orbitals is None:
            # use core orbitals
            _, core_orbs = sp.linalg.eigh(one_body_integrals, b=overlap)

            molecular_hamiltonian = generate_hamiltonian(
                one_body_integrals=general_basis_change(self.obi, core_orbs,
                                                        (1, 0)),
                two_body_integrals=general_basis_change(self.tbi, core_orbs,
                                                        (1, 1, 0, 0)),
                constant=self.constant_offset)
            self.hamiltonian = molecular_hamiltonian
        else:
            self.hamiltonian = initial_orbitals(self.obi, self.tbi,
                                                self.num_electrons)

    def rdms_from_rhf_opdm(self, opdm_aa: np.ndarray) -> InteractionRDM:
        """
        Generate spin-orbital InteractionRDM object from the alpha-spin
        opdm.

        Args:
            opdm_aa: single spin sector of the 1-particle denstiy matrix

        Returns:  InteractionRDM object for full spin-orbital 1-RDM and 2-RDM
        """

        opdm = np.zeros((2 * self.num_orbitals, 2 * self.num_orbitals),
                        dtype=np.complex128)
        opdm[::2, ::2] = opdm_aa
        opdm[1::2, 1::2] = opdm_aa
        tpdm = wedge(opdm, opdm, (1, 1), (1, 1))
        rdms = InteractionRDM(opdm, 2 * tpdm)
        return rdms

    def energy_from_rhf_opdm(self, opdm_aa: np.ndarray) -> float:
        """
        Compute the energy given a spin-up opdm

        Args:
            opdm_aa: spin-up opdm.  Should be an n x n matrix where n is
                     the number of spatial orbitals

        Returns: RHF energy
        """
        rdms = self.rdms_from_rhf_opdm(opdm_aa)
        return rdms.expectation(self.hamiltonian).real

    def rhf_global_gradient(self, params: np.ndarray, alpha_opdm: np.ndarray):
        """
        Compute rhf global gradient

        Args:
            params: rhf-parameters for rotation matrix.
            alpha_opdm: 1-RDM corresponding to results of basis rotation
                        parameterized by `params'.

        Returns: gradient vector the same size as the input `params'
        """
        opdm = np.zeros((2 * self.num_orbitals, 2 * self.num_orbitals),
                        dtype=np.complex128)
        opdm[::2, ::2] = alpha_opdm
        opdm[1::2, 1::2] = alpha_opdm
        tpdm = 2 * wedge(opdm, opdm, (1, 1), (1, 1))

        # now go through and generate all the necessary Z, Y, Y_kl matrices
        kappa_matrix = rhf_params_to_matrix(params,
                                            len(self.occ) + len(self.virt),
                                            self.occ, self.virt)
        kappa_matrix_full = np.kron(kappa_matrix, np.eye(2))
        w_full, v_full = np.linalg.eigh(
            -1j * kappa_matrix_full)  # so that kappa = i U lambda U^
        eigs_scaled_full = get_matrix_of_eigs(w_full)

        grad = np.zeros(self.nocc * self.nvirt, dtype=np.complex128)
        # kdelta = np.eye(2 * self.num_orbitals)

        # NOW GENERATE ALL TERMS ASSOCIATED WITH THE GRADIENT!!!!!!
        for p in range(self.nocc * self.nvirt):
            grad_params = np.zeros_like(params)
            grad_params[p] = 1
            Y = rhf_params_to_matrix(grad_params,
                                     len(self.occ) + len(self.virt), self.occ,
                                     self.virt)
            Y_full = np.kron(Y, np.eye(2))

            # Now rotate Y int othe basis that diagonalizes Z
            Y_kl_full = v_full.conj().T.dot(Y_full).dot(v_full)
            # now rotate Y_{kl} * (exp(i(l_{k} - l_{l})) - 1) / (i(l_{k} - l_{l}))
            # into the original basis
            pre_matrix_full = v_full.dot(eigs_scaled_full * Y_kl_full).dot(
                v_full.conj().T)

            grad_expectation = -1.0 * np.einsum(
                'ab,pa,pb',
                self.hamiltonian.one_body_tensor,
                pre_matrix_full,
                opdm,
                optimize='optimal').real

            grad_expectation += 1.0 * np.einsum(
                'ab,bq,aq',
                self.hamiltonian.one_body_tensor,
                pre_matrix_full,
                opdm,
                optimize='optimal').real

            grad_expectation += 1.0 * np.einsum(
                'ijkl,pi,jpkl',
                self.hamiltonian.two_body_tensor,
                pre_matrix_full,
                tpdm,
                optimize='optimal').real

            grad_expectation += -1.0 * np.einsum(
                'ijkl,pj,ipkl',
                self.hamiltonian.two_body_tensor,
                pre_matrix_full,
                tpdm,
                optimize='optimal').real

            grad_expectation += -1.0 * np.einsum(
                'ijkl,kq,ijlq',
                self.hamiltonian.two_body_tensor,
                pre_matrix_full,
                tpdm,
                optimize='optimal').real

            grad_expectation += 1.0 * np.einsum(
                'ijkl,lq,ijkq',
                self.hamiltonian.two_body_tensor,
                pre_matrix_full,
                tpdm,
                optimize='optimal').real
            grad[p] = grad_expectation

        return grad


def generate_hamiltonian(one_body_integrals: np.ndarray,
                         two_body_integrals: np.ndarray,
                         constant: float,
                         EQ_TOLERANCE: Optional[float] = 1.0E-12
                        ) -> InteractionOperator:
    n_qubits = 2 * one_body_integrals.shape[0]
    # Initialize Hamiltonian coefficients.
    one_body_coefficients = np.zeros((n_qubits, n_qubits))
    two_body_coefficients = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits))
    # Loop through integrals.
    for p in range(n_qubits // 2):
        for q in range(n_qubits // 2):

            # Populate 1-body coefficients. Require p and q have same spin.
            one_body_coefficients[2 * p, 2 * q] = one_body_integrals[p, q]
            one_body_coefficients[2 * p + 1, 2 * q +
                                  1] = one_body_integrals[p, q]
            # Continue looping to prepare 2-body coefficients.
            for r in range(n_qubits // 2):
                for s in range(n_qubits // 2):
                    # Mixed spin
                    two_body_coefficients[2 * p, 2 * q + 1, 2 * r + 1, 2 *
                                          s] = (two_body_integrals[p, q, r, s] /
                                                2.)
                    two_body_coefficients[2 * p + 1, 2 * q, 2 * r, 2 * s +
                                          1] = (two_body_integrals[p, q, r, s] /
                                                2.)

                    # Same spin
                    two_body_coefficients[2 * p, 2 * q, 2 * r, 2 *
                                          s] = (two_body_integrals[p, q, r, s] /
                                                2.)
                    two_body_coefficients[2 * p + 1, 2 * q + 1, 2 * r +
                                          1, 2 * s +
                                          1] = (two_body_integrals[p, q, r, s] /
                                                2.)

    # Truncate.
    one_body_coefficients[
        np.absolute(one_body_coefficients) < EQ_TOLERANCE] = 0.
    two_body_coefficients[
        np.absolute(two_body_coefficients) < EQ_TOLERANCE] = 0.

    # Cast to InteractionOperator class and return.
    molecular_hamiltonian = InteractionOperator(constant, one_body_coefficients,
                                                two_body_coefficients)
    return molecular_hamiltonian


def rhf_params_to_matrix(parameters: np.ndarray,
                         num_orbitals: int,
                         occ: Optional[Union[None, List[int]]] = None,
                         virt: Optional[Union[None, List[int]]] = None
                        ) -> np.ndarray:
    """
    For restricted Hartree-Fock we have nocc * nvirt parameters.  These are
    provided as a list that is ordered by (virtuals) \times (occupied).

    For example, for H4 we have 2 orbitals occupied and 2 virtuals

    occupied = [0, 1]  virtuals = [2, 3]

    parameters = [(v_{0}, o_{0}), (v_{0}, o_{1}), (v_{1}, o_{0}), (v_{1}, o_{1})]
               = [(2, 0), (2, 1), (3, 0), (3, 1)]

    You can think of the tuples of elements of the upper right triangle of the
    antihermitian matrix that specifies the c_{b, i} coefficients.

    coefficient matrix
    [[ c_{0, 0}, -c_{1, 0}, -c_{2, 0}, -c_{3, 0}],
     [ c_{1, 0},  c_{1, 1}, -c_{2, 1}, -c_{3, 1}],
     [ c_{2, 0},  c_{2, 1},  c_{2, 2}, -c_{3, 2}],
     [ c_{3, 0},  c_{3, 1},  c_{3, 2},  c_{3, 3}]]

    Since we are working with only non-redundant operators we know c_{i, i} = 0
    and any c_{i, j} where i and j are both in occupied or both in virtual = 0.

    Args:
        parameters: array of parameters for kappa matrix
        num_orbitals: total number of spatial orbitals
        occ: (Optional) indices for doubly occupied sector
        virt: (Optional) indices for virtual sector

    Returns: np.ndarray kappa matrix
    """
    if occ is None:
        occ = range(num_orbitals // 2)
    if virt is None:
        virt = range(num_orbitals // 2, num_orbitals)

    # check that parameters are a real array
    if not np.allclose(parameters.imag, 0):
        raise ValueError("parameters input must be real valued")

    kappa = np.zeros((len(occ) + len(virt), len(occ) + len(virt)))
    for idx, (v, o) in enumerate(product(virt, occ)):
        kappa[v, o] = parameters[idx].real
        kappa[o, v] = -parameters[idx].real
    return kappa


def rhf_func_generator(rhf_func: HartreeFockFunctional,
                       init_occ_vec: Optional[Union[None, np.ndarray]] = None,
                       get_opdm_func: Optional[bool] = False
                      ) -> Union[Tuple[Callable, Callable, Callable],
                                 Tuple[Callable, Callable, Callable, Callable]]:
    """
    Generate the energy, gradient, and unitary functions

    Args:
        rhf_func: objective function object.
        init_occ_vec: (optional) vector for occupation numbers of
                      the alpha-opdm.
        get_opdm_func: (optional) flag for returning Callable that returns
                       the final opdm.
    Returns: functions for unitary, energy, gradient (in that order)
    """
    if init_occ_vec is None:
        initial_opdm = np.diag([1] * rhf_func.nocc + [0] * rhf_func.nvirt)
    else:
        initial_opdm = np.diag(init_occ_vec)

    def energy(params):
        u = unitary(params)
        final_opdm_aa = u.dot(initial_opdm).dot(np.conjugate(u).T)
        tenergy = rhf_func.energy_from_rhf_opdm(final_opdm_aa)
        return tenergy

    def gradient(params):
        u = unitary(params)
        final_opdm_aa = u.dot(initial_opdm).dot(np.conjugate(u).T)
        return rhf_func.rhf_global_gradient(params, final_opdm_aa).real

    def unitary(params):
        kappa = rhf_params_to_matrix(params, rhf_func.nocc + rhf_func.nvirt,
                                     rhf_func.occ, rhf_func.virt)
        return sp.linalg.expm(kappa)

    def get_opdm(params):
        u = unitary(params)
        return u.dot(initial_opdm).dot(np.conjugate(u).T)

    if get_opdm_func:
        return unitary, energy, gradient, get_opdm
    return unitary, energy, gradient


def rhf_minimization(rhf_object: HartreeFockFunctional,
                     method: Optional[str] = 'CG',
                     initial_guess: Optional[Union[None, np.ndarray]] = None,
                     verbose: Optional[bool] = True,
                     sp_options: Optional[Union[None, Dict]] = None
                    ) -> OptimizeResult:
    """
    Perform Hartree-Fock energy minimization

    Args:
        rhf_object: An instantiation of the HartreeFockFunctional
        method: (optional) scipy optimization method
        initial_guess: (optional) initial rhf parameter vector.  If None
                       zero vector is used.
        verbose: (optional) turn on printing.  This is passed to the
                 scipy 'disp' option.
        sp_options:
    Returns: scipy.optimize result object
    """
    _, energy, gradient = rhf_func_generator(rhf_object)
    if initial_guess is None:
        init_guess = np.zeros(rhf_object.nocc * rhf_object.nvirt)
    else:
        init_guess = np.asarray(initial_guess).flatten()

    sp_optimizer_options = {'disp': verbose}
    if sp_options is not None:
        sp_optimizer_options.update(sp_options)

    return sp.optimize.minimize(energy,
                                init_guess,
                                jac=gradient,
                                method=method,
                                options=sp_optimizer_options)
