# coverage:ignore
import os
from uuid import uuid4
import scipy.optimize
import jax.numpy as jnp
from jax.config import config
from jax import jit, grad
import h5py
import numpy
import numpy.random
import numpy.linalg
from scipy.optimize import minimize
from .adagrad import adagrad

# set mkl thread count for numpy einsum/tensordot calls
# leave one CPU un used  so we can still access this computer
os.environ["MKL_NUM_THREADS"] = "{}".format(os.cpu_count() - 1)
config.update("jax_enable_x64", True)


def thc_objective_jax(xcur, norb, nthc, eri):
    """
    Loss function for THC factorization using jax numpy

    0.5 sum_{pqrs}(eri(pqrs) - G(pqrs))^{2}

    G(pqrs) = sum_{uv}X_{u,p}X_{u,q}Z_{uv}X_{v,r}X_{v,s}

    :param xcur: Current parameters for eta and Z
    :param norb: number of orbitals
    :param nthc: thc-basis dimension
    :param eri: two-electron repulsion integrals in chemist notation
    :return:
    """
    etaPp = xcur[: norb * nthc].reshape(nthc, norb)  # leaf tensor  nthc x norb
    MPQ = xcur[norb * nthc : norb * nthc + nthc * nthc].reshape(nthc, nthc)  # central tensor

    CprP = jnp.einsum("Pp,Pr->prP", etaPp, etaPp)  # this is einsum('mp,mq->pqm', etaPp, etaPp)
    Iapprox = jnp.einsum('pqU,UV,rsV->pqrs', CprP, MPQ, CprP, optimize=[(0, 1), (0, 1)])
    deri = eri - Iapprox
    res = 0.5 * jnp.sum((deri) ** 2)
    return res


def thc_objective_grad_jax(xcur, norb, nthc, eri):
    """
    Gradient for THC least-squares objective jax compatible

    :param xcur: Current parameters for eta and Z
    :param norb: number of orbitals
    :param nthc: thc-basis dimension
    :param eri: two-electron repulsion integrals in chemist notation
    :param verbose: optional (False) for print iteration residual and inf norm
    """
    etaPp = xcur[: norb * nthc].reshape(nthc, norb)  # leaf tensor  nthc x norb
    MPQ = xcur[norb * nthc : norb * nthc + nthc * nthc].reshape(nthc, nthc)  # central tensor

    # m indexes the nthc and p,q,r,s are orbital indices
    CprP = jnp.einsum("Pp,Pr->prP", etaPp, etaPp)  # this is einsum('mp,mq->pqm', etaPp, etaPp)
    Iapprox = jnp.einsum('pqU,UV,rsV->pqrs', CprP, MPQ, CprP, optimize=[(0, 1), (0, 1)])
    deri = eri - Iapprox

    # O(norb^5)
    dL_dZab = -jnp.einsum('pqrs,pqA,rsB->AB', deri, CprP, CprP, optimize=[(0, 1), (0, 1)])

    # O(norb^5)
    dL_dX_GT = -2 * jnp.einsum(
        'Tqrs,Gq,Gv,rsv->GT', deri, etaPp, MPQ, CprP, optimize=[(0, 3), (1, 2), (0, 1)]
    )

    dL_dX_GT -= 2 * jnp.einsum(
        'pqTs,pqu,uG,Gs->GT', deri, CprP, MPQ, etaPp, optimize=[(0, 1), (0, 2), (0, 1)]
    )

    return jnp.hstack((dL_dX_GT.ravel(), dL_dZab.ravel()))


def thc_objective(xcur, norb, nthc, eri, verbose=False):
    """
    Loss function for THC factorization

    0.5 sum_{pqrs}(eri(pqrs) - G(pqrs))^{2}

    G(pqrs) = sum_{uv}X_{u,p}X_{u,q}Z_{uv}X_{v,r}X_{v,s}

    :param xcur: Current parameters for eta and Z
    :param norb: number of orbitals
    :param nthc: thc-basis dimension
    :param eri: two-electron repulsion integrals in chemist notation
    :param verbose: optional (False) for print iteration residual and inf norm
    :return:
    """
    etaPp = xcur[: norb * nthc].reshape(nthc, norb)  # leaf tensor  nthc x norb
    MPQ = xcur[norb * nthc : norb * nthc + nthc * nthc].reshape(nthc, nthc)  # central tensor

    CprP = numpy.einsum("Pp,Pr->prP", etaPp, etaPp)  # this is einsum('mp,mq->pqm', etaPp, etaPp)
    Iapprox = numpy.einsum(
        'pqU,UV,rsV->pqrs', CprP, MPQ, CprP, optimize=['einsum_path', (0, 1), (0, 1)]
    )
    deri = eri - Iapprox
    res = 0.5 * numpy.sum((deri) ** 2)

    if verbose:
        print("res, max, lambda = {}, {}".format(res, numpy.max(numpy.abs(deri))))

    return res


def thc_objective_regularized(xcur, norb, nthc, eri, penalty_param, verbose=False):
    """
    Loss function for THC factorization

    0.5 sum_{pqrs}(eri(pqrs) - G(pqrs))^{2}

    G(pqrs) = sum_{uv}X_{u,p}X_{u,q}Z_{uv}X_{v,r}X_{v,s}

    :param xcur: Current parameters for eta and Z
    :param norb: number of orbitals
    :param nthc: thc-basis dimension
    :param eri: two-electron repulsion integrals in chemist notation
    :param verbose: optional (False) for print iteration residual and inf norm
    :return:
    """
    etaPp = xcur[: norb * nthc].reshape(nthc, norb)  # leaf tensor  nthc x norb
    MPQ = xcur[norb * nthc : norb * nthc + nthc * nthc].reshape(nthc, nthc)  # central tensor

    CprP = jnp.einsum("Pp,Pr->prP", etaPp, etaPp)  # this is einsum('mp,mq->pqm', etaPp, etaPp)
    Iapprox = jnp.einsum('pqU,UV,rsV->pqrs', CprP, MPQ, CprP, optimize=[(0, 1), (0, 1)])
    deri = eri - Iapprox

    SPQ = etaPp.dot(etaPp.T)  # (nthc x norb)  x (norb x nthc) -> (nthc  x nthc) metric
    cP = jnp.diag(jnp.diag(SPQ))  # grab diagonal elements. equivalent to np.diag(np.diagonal(SPQ))
    # no sqrts because we have two normalized THC vectors (index by mu and nu)
    # on each side.
    MPQ_normalized = cP.dot(MPQ).dot(cP)  # get normalized zeta in Eq. 11 & 12

    lambda_z = jnp.sum(jnp.abs(MPQ_normalized)) * 0.5

    res = 0.5 * jnp.sum((deri) ** 2) + penalty_param * (lambda_z**2)

    if verbose:
        print("res, max, lambda**2 = {}, {}".format(res, lambda_z**2))

    return res


def thc_objective_grad(xcur, norb, nthc, eri, verbose=False):
    """
    Gradient for THC least-squares objective

    :param xcur: Current parameters for eta and Z
    :param norb: number of orbitals
    :param nthc: thc-basis dimension
    :param eri: two-electron repulsion integrals in chemist notation
    :param verbose: optional (False) for print iteration residual and inf norm
    """
    etaPp = numpy.array(xcur[: norb * nthc]).reshape(nthc, norb)  # leaf tensor  nthc x norb
    MPQ = numpy.array(xcur[norb * nthc : norb * nthc + nthc * nthc]).reshape(
        nthc, nthc
    )  # central tensor

    # m indexes the nthc and p,q,r,s are orbital indices
    CprP = numpy.einsum("Pp,Pr->prP", etaPp, etaPp)  # this is einsum('mp,mq->pqm', etaPp, etaPp)
    Iapprox = numpy.einsum(
        'pqU,UV,rsV->pqrs', CprP, MPQ, CprP, optimize=['einsum_path', (0, 1), (0, 1)]
    )
    deri = eri - Iapprox
    res = 0.5 * numpy.sum((deri) ** 2)

    if verbose:
        print("res, max, lambda = {}, {}".format(res, numpy.max(numpy.abs(deri))))

    # O(norb^5)
    dL_dZab = -numpy.einsum(
        'pqrs,pqA,rsB->AB', deri, CprP, CprP, optimize=['einsum_path', (0, 1), (0, 1)]
    )
    # O(norb^5)
    dL_dX_GT = -2 * numpy.einsum(
        'Tqrs,Gq,Gv,rsv->GT',
        deri,
        etaPp,
        MPQ,
        CprP,
        optimize=['einsum_path', (0, 3), (1, 2), (0, 1)],
    )

    dL_dX_GT -= 2 * numpy.einsum(
        'pqTs,pqu,uG,Gs->GT',
        deri,
        CprP,
        MPQ,
        etaPp,
        optimize=['einsum_path', (0, 1), (0, 2), (0, 1)],
    )

    return numpy.hstack((dL_dX_GT.ravel(), dL_dZab.ravel()))


def thc_objective_and_grad(xcur, norb, nthc, eri, verbose=False):
    """
    Loss function for THC factorization

    0.5 sum_{pqrs}(eri(pqrs) - G(pqrs))^{2}

    G(pqrs) = sum_{uv}X_{u,p}X_{u,q}Z_{uv}X_{v,r}X_{v,s}

    :param xcur: Current parameters for eta and Z
    :param norb: number of orbitals
    :param nthc: thc-basis dimension
    :param eri: two-electron repulsion integrals in chemist notation
    :param verbose: optional (False) for print iteration residual and inf norm
    :return:
    """
    etaPp = xcur[: norb * nthc].reshape(nthc, norb)  # leaf tensor  nthc x norb
    MPQ = xcur[norb * nthc : norb * nthc + nthc * nthc].reshape(nthc, nthc)  # central tensor
    CprP = numpy.einsum("Pp,Pr->prP", etaPp, etaPp)  # this is einsum('mp,mq->pqm', etaPp, etaPp)

    Iapprox = numpy.einsum(
        'pqU,UV,rsV->pqrs', CprP, MPQ, CprP, optimize=['einsum_path', (0, 1), (0, 1)]
    )
    deri = eri - Iapprox
    res = 0.5 * numpy.sum((deri) ** 2)
    # O(norb^5)
    dL_dZab = -numpy.einsum(
        'pqrs,pqA,rsB->AB', deri, CprP, CprP, optimize=['einsum_path', (0, 1), (0, 1)]
    )
    # O(norb^4 * nthc)
    dL_dX_GT = -2 * numpy.einsum(
        'Tqrs,Gq,Gv,rsv->GT',
        deri,
        etaPp,
        MPQ,
        CprP,
        optimize=['einsum_path', (0, 3), (1, 2), (0, 1)],
    )

    dL_dX_GT -= 2 * numpy.einsum(
        'pqTs,pqu,uG,Gs->GT',
        deri,
        CprP,
        MPQ,
        etaPp,
        optimize=['einsum_path', (0, 1), (0, 2), (0, 1)],
    )

    return res, numpy.hstack((dL_dX_GT.ravel(), dL_dZab.ravel()))


def cp_ls_cholesky_factor_objective(beta_gamma, norb, nthc, cholesky_factor, calcgrad=False):
    """cholesky_factor is reshaped into (norb, norb, num_cholesky)

    Cholesky factor B_{ab,x}

    Lst sq fit obj ||B_{ab,x} - sum_{r}beta_{a,x}beta_{b,x}gamma_{ab,x}||

    This function provides the objective function value and gradient with
      respect to beta and gamma
    """
    # compute objective
    num_cholfactors = cholesky_factor.shape[-1]
    beta_bR = beta_gamma[: norb * nthc].reshape((norb, nthc))
    gamma_yR = beta_gamma[norb * nthc : norb * nthc + nthc * num_cholfactors].reshape(
        (num_cholfactors, nthc)
    )
    beta_abR = numpy.einsum('aR,bR->abR', beta_bR, beta_bR)
    chol_approx = numpy.einsum('abR,XR->abX', beta_abR, gamma_yR)
    delta = cholesky_factor - chol_approx
    fval = 0.5 * numpy.sum((delta) ** 2)

    if calcgrad:
        # compute grad
        # \partial O / \partial beta_{c,s}
        grad_beta = -2 * numpy.einsum(
            'Cbx,bS,xS->CS', delta, beta_bR, gamma_yR, optimize=['einsum_path', (0, 2), (0, 1)]
        )
        grad_gamma = -numpy.einsum(
            'abY,aS,bS->YS', delta, beta_bR, beta_bR, optimize=['einsum_path', (1, 2), (0, 1)]
        )
        grad = numpy.hstack((grad_beta.ravel(), grad_gamma.ravel()))
        return fval, grad
    else:
        return fval
