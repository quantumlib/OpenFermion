from typing import List, Optional, Union
from itertools import product
from openfermion.third_party.representability._dualbasis import \
    DualBasisElement, DualBasis
from openfermion.utils._rdm_mapping_functions import kronecker_delta


def tpdm_trace_constraint(dim: int, normalization: float) -> DualBasisElement:
    """
    Generate the trace constraint on the 2-PDM

    Args:
        dim:  Dimension of the spin-orbital single-particle Hilbert space
        normalization: Desired trace value.

    Returns: A DualBasisElement (Row of the constraint matrix) encoding
             the trace constraint on the 2-RDM.
    """
    tensor_elements = [(i, j, i, j) for i, j in product(range(dim), repeat=2)]
    tensor_names = ['cckk'] * (dim**2)
    tensor_coeffs = [1.0] * (dim**2)
    bias = 0
    return DualBasisElement(tensor_names=tensor_names,
                            tensor_elements=tensor_elements,
                            tensor_coeffs=tensor_coeffs,
                            bias=bias,
                            scalar=normalization)


def tpdm_antisymmetry_constraint(dim: int) -> DualBasis:
    """
    The dual basis elements representing the antisymmetry constraints

    :param dim: spinless Fermion basis rank
    :return: the dual basis of antisymmetry_constraints
    :rtype: DualBasis
    """
    # dual_basis = DualBasis()
    dbe_list = []
    for p, q, r, s in product(range(dim), repeat=4):
        if p * dim + q <= r * dim + s:
            if p < q and r < s:
                tensor_elements = [
                    tuple(indices) for indices in _coord_generator(p, q, r, s)
                ]
                tensor_names = ['cckk'] * len(tensor_elements)
                tensor_coeffs = [0.5] * len(tensor_elements)
                dbe = DualBasisElement()
                for n, e, c in zip(tensor_names, tensor_elements,
                                   tensor_coeffs):
                    dbe.add_element(n, e, c)

                # dual_basis += dbe
                dbe_list.append(dbe)

    return DualBasis(elements=dbe_list)


def tpdm_to_opdm_mapping(dim: int,
                         normalization: Union[float, int]) -> DualBasis:
    """
    Construct the DualBasis for mapping of the tpdm to the opdm

    Args:
        dim: dimension of the spin-orbital basis.
        normalization: Scalar for mapping tpdm to opdm.  Generally, this is
                       1 / (N - 1) where N is the number of electrons.
    Returns:
        DualBasis for all dim^2
    """
    db_basis = DualBasis()
    for i in range(dim):
        for j in range(i, dim):
            dbe = DualBasisElement()
            # contract over tpdm terms
            for r in range(dim):
                # duplicate entries get summed in DualBasisElement
                dbe.add_element('cckk', (i, r, j, r), 0.5)
                dbe.add_element('cckk', (j, r, i, r), 0.5)

            # opdm terms
            dbe.add_element('ck', (i, j), -0.5 * normalization)
            dbe.add_element('ck', (j, i), -0.5 * normalization)
            dbe.simplify()
            db_basis += dbe

    return db_basis


def opdm_to_ohdm_mapping(dim: int) -> DualBasis:
    """
    Map the ck to kc

    D1 + Q1 = I

    Args:
        dim: dimension of the spin-orbital basis
    Returns:
        DualBasis for the 1-RDM representability constraint
    """
    dbe_list = []
    for i in range(dim):
        for j in range(i, dim):
            dbe = DualBasisElement()
            if i != j:
                dbe.add_element('ck', (i, j), 0.5)
                dbe.add_element('ck', (j, i), 0.5)
                dbe.add_element('kc', (j, i), 0.5)
                dbe.add_element('kc', (i, j), 0.5)
                dbe.dual_scalar = 0.0
            else:
                dbe.add_element('ck', (i, j), 1.0)
                dbe.add_element('kc', (i, j), 1.0)
                dbe.dual_scalar = 1.0

            # db += dbe
            dbe_list.append(dbe)

    return DualBasis(elements=dbe_list)  # db


def sz_constraint(dim: int, sz: Union[float, int]) -> DualBasis:
    """
    Constraint on the 1-RDM

    Args:
        dim: dimension of the spin-orbital basis.
        sz: expectation value of the magnetic quantum number.

    Returns:
        DualBasis
    """
    dbe = DualBasisElement()
    for i in range(dim // 2):
        dbe.add_element('ck', (2 * i, 2 * i), 0.5)
        dbe.add_element('ck', (2 * i + 1, 2 * i + 1), -0.5)
    dbe.dual_scalar = sz
    return DualBasis(elements=[dbe])


def na_constraint(dim: int, na: Union[float, int]) -> DualBasis:
    """
    Constraint the trace of the alpha block of the opdm to equal the number
    of spin-up electrons

    Args:
        dim: Dimension of the spin-orbital basis.
        na: Number of spin up electrons

    Returns:
        DualBasis representing the cosntraint that is length 1
    """

    dbe = DualBasisElement()
    for i in range(dim // 2):
        dbe.add_element('ck', (2 * i, 2 * i), 1.0)
    dbe.dual_scalar = na
    return DualBasis(elements=[dbe])


def nb_constraint(dim, nb):
    """
    Constraint the trace of the alpha block of the opdm to equal the number
    of spin-down electrons

    Args:
        dim: Dimension of the spin-orbital basis.
        na: Number of spin down electrons

    Returns:
        DualBasis representing the cosntraint that is length 1
    """
    dbe = DualBasisElement()
    for i in range(dim // 2):
        dbe.add_element('ck', (2 * i + 1, 2 * i + 1), 1.0)
    dbe.dual_scalar = nb
    return DualBasis(elements=[dbe])


def tpdm_to_thdm_mapping(dim: int) -> DualBasis:
    """
    Generate the dual basis elements for a mapping of the 2-RDM to the
    two-hole-RDM

    Args:
        dim: Dimension of the spin-orbital basis

    Returns:
        DualBasis representing the equality constraint.
    """
    dbe_list = []

    def d2q2element(p: int, q: int, r: int, s: int, factor: Union[float, int])\
            -> DualBasisElement:
        """
        Build the dual basis element for symmetric form of 2-marginal

        :param p: tensor index
        :param q: tensor index
        :param r: tensor index
        :param s: tensor index
        :param factor: scaling coeff for a symmetric constraint
        :return: the dual basis of the mapping
        """
        dbe = DualBasisElement()
        dbe.add_element('cckk', (p, q, r, s), -1.0 * factor)
        dbe.add_element('kkcc', (r, s, p, q), +1.0 * factor)
        if q == s:
            dbe.add_element('ck', (p, r), factor)
        if p == r:
            dbe.add_element('ck', (q, s), factor)
        if q == r:
            dbe.add_element('ck', (p, s), -1. * factor)
        if p == s:
            dbe.add_element('ck', (q, r), -1. * factor)

        dbe.dual_scalar = (
            kronecker_delta(q, s) * kronecker_delta(p, r) -
            kronecker_delta(q, r) * kronecker_delta(p, s)) * factor
        return dbe

    for i, j, k, l in product(range(dim), repeat=4):
        if i * dim + j <= k * dim + l:
            db_element = d2q2element(i, j, k, l, 0.5)
            db_element_2 = d2q2element(k, l, i, j, 0.5)
            dbe_list.append(db_element.join_elements(db_element_2))

    return DualBasis(elements=dbe_list)


def tpdm_to_phdm_mapping(dim: int) -> DualBasis:
    """

    Args:
        dim: Dimension of the spin-orbital basis

    Returns:
        DualBasis representing the equality constraint
    """
    dbe_list = []

    def g2d2map(p: int,
                q: int,
                r: int,
                s: int,
                factor: Optional[Union[float, int]] = 1) -> DualBasisElement:
        """
        Build the dual basis element for a symmetric 2-marginal

        :param p: tensor index
        :param q: tensor index
        :param r: tensor index
        :param s: tensor index
        :param factor: weighting of the element
        :return: the dual basis element
        """
        dbe = DualBasisElement()
        if q == s:
            dbe.add_element('ck', (p, r), -1. * factor)
        dbe.add_element('ckck', (p, s, r, q), 1.0 * factor)
        dbe.add_element('cckk', (p, q, r, s), 1.0 * factor)
        dbe.dual_scalar = 0

        return dbe

    for i, j, k, l in product(range(dim), repeat=4):
        if i * dim + j <= k * dim + l:
            db_element = g2d2map(i, j, k, l, factor=0.5)
            db_element_2 = g2d2map(k, l, i, j, factor=0.5)
            dbe_list.append(db_element.join_elements(db_element_2))

    return DualBasis(elements=dbe_list)


def spin_orbital_linear_constraints(dim: int,
                                    num_alpha: Union[int, float],
                                    num_beta: Union[int, float],
                                    constraint_list: List[str],
                                    sz: Optional[Union[None, float, int]] = None
                                   ) -> DualBasis:
    """
    Construct dual basis constraints for 2-positivity in a spin-orbital
    basis

    Args:
        dim: Dimension of the spin-orbital basis
        num_alpha: number of spin-up electrons.
        num_beta: number of spin-down electrons.
        constraint_list: list of which matrices to include in the 2-pos set.
        sz: target magnetic quantum number.

    Returns:
        DualBasis element.
    """
    N = num_alpha + num_beta
    dual_basis = DualBasis()
    if 'cckk' in constraint_list:
        # Natural constraints on particle-conserving 2-RDMs
        # trace constraint on D2
        dual_basis += tpdm_trace_constraint(dim, N * (N - 1))
        # antisymmetry constraint
        dual_basis += tpdm_antisymmetry_constraint(dim)

    if 'ck' in constraint_list:
        dual_basis += tpdm_to_opdm_mapping(dim, N - 1)
        if sz is not None:
            dual_basis += sz_constraint(dim, sz)

        dual_basis += na_constraint(dim, num_alpha)
        dual_basis += nb_constraint(dim, num_beta)

        if 'kc' in constraint_list:
            dual_basis += opdm_to_ohdm_mapping(dim)

    # d2 -> q2
    if 'kkcc' in constraint_list:
        print('tqdm constraints')
        dual_basis += tpdm_to_thdm_mapping(dim)

    # d2 -> g2
    if "ckck" in constraint_list:
        print('phdm constraints')
        dual_basis += tpdm_to_phdm_mapping(dim)

    return dual_basis


def _coord_generator(i, j, k, l):
    """
    Generator for equivalent spin-orbital indices given a set of four
    spin-orbitals. Indices are in chemist notation so for real-valued chemical
    spinless Hamiltonians the integrals have 8-fold symmetry.

    using this function and iterating over the following:
        i >= j && k >= l && ij >= kl
        where ij = i*(i + 1)/2 + j
              kl = k*(k + 1)/2 + l

    spatial real-values
    i, j, k, l
    j, i, k, l
    i, j, l, k
    j, i, l, k
    k, l, i, j
    k, l, j, i
    l, k, i, j
    l, k, j, i
    """
    unique_set = {(i, j, k, l), (j, i, k, l), (i, j, l, k), (j, i, l, k),
                  (k, l, i, j), (k, l, j, i), (l, k, i, j), (l, k, j, i)}
    for index_element in unique_set:
        yield index_element
