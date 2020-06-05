from itertools import product
import os
import numpy as np
from openfermion.third_party.representability.constraints.spin_orbital_2pos_constraints import (  # pylint: disable=line-too-long
    tpdm_antisymmetry_constraint, tpdm_trace_constraint, _coord_generator,
    tpdm_to_opdm_mapping, opdm_to_ohdm_mapping, sz_constraint, na_constraint,
    nb_constraint, tpdm_to_thdm_mapping, tpdm_to_phdm_mapping,
    spin_orbital_linear_constraints)
from openfermion.third_party.representability._dualbasis import \
    DualBasisElement, DualBasis
from openfermion.third_party.representability._namedtensor import Tensor
from openfermion.third_party.representability._multitensor import MultiTensor
from openfermion.config import DATA_DIRECTORY
from openfermion import MolecularData
from openfermion.utils import map_two_pdm_to_two_hole_dm, \
    map_two_pdm_to_particle_hole_dm


def test_trace_constraint():
    dbe = tpdm_trace_constraint(4, 10)
    assert dbe.primal_tensors_names == ['cckk'] * 4**2
    assert dbe.primal_elements == [
        (i, j, i, j) for i, j in product(range(4), repeat=2)
    ]
    assert np.isclose(dbe.constant_bias, 0)
    assert np.isclose(dbe.dual_scalar, 10)


def test_coord_generator():
    i, j, k, l = 0, 1, 2, 3
    true_set = {(i, j, k, l), (j, i, k, l), (i, j, l, k), (j, i, l, k),
                (k, l, i, j), (k, l, j, i), (l, k, i, j), (l, k, j, i)}

    assert true_set == set(_coord_generator(i, j, k, l))

    i, j, k, l = 1, 1, 2, 3
    true_set = {(i, j, k, l), (j, i, k, l), (i, j, l, k), (j, i, l, k),
                (k, l, i, j), (k, l, j, i), (l, k, i, j), (l, k, j, i)}
    assert true_set == set(_coord_generator(i, j, k, l))


def test_d2_antisymm():
    db = tpdm_antisymmetry_constraint(4)
    for dbe in db:
        i, j, k, l = dbe.primal_elements[0]
        assert len(list(_coord_generator(i, j, k,
                                         l))) == len(dbe.primal_elements)
        assert np.isclose(dbe.constant_bias, 0)
        assert np.isclose(dbe.dual_scalar, 0)
        assert np.unique(dbe.primal_coeffs)


def test_tpdm_opdm_mapping():
    db = tpdm_to_opdm_mapping(6, 1 / 2)
    for dbe in db:
        assert isinstance(dbe, DualBasisElement)
        assert set(dbe.primal_tensors_names) == {'cckk', 'ck'}
        assert len(dbe.primal_elements) == 7 or len(dbe.primal_elements) == 14
        assert np.isclose(dbe.dual_scalar, 0)
        assert np.isclose(dbe.constant_bias, 0)
        print(vars(dbe))
        for idx, element in enumerate(dbe.primal_elements):
            if len(element) == 4:
                assert element[1] == element[3]
                if element[0] == element[2]:
                    assert np.isclose(dbe.primal_coeffs[idx], 1.0)
                else:
                    assert np.isclose(dbe.primal_coeffs[idx], 0.5)
                assert dbe.primal_coeffs
            elif len(element) == 2:
                gem_idx = [len(x) for x in dbe.primal_elements].index(4)
                assert sorted(element) == sorted([
                    dbe.primal_elements[gem_idx][0],
                    dbe.primal_elements[gem_idx][2]
                ])


def test_opdm_to_ohdm_mapping():
    db = opdm_to_ohdm_mapping(6)
    for dbe in db:
        assert isinstance(dbe, DualBasisElement)
        assert set(dbe.primal_tensors_names) == {'ck', 'kc'}
        if len(dbe.primal_tensors_names) == 4:
            assert np.allclose(dbe.primal_coeffs, 0.5)
            assert np.isclose(dbe.dual_scalar, 0.0)
        elif len(dbe.primal_tensors_names) == 2:
            assert np.allclose(dbe.primal_coeffs, 1.0)
            assert np.isclose(dbe.dual_scalar, 1.0)


def test_sz_constraint():
    db = sz_constraint(6, 3.5)
    test_dbe = db.elements[0]
    for i in range(3):
        np.isclose(test_dbe.primal_coeffs[2 * i], 0.5)
        np.isclose(test_dbe.primal_coeffs[2 * i + 1], 0.5)
        assert test_dbe.primal_tensors_names[2 * i] == 'ck'
        assert test_dbe.primal_tensors_names[2 * i + 1] == 'ck'
    assert np.isclose(test_dbe.dual_scalar, 3.5)


def test_na_constraint():
    db = na_constraint(6, 3)
    assert len(db) == 1
    test_dbe = db.elements[0]
    for i in range(3):
        np.isclose(test_dbe.primal_coeffs[i], 1.0)
        assert test_dbe.primal_elements[i] == (2 * i, 2 * i)
    assert np.isclose(test_dbe.dual_scalar, 3)


def test_nb_constraint():
    db = nb_constraint(6, 3)
    assert len(db) == 1
    test_dbe = db.elements[0]
    for i in range(3):
        np.isclose(test_dbe.primal_coeffs[i], 1.0)
        assert test_dbe.primal_elements[i] == (2 * i + 1, 2 * i + 1)
    assert np.isclose(test_dbe.dual_scalar, 3)


def test_d2_to_q2():
    # this should test for correctness
    filename = os.path.join(DATA_DIRECTORY, "H1-Li1_sto-3g_singlet_1.45.hdf5")
    molecule = MolecularData(filename=filename)
    molecule.load()
    opdm = molecule.fci_one_rdm
    tpdm = molecule.fci_two_rdm
    thdm = map_two_pdm_to_two_hole_dm(tpdm, opdm)

    db = tpdm_to_thdm_mapping(6)
    topdm = Tensor(tensor=opdm, name='ck')
    ttpdm = Tensor(tensor=np.einsum('ijlk', tpdm), name='cckk')
    tthdm = Tensor(tensor=np.einsum('ijlk', thdm), name='kkcc')
    mt = MultiTensor(tensors=[topdm, ttpdm, tthdm], dual_basis=db)
    A, _, b = mt.synthesize_dual_basis()
    vec_rdms = mt.vectorize_tensors()
    assert np.isclose(np.linalg.norm(A.dot(vec_rdms) - b), 0)


def test_d2_to_g2():
    # this should test for correctness
    filename = os.path.join(DATA_DIRECTORY, "H1-Li1_sto-3g_singlet_1.45.hdf5")
    molecule = MolecularData(filename=filename)
    molecule.load()
    opdm = molecule.fci_one_rdm
    tpdm = molecule.fci_two_rdm
    phdm = map_two_pdm_to_particle_hole_dm(tpdm, opdm)

    db = tpdm_to_phdm_mapping(molecule.n_qubits)
    topdm = Tensor(tensor=opdm, name='ck')
    ttpdm = Tensor(tensor=np.einsum('ijlk', tpdm), name='cckk')
    tphdm = Tensor(tensor=np.einsum('ijlk', phdm), name='ckck')
    mt = MultiTensor(tensors=[topdm, ttpdm, tphdm], dual_basis=db)
    A, _, b = mt.synthesize_dual_basis()
    vec_rdms = mt.vectorize_tensors()
    assert np.isclose(np.linalg.norm(A.dot(vec_rdms) - b), 0)


def test_spin_orbital_dual_basis_construction():
    db = spin_orbital_linear_constraints(
        dim=6,
        num_alpha=3,
        num_beta=3,
        constraint_list=['ck', 'kc', 'cckk', 'kkcc', 'ckck'],
        sz=0)
    assert isinstance(db, DualBasis)
