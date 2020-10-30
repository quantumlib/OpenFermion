"""
test coverage for performance benchmarks. Check type returns.
"""
from .performance_benchmarks import (
    benchmark_molecular_operator_jordan_wigner,
    benchmark_fermion_math_and_normal_order,
    benchmark_jordan_wigner_sparse,
    benchmark_linear_qubit_operator,
    benchmark_commutator_diagonal_coulomb_operators_2D_spinless_jellium,
    run_molecular_operator_jordan_wigner,
    run_fermion_math_and_normal_order,
    run_jordan_wigner_sparse,
    run_linear_qubit_operator,
    run_diagonal_commutator,
)


def test_run_jw_speed():
    timing = benchmark_molecular_operator_jordan_wigner(2)
    assert isinstance(timing, float)


def test_fermion_normal_order():
    r1, r2 = benchmark_fermion_math_and_normal_order(4, 2, 2)
    assert isinstance(r1, float)
    assert isinstance(r2, float)


def test_jw_sparse():
    r1 = benchmark_jordan_wigner_sparse(4)
    assert isinstance(r1, float)


def test_linear_qop():
    r1, r2 = benchmark_linear_qubit_operator(3, 2)
    assert isinstance(r1, float)
    assert isinstance(r2, float)


def test_comm_diag_coulomb():
    r1, r2 = \
        benchmark_commutator_diagonal_coulomb_operators_2D_spinless_jellium(4)
    assert isinstance(r1, float)
    assert isinstance(r2, float)


def test_mol_jw():
    r1 = run_molecular_operator_jordan_wigner(3)
    assert isinstance(r1, float)


def test_run_fermion_no():
    r1, r2 = run_fermion_math_and_normal_order(2)
    assert isinstance(r1, float)
    assert isinstance(r2, float)


def test_jw_sparse_time():
    r1 = run_jordan_wigner_sparse(2)
    assert isinstance(r1, float)


def test_run_linear_qop():
    r1, r2 = run_linear_qubit_operator(2)
    assert isinstance(r1, float)
    assert isinstance(r2, float)


def test_run_diag_comm():
    r1, r2 = run_diagonal_commutator(4)
    assert isinstance(r1, float)
    assert isinstance(r2, float)
