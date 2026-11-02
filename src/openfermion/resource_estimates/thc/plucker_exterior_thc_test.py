# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");

import numpy as np
try:
    from openfermion.resource_estimates.thc.plucker_exterior_thc import (
        bianchi_kronecker_action_matrix_free,
        closed_loop_differentiable_bianchi_bound,
        faddeev_leverrier_exterior_traces,
        packed_to_symmetric_matrix,
        plucker_wedge_pullback_action,
        polarized_bianchi_action_matrix_free,
        symmetric_matrix_to_packed,
    )
except ImportError:
    try:
        from qualtran_openfermion_dropin.plucker_exterior_thc import (
            bianchi_kronecker_action_matrix_free,
            closed_loop_differentiable_bianchi_bound,
            faddeev_leverrier_exterior_traces,
            packed_to_symmetric_matrix,
            plucker_wedge_pullback_action,
            polarized_bianchi_action_matrix_free,
            symmetric_matrix_to_packed,
        )
    except ImportError:
        from plucker_exterior_thc import (
            bianchi_kronecker_action_matrix_free,
            closed_loop_differentiable_bianchi_bound,
            faddeev_leverrier_exterior_traces,
            packed_to_symmetric_matrix,
            plucker_wedge_pullback_action,
            polarized_bianchi_action_matrix_free,
            symmetric_matrix_to_packed,
        )


def test_packed_isometry_and_bianchi_quartic_vanishing():
    rng = np.random.default_rng(123)
    n_orb = 12
    mat = rng.standard_normal((n_orb, n_orb))
    mat = 0.5 * (mat + mat.T)
    packed = symmetric_matrix_to_packed(mat)
    assert np.isclose(np.linalg.norm(packed), np.linalg.norm(mat, "fro"), atol=1e-15)
    unpacked = packed_to_symmetric_matrix(packed, n_orb)
    np.testing.assert_allclose(unpacked, mat, atol=1e-15)

    k_m = rng.standard_normal((n_orb, n_orb))
    k_m = 0.5 * (k_m + k_m.T)
    k_n = rng.standard_normal((n_orb, n_orb))
    k_n = 0.5 * (k_n + k_n.T)
    for _ in range(20):
        v = rng.standard_normal(n_orb)
        v = v / np.linalg.norm(v)
        vv = np.outer(v, v)
        b_vv = bianchi_kronecker_action_matrix_free(k_m, vv)
        pol_vv = polarized_bianchi_action_matrix_free(k_m, k_n, vv)
        assert abs(float(np.sum(vv * b_vv))) < 1e-13
        assert abs(float(np.sum(vv * pol_vv))) < 1e-13


def test_plucker_exterior_pullback_and_faddeev_leverrier_rank_annihilation():
    rng = np.random.default_rng(99)
    n_orb = 10
    u = rng.standard_normal(n_orb)
    v = rng.standard_normal(n_orb)
    omega = np.outer(u, v) - np.outer(v, u)

    a = rng.standard_normal(n_orb)
    k_rank1 = np.outer(a, a)
    np.testing.assert_allclose(plucker_wedge_pullback_action(k_rank1, omega), 0.0, atol=1e-14)
    traces_r1 = faddeev_leverrier_exterior_traces(k_rank1)
    assert abs(traces_r1[2]) < 1e-13
    assert abs(traces_r1[3]) < 1e-13
    assert abs(traces_r1[4]) < 1e-13

    b = rng.standard_normal(n_orb)
    k_rank2 = np.outer(a, a) + np.outer(b, b)
    traces_r2 = faddeev_leverrier_exterior_traces(k_rank2)
    assert traces_r2[2] > 1e-3
    assert abs(traces_r2[3]) < 1e-12
    assert abs(traces_r2[4]) < 1e-11
