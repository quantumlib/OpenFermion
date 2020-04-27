# BSD 3-Clause License
#
# Copyright (c) 2018 Rigetti & Co, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# pylint: disable=C
from itertools import product
import numpy as np
import pytest
from openfermion.third_party.representability._higham import (
    heaviside, higham_polynomial, higham_root, map_to_tensor, map_to_matrix,
    fixed_trace_positive_projection)


def test_heaviside():
    assert np.isclose(heaviside(0), 1.0)
    assert np.isclose(heaviside(0.5), 1.0)
    assert np.isclose(heaviside(-0.5), 0.0)
    assert np.isclose(heaviside(-0.5, -1), 1.0)
    assert np.isclose(heaviside(-2, -1), 0)


def test_highham_polynomial():
    eigs = np.arange(10)
    assert np.isclose(higham_polynomial(eigs, eigs[-1]), 0.0)
    assert np.isclose(higham_polynomial(eigs, 0), sum(eigs))
    assert np.isclose(higham_polynomial(eigs, 5), sum(eigs[5:] - 5))
    assert np.isclose(higham_polynomial(eigs, 8), sum(eigs[8:] - 8))


def test_higham_root():
    dim = 20
    np.random.seed(42)
    mat = np.random.random((dim, dim))
    mat = 0.5 * (mat + mat.T)
    w, _ = np.linalg.eigh(mat)
    target_trace = np.round(w[-1] - 1)
    sigma = higham_root(w, target_trace)
    assert np.isclose(higham_polynomial(w, shift=sigma), target_trace)

    with pytest.raises(ValueError):
        higham_root(w, target_trace=-1)

    tw = higham_root(w, target_trace=0)
    assert np.isclose(tw, w[-1])


def test_matrix_2_tensor():
    dim = 10
    np.random.seed(42)
    mat = np.random.random((dim**2, dim**2))
    mat = 0.5 * (mat + mat.T)
    tensor = map_to_tensor(mat)
    for p, q, r, s in product(range(dim), repeat=4):
        assert np.isclose(tensor[p, q, r, s], mat[p * dim + q, r * dim + s])

    test_mat = map_to_matrix(tensor)
    assert np.allclose(test_mat, mat)

    with pytest.raises(TypeError):
        map_to_tensor(np.zeros((4, 4, 4, 4)))

    with pytest.raises(TypeError):
        map_to_matrix(np.zeros((4, 4)))


def test_reconstruction():
    dim = 20
    np.random.seed(42)
    mat = np.random.random((dim, dim))
    mat = 0.5 * (mat + mat.T)
    test_mat = np.zeros_like(mat)
    w, v = np.linalg.eigh(mat)
    for i in range(w.shape[0]):
        test_mat += w[i] * v[:, [i]].dot(v[:, [i]].T)
    assert np.allclose(test_mat - mat, 0.0)

    test_mat = fixed_trace_positive_projection(mat, np.trace(mat))
    assert np.isclose(np.trace(test_mat), np.trace(mat))
    w, v = np.linalg.eigh(test_mat)
    assert np.all(w >= -(float(4.0E-15)))

    mat = np.arange(16).reshape((4, 4))
    mat = 0.5 * (mat + mat.T)
    mat_tensor = map_to_tensor(mat)
    trace_mat = np.trace(mat)
    true_mat = fixed_trace_positive_projection(mat, trace_mat)
    test_mat = map_to_matrix(
        fixed_trace_positive_projection(mat_tensor, trace_mat))
    assert np.allclose(true_mat, test_mat)

    assert np.allclose(true_mat,
                       fixed_trace_positive_projection(true_mat, trace_mat))


def test_mlme():
    """
    Test from fig 1 of maximum likelihood minimum effort!
    """
    eigs = np.array(
        list(reversed([3.0 / 5, 1.0 / 2, 7.0 / 20, 1.0 / 10, -11.0 / 20])))
    target_trace = 1.0
    sigma = higham_root(eigs, target_trace)
    shifted_eigs = np.multiply(heaviside(eigs - sigma), (eigs - sigma))
    assert np.allclose(shifted_eigs, [0, 0, 1.0 / 5, 7.0 / 20, 9.0 / 20])
