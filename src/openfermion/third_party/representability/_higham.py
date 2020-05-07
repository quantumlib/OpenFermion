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
"""
This module contains methods to find the closest positive semidefinite matrix
with fixed trace by the method in arXiv 1707.01022v1 and
 N.J. Higham, Linear Algebra and Its Applications 103, 103 (1998)
"""
from itertools import product
import numpy as np


@np.vectorize
def heaviside(x, bias=0) -> int:
    """
    Heaviside function Theta(x - bias)

    returns 1 if x >= bias else 0

    :param x: floating point number as input to heavisde
    :param bias: shift on the heaviside function
    :return: 1 or 0 int
    """
    indicator = 1 if x >= bias else 0
    return indicator


def higham_polynomial(eigenvalues, shift):
    """
    Calculate the higham_polynomial

    Args:
        eigenvalues:  vector of eigenvalues
        shift: where to put the bias for the heaviside function
    """
    heaviside_indicator = np.asarray(heaviside(eigenvalues, bias=shift))
    return heaviside_indicator.T.dot(eigenvalues - shift)


def higham_root(eigenvalues, target_trace, epsilon=1.0E-15):
    """
    Find the root of f(sigma) = sum_{j}Theta(l_{i} - sigma)(l_{i} - sigma) = T

    Args:
        eigenvalues: ordered list of eigenvalues from least to greatest
        target_trace: trace to maintain on new matrix
        epsilon: precision on bisection linesearch
    """
    if target_trace < 0.0:
        raise ValueError("Target trace needs to be a non-negative number")

    # when we want the trace to be zero
    if np.isclose(target_trace, 0.0):
        return eigenvalues[-1]

    # find top sigma
    sigma = eigenvalues[-1]
    while higham_polynomial(eigenvalues, sigma) < target_trace:
        sigma -= eigenvalues[-1]

    sigma_low = sigma
    sigma_high = eigenvalues[-1]

    while sigma_high - sigma_low >= epsilon:
        midpoint = sigma_high - (sigma_high - sigma_low) / 2.0
        if higham_polynomial(eigenvalues, midpoint) < target_trace:
            sigma_high = midpoint
        else:
            sigma_low = midpoint

    return sigma_high


def map_to_matrix(mat):
    if mat.ndim != 4:
        raise TypeError(
            "I only map rank-4 tensors to matices with symmetric support")
    dim = mat.shape[0]
    matform = np.zeros((dim**2, dim**2))
    for p, q, r, s in product(range(dim), repeat=4):
        assert np.isclose(mat[p, q, r, s].imag, 0.0)
        matform[p * dim + q, r * dim + s] = mat[p, q, r, s].real
    return matform


def map_to_tensor(mat):
    if mat.ndim != 2:
        raise TypeError(
            "I only map matrices to rank-4 tensors with symmetric support")
    dim = int(np.sqrt(mat.shape[0]))
    tensor_form = np.zeros((dim, dim, dim, dim))
    for p, q, r, s in product(range(dim), repeat=4):
        tensor_form[p, q, r, s] = mat[p * dim + q, r * dim + s]
    return tensor_form


def fixed_trace_positive_projection(bmat, target_trace):
    """
    Perform the positive projection with fixed trace

    Args:
        bmat:  Symmetric matrix to perform positive projection on
        target_trace:  What the trace should be
    Returns: new matrix that has the target trace and is positive semidefinite
    """
    bmat = np.asarray(bmat)
    map_to_four_tensor = False
    if bmat.ndim == 4:
        bmat = map_to_matrix(bmat)
        map_to_four_tensor = True

    # symmeterize bmat
    if np.allclose(bmat - bmat.conj().T, np.zeros_like(bmat)):
        bmat = 0.5 * (bmat + bmat.conj().T)

    w, v = np.linalg.eigh(bmat)
    if np.all(w >= -1.0 * float(1.0E-14)) and np.isclose(
            np.sum(w), target_trace):
        purified_matrix = bmat
    else:
        sigma = higham_root(w, target_trace)
        shifted_eigs = np.multiply(heaviside(w - sigma), (w - sigma))
        purified_matrix = np.zeros_like(bmat)
        for i in range(w.shape[0]):
            purified_matrix += shifted_eigs[i] * \
                               v[:, [i]].dot(v[:, [i]].conj().T)

    if map_to_four_tensor:
        purified_matrix = map_to_tensor(purified_matrix)

    return purified_matrix
