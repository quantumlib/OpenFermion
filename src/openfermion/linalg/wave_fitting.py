#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
'''Functions for fitting simple oscillating functions'''

from typing import Tuple
import numpy
import scipy


def fit_known_frequencies(
    signal: numpy.ndarray, times: numpy.ndarray, frequencies: numpy.ndarray
) -> numpy.ndarray:
    """Fits a set of known exponential components to a dataset

    Decomposes a function g(t) as g(t)=sum_jA_jexp(iw_jt), where the frequencies
    w_j are already known. Namely, makes a least-squares fit.

    Arguments:
        signal {numpy.ndarray} -- the signal g(t) to be fit
        times {numpy.ndarray} -- t values of the signal
        frequencies {numpy.ndarray} -- known frequencies w_j

    Returns:
        amplitudes {numpy.ndarray} -- the found amplitudes A_j
    """
    generation_matrix = numpy.array(
        [[numpy.exp(1j * time * freq) for freq in frequencies] for time in times]
    )
    amplitudes = scipy.linalg.lstsq(generation_matrix, signal)[0]
    return amplitudes


def prony(signal: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Estimates amplitudes and phases of a sparse signal using Prony's method.

    Single-ancilla quantum phase estimation returns a signal
    g(k)=sum (aj*exp(i*k*phij)), where aj and phij are the amplitudes
    and corresponding eigenvalues of the unitary whose phases we wish
    to estimate. When more than one amplitude is involved, Prony's method
    provides a simple estimation tool, which achieves near-Heisenberg-limited
    scaling (error scaling as N^{-1/2}K^{-3/2}).

    Args:
        signal(1d complex array): the signal to fit

    Returns:
        amplitudes(list of complex values): the amplitudes a_i,
        in descending order by their complex magnitude
        phases(list of complex values): the complex frequencies gamma_i,
            correlated with amplitudes.
    """

    num_freqs = len(signal) // 2
    hankel0 = scipy.linalg.hankel(c=signal[:num_freqs], r=signal[num_freqs - 1 : -1])
    hankel1 = scipy.linalg.hankel(c=signal[1 : num_freqs + 1], r=signal[num_freqs:])
    shift_matrix = scipy.linalg.lstsq(hankel0.T, hankel1.T)[0]
    phases = numpy.linalg.eigvals(shift_matrix.T)

    generation_matrix = numpy.array([[phase**k for phase in phases] for k in range(len(signal))])
    amplitudes = scipy.linalg.lstsq(generation_matrix, signal)[0]

    amplitudes, phases = zip(
        *sorted(zip(amplitudes, phases), key=lambda x: numpy.abs(x[0]), reverse=True)
    )

    return numpy.array(amplitudes), numpy.array(phases)
