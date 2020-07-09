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
"""An implementation of Prony's method (or the matrix pencil method)
This fits a signal f(t) to sum_i=1^M a_i gamma_i^t, where a_i, gamma_i
are complex numbers
"""
import numpy
import scipy


def prony(signal):
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
    hankel0 = scipy.linalg.hankel(c=signal[:num_freqs],
                                  r=signal[num_freqs - 1:-1])
    hankel1 = scipy.linalg.hankel(c=signal[1:num_freqs + 1],
                                  r=signal[num_freqs:])
    shift_matrix = scipy.linalg.lstsq(hankel0.T, hankel1.T)[0]
    phases = numpy.linalg.eigvals(shift_matrix.T)

    generation_matrix = numpy.array(
        [[phase**k for phase in phases] for k in range(len(signal))])
    amplitudes = scipy.linalg.lstsq(generation_matrix, signal)[0]

    amplitudes, phases = zip(*sorted(
        zip(amplitudes, phases), key=lambda x: numpy.abs(x[0]), reverse=True))

    return numpy.array(amplitudes), numpy.array(phases)
