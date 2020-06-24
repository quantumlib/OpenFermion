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


from openfermion.config import EQ_TOLERANCE


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


def heisenberg_prony(signal, max_order, prony_length,
                     cutoff=1e-2, base=2, phase_cutoff=1e-2):
    """A sparse sampling form of Prony's method to achieve
    Heisenberg-limited scaling of multiple eigenvalues

    Args:
        signal(dict): The input sparse signal. Needs to have
            entries for signal[base**j*k], for j in 0,...,max_order-1
            and k in 0,...,prony_length
        max_order(int): The number of orders to estimate to
        prony_length(int): The length of the signal at each order
        base(int): the base of the exponent for each order
        cutoff(float): the minimum amplitude to keep a signal
        phase_cutoff(float): at each order, we consider two signals
            identical if they differ by base**order

    Returns:
        amplitudes(complex array): the amplitudes a_i, in descending order
            by their complex magnitude
        phases(complex array): the complex frequencies gamma_i,
            correlated with amplitudes.
    """

    for order in range(max_order):

        # Get amplitudes/phases at this order
        signal_piece = numpy.array([signal[base**order * k]
                                    for k in range(prony_length+1)])
        temp_amplitudes, temp_phases = prony(signal_piece)
        temp_amplitudes = temp_amplitudes.tolist()
        temp_phases = [float(numpy.angle(x)) % (2 * numpy.pi)
                       for x in temp_phases.tolist()]

        # Delete spurious phases
        for j in range(prony_length // 2):
            if numpy.abs(temp_amplitudes[j]) < cutoff:
                del temp_amplitudes[j:]
                del temp_phases[j:]
                break

        # Don't need to do matching if order = 0
        if order == 0:
            amplitudes = temp_amplitudes
            phases = temp_phases
            continue

        # Match phases with previous order and calculate new
        amplified_phases = [(base**order * phase) % (2 * numpy.pi)
                            for phase in phases]

        new_phase_pairings = match_phases(
            amplified_phases, amplitudes,
            temp_phases, temp_amplitudes)

        new_phases = [phases[bi] + phase_difference(
                      temp_phases[fi], amplified_phases[bi]) / base**order
                      for bi, fi in new_phase_pairings]

        # Delete duplicate phases - these may arise if a spurious signal
        # appeared in the previous round, and then disappears in this round.
        for j in range(len(new_phases)-1, -1, -1):
            if len([p for p in new_phases[:j]
                    if numpy.abs(phase_difference(p, new_phases[j])) <
                    phase_cutoff / base**order]):
                del new_phases[j]

        # Get new amplitudes by fitting to entire signal up to this order
        full_signal_x = [base ** l * k for l in range(order+1)
                         for k in range(prony_length+1)]
        full_signal_y = [signal[x] for x in full_signal_x]

        generation_matrix = numpy.array([
                [numpy.exp(1j*phase*k) for phase in new_phases]
                for k in full_signal_x])

        new_amplitudes =\
            scipy.linalg.lstsq(generation_matrix, full_signal_y)[0]

        # Sort by amplitudes
        if len(new_amplitudes) > 0:
            amplitudes, phases = zip(
                *sorted(zip(new_amplitudes, new_phases),
                        key=lambda x: numpy.abs(x[0]), reverse=True))

        amplitudes = list(amplitudes)
        phases = list(phases)

        # Delete spurious phases
        for j in range(len(amplitudes)):
            if amplitudes[j] < cutoff:
                del amplitudes[j:]
                del phases[j:]
                break

    return amplitudes, phases


def match_phases(old_phases, old_amplitudes,
                 new_phases, new_amplitudes):
    '''Matches a set of new and old phases to generate a set of
    pairings (allowing for the fact that multiple phases from one
    set may be indistinguishable in the second). Does this in a
    top-down strategy

    Args:
        new_phases(list of floats): the new phases to be matched
        old_phases(list of floats): the old (amplified) phases
            to be matched
        new_amplitudes(list of floats): corresponding amplitudes
            to new_phases
        old_amplitudes(list of floats): corresponding amplitudes
            to old_phases

    Returns:
        pairing(list of pairs of integers): the set of signals found
    '''

    pairing = []
    temp_amplitudes = [old_amplitudes, new_amplitudes]
    phases = [old_phases, new_phases]
    amplitude_queue = sorted([(order, index)
                              for order in range(2)
                              for index in range(len(temp_amplitudes[order]))],
                             key=lambda x: abs(temp_amplitudes[x[0]][x[1]]),
                             reverse=True)

    for order, index in amplitude_queue:

        # If this phase has been assigned to a larger signal, break
        if abs(temp_amplitudes[order][index]) < EQ_TOLERANCE:
            continue

        # Currently assuming a fixed variance.
        other_order = 1 - order
        other_index = min([
            oi for oi in range(len(temp_amplitudes[other_order]))],
            key=lambda x: log_likelihood(
                phases, temp_amplitudes, order, other_order, index, x))

        if order == 0:
            pair = (index, other_index)
        else:
            pair = (other_index, index)
        if pair in pairing:
            continue

        pairing.append(pair)

        lost_amplitude = min(temp_amplitudes[order][index],
                             temp_amplitudes[other_order][other_index],
                             key=lambda x: abs(x))
        temp_amplitudes[order][index] -= lost_amplitude
        temp_amplitudes[other_order][other_index] -= lost_amplitude

    return pairing


def log_likelihood(phases, amplitudes, order1, order2, index1, index2):
    '''Returns the log-likelihood of matching a given phase
    and amplitude (selected from the corresponding lists), modulo
    constant factors, and assuming variances are identical.

    Args:
        phases(list of lists of floats): list of phases
        amplitudes(list of lists of floats): list of amplitudes
        order1(0 or 1): order of first phase
        order2(0 or 1): order of second phase
        index1(int): index of first phase
        index2(int): index of second phase

    Returns:
        log(P(phase_1 <-> phase_2)*P(A_1 <-> A_2)) modulo constant factors
    '''
    return (phase_difference(phases[order1][index1],
                             phases[order2][index2])**2 +
            abs(amplitudes[order1][index1] -
                amplitudes[order2][index2])**2)


def phase_difference(angle1, angle2):
    '''Calculates the difference between two angles, including checking if
    they wrap around the circle.

    Args:
        angle1, angle2 (floats): the two angles to take the difference of

    Returns:
        angle1 - angle2: the difference **taken around the circle**
    '''
    if angle1 - angle2 > numpy.pi:
        return angle1 - angle2 - 2 * numpy.pi
    elif angle2 - angle1 > numpy.pi:
        return 2 * numpy.pi - angle2 - angle1
    else:
        return angle1 - angle2
