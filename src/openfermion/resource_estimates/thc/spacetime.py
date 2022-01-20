#coverage:ignore
"""Compute qubit vs toffoli for THC LCU"""
from math import pi
import itertools
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.scimath import arccos, arcsin  # has analytc continuation to cplx
from openfermion.resource_estimates.utils import QR, QI


def qubit_vs_toffoli(lam,
                     dE,
                     eps,
                     n,
                     chi,
                     beta,
                     M,
                     algorithm='half',
                     verbose=False):
    """
    Args:
        lam (float) - the lambda-value for the Hamiltonian
        dE (float) - allowable error in phase estimation. usually 0.001
        eps (float) - allowable error for synthesis (dE/(10 * lam)) usually
        n (int) - number of spin orbitals.
        chi (int) - number of bits of precision for state prep
        beta (int) - number of bits of precision for rotations
        M (int) - THC rank or r_{Thc}
        algorithm (str) - 'half', where half of the phasing angles are loaded
                                  at a time
                          'full', where angles loaded from QROM to perform
                                  phasing operations are loaded at the same time
                          Note: In arXiv:2011.03494,
                              'half' corresponds to Fig 11, while
                              'full' corresponds to Fig 12.
        verbose (bool) - do additional printing of intermediates?

    """
    # only valid algorithms accepted
    assert algorithm in ['half', 'full']

    # The number of iterations for the phase estimation.
    iters = np.ceil(pi * lam / (dE * 2))
    # The number of bits used for each register.
    nM = np.ceil(np.log2(M + 1))
    #This is number of distinct items of data we need to output, see Eq. (28).
    d = M * (M + 1) / 2 + n / 2
    # The number of bits used for the contiguous register.
    nc = np.ceil(np.log2(d))
    # The output size is 2*Log[M] for the alt values, χ for the keep value,
    # and 2 for the two sign bits.
    m = 2 * nM + 2 + chi

    # The next block of code finds the optimal number of bits to use for the
    # rotation angle for the amplitude amplification taking into account the
    # probability of failure and the cost of the rotations.
    oh = np.zeros(20, dtype=float)
    for p in range(1, 20 + 1):
        cos_term = arccos(np.power(2, nM) / np.sqrt(d) / 2)
        # print(cos_term)
        v = np.round(np.power(2, p) / (2 * pi) * cos_term)
        asin_term = arcsin(
            np.cos(v * 2 * pi / np.power(2, p)) * np.sqrt(d) / np.power(2, nM))
        sin_term = np.sin(3 * asin_term)**2
        oh[p - 1] = (20_000 * (1 / sin_term - 1) + 4 * p).real
    # br is the number of bits used in the rotation.
    br = np.argmin(oh) + 1
    # Next are the costs for the state preparation.
    cp1 = 2 * (10 * nM + 2 * br - 9)
    # There is cost 10*Log[M] for preparing the equal superposition over the
    # input registers. This is the costing from above Eq. (29).
    # This is the cost of computing the contiguous register and inverting it.
    # This is with a sophisticated scheme adding together triplets of bits.
    # This is the cost of step 2 in the list on page 14.
    cp2 = 2 * (nM**2 + nM - 1)
    # This is the cost of the QROM for the state preparation and its inverse.
    cp3 = QR(d, m)[1] + QI(d)[1]
    # The cost for the inequality test.
    cp4 = 2 * chi
    # The cost 2*nM for the controlled swaps.
    cp5 = 4 * nM
    # Then there is a cost of nM+1 for swapping the μ and ν registers, where
    # the +3 is because we need to control on two registers, and
    # control swap of the spin registers.
    cp6 = 2 * nM + 3
    #  The total cost in Eq. (33).
    CPCP = cp1 + cp2 + cp3 + cp4 + cp5 + cp6

    # Next are the costs for the select operation.
    # This is the cost of swapping based on the spin register. These costs are
    # from the list on page 15, and this is steps 1 and 7.
    cs1 = 2 * n
    k1 = 2**QI(M + n / 2)[0]
    cs2a = M + n / 2 - 2 + np.ceil(M / k1) + np.ceil(n / 2 / k1) + k1

    # The QROM for the rotation angles the first time.  Here M+n/2-2 is the cost
    # for generating them, and the second part is the cost for inverting them
    # with advanced QROM. The QROM for the rotation angles the second time.
    # Here the cost M-2 is for generating the angles the second time, and QI[M]
    # is for inverting the QROM. Steps 2 and 6.
    cs2b = M - 2 + QI(M)[1]
    # The cost of the rotations steps 3 and 5.
    cs3 = 4 * n * (beta - 2)
    # Cost for extra part in making the Z doubly controlled step 4.
    cs4 = 1
    # The total select cost in Eq. (43).
    CS = cs1 + cs2a + cs2b + cs3 + cs4
    # The cost given slightly above Eq. (44) is 2*nM+5. That is a typo and it
    # should have the aleph (equivalent to χ here) like at the top of the
    # column. Here we have +3 in this line, +1 in the next line and +1 for cs4,
    #  to give the same total.
    costref = 2 * nM + chi + 3
    cost = CPCP + CS + costref + 1

    # Next are qubit costs.
    ac1 = 2 * np.ceil(np.log2(iters + 1)) - 1
    ac2 = n
    ac3 = 2 * nM
    ac47 = 5
    ac8 = beta
    ac9 = nc

    kt = 2**QR(d, m)[0]
    ac10 = m * kt + np.ceil(np.log2(d / kt))
    # This is for the equal superposition state to perform the inequality test
    # with the keep register.
    ac11 = chi
    # The qubit to control the swap of the μ and ν registers.
    ac12 = 1
    aca = ac1 + ac2 + ac3 + ac47 + ac8 + ac9 + ac11 + ac12
    # This is the data needed for the rotations.
    ac13 = beta * n / 2
    # These are the ancillas needed for adding into the phase gradient state.
    ac14 = beta - 2
    # These are the temporary ancillas in between erasing the first QROM
    # ancillas and inverting that QROM. The +m is for output of first QROM.
    acc = ac13 + ac14 + m

    if verbose:
        print("Total Toffoli cost ", cost * iters)
        print("Ancilla for first QROM ", aca + ac10)
        print("Actual ancilla ... ", np.max([aca + ac10, aca + acc]))
        print("Spacetime volume ", np.max([aca + ac10, aca + acc]) * cost)

    # First are the numbers of qubits that must be kept throughout the
    # computation. See page 18.
    if algorithm == 'half':
        # The qubits used as the control registers for the phase estimation,
        # that must be kept the whole way through. If we used independent
        # controls each time that would increase the Toffoli cost by
        # np.ceil(np.log2iters+1]]-3, while saving
        # np.ceil(np.log2iters+1]]-1 qubits
        ac1 = np.ceil(np.log2(iters + 1))
    elif algorithm == 'full':
        # The qubits used as the control registers for the phase estimation,
        # that must be kept the whole way through. If we used independent
        # controls each time that would increase the Toffoli cost by
        # np.ceil(np.log2iters+1]]-3, while saving
        # np.ceil(np.log2iters+1]]-1 qubits.
        ac1 = 2 * np.ceil(np.log2(iters + 1)) - 1
    # The system qubits that must always be included.
    ac2 = n
    # The μ and ν registers, that must be kept because they are control
    # registers that aren't fully erased and must be reflected on.
    ac3 = 2 * nM
    # These are the qubits for the spin in the control state as well as the
    # qubit that is rotated for the preparation of the equal superposition
    # state, AND the qubit that is used to control.
    # None of these are fully inversely prepared.
    ac4512 = 4
    # The qubits for the phase gradient state.
    ac8 = beta
    # This is for the equal superposition state to perform the inequality test
    # with the keep register. It must be kept around and reflected upon.
    ac11 = chi
    # The total number of permanent qubits.
    perm = ac1 + ac2 + ac3 + ac4512 + ac8 + ac11
    # In preparing the equal superposition state there are 6 temporary qubits
    # used in the rotation of the ancilla.  There are another three that are
    # needed for the temporary results of inequality tests. By far the largest
    # number, however, come from keeping the temporary ancillas from the
    # inequality tests.  That should be 3*nM+nN-4.  There are an other two
    # qubits in output at the end that will be kept until this step is undone.
    # Note: not used?
    #nN = np.ceil(np.log2(n / 2))

    # This is the maximum number of qubits used while preparing the equal
    # superposition state.
    qu1 = perm + 4 * nM - 1
    # To explain the number of temporary ancillas, we have nM+1 to perform the
    # inequality test on mu and nu with out-of-place addition.  We have another
    # nM-2 for the equality test.  Then we can do the inequality tests on
    # mu and nu with constants (temporarily) overwriting these variables, and
    # keeping nM-1 qubits on each.  Then there are another 2 temporary qubits
    # used for the reflection.  That gives 4*nM-1 total.
    # This is the number of Toffolis during this step.
    tof1 = 10 * nM + 2 * br - 9

    # This is increasing the running number of permanent ancillas by 2 for the
    #  ν=M+1 flag qubit and the success flag qubit.
    perm = perm + 2
    # The number of temporary qubits used in this computation is the the same
    # as the number of Toffolis plus one.
    qu2 = perm + nM**2 + nM
    # The Toffoli cost of computing the contiguous register.
    tof2 = nM**2 + nM - 1
    # The running number of qubits is increased by the number needed for the
    # contiguous register.
    perm = perm + nc

    if algorithm == 'half':
        # Here I'm setting the k-value for the QROM by hand instead of choosing
        # the optimal one for Toffolis.
        kt = 16
    elif algorithm == 'full':
        # Here I'm setting the k-value for the QROM by hand instead of choosing
        #  the optimal one for Toffolis.
        kt = 32
    # This is the number of qubits needed during the QROM.
    qu3 = perm + m * kt + np.ceil(np.log2(d / kt))
    # The number of Toffolis for the QROM.
    tof3 = np.ceil(d / kt) + m * (kt - 1)
    # The number of ancillas used increases by the actual output size of  QROM.
    perm = perm + m
    # The number of ancilla qubits used for the subtraction for the ineql test.
    qu4 = perm + chi
    # We can use one of the qubits from the registers that are subtracted as
    # the flag qubit so we don't need an extra flag qubit.
    # The number of Toffolis needed for the inequality test. The number of
    # permanent ancillas is unchanged.*)
    tof4 = chi
    # We don't need any extra ancillas for the controlled swaps.
    qu5 = perm
    # We are swapping pairs of registers of size nM
    tof5 = 2 * nM
    # One extra ancilla for the controlled swap of mu and nu because it is
    # controlled on two qubits.
    qu6 = perm
    # One more Toffoli for the double controls.
    tof6 = nM + 1
    # Swapping based on the spin register.
    qu7 = perm
    tof7 = n / 2

    if algorithm == 'half':
        # We use these temporary ancillas for the first QROM for the rot angles.
        qu8 = perm + nM + beta * n / 4
    elif algorithm == 'full':
        # We use these temporary ancillas for the first QROM for the rot angles.
        qu8 = perm + nM + beta * n / 2
    # The cost of outputting the rotation angles including those for the
    # one-electron part.
    tof8 = M + n / 2 - 2

    if algorithm == 'half':
        # We are now need the output rotation angles, though we don't need the
        # temporary qubits from the unary iteration.
        perm = perm + beta * n / 4
    elif algorithm == 'full':
        # We are now need the output rotation angles, though we don't need the
        # temporary qubits from the unary iteration.
        perm = perm + beta * n / 2
    # We need a few temp registers for adding into the phase grad register.
    qu9 = perm + (beta - 2)

    if algorithm == 'half':
        # The cost of the rotations.
        tof9 = n * (beta - 2) / 2
        # Make a list where we keep subtr the data qubits that can be erased.
        # Table[-j*beta,{j,0,n/4-1}]+perm+(beta-2)
        qu10 = np.array([-j * beta for j in range(int(n / 4))]) \
               + perm + beta - 2
        # The cost of the rotations.
        # Table[2*(beta-2),{j,0,n/4-1}]
        tof10 = np.array([2 * (beta - 2) for j in range(int(n / 4))])
        # We've erased the data
        perm = perm - beta * n / 4
    elif algorithm == 'full':
        # The cost of the rotations.
        tof9 = n * (beta - 2)
        # Make a list where we keep subtr the data qubits that can be erased.
        # Table[-j*beta,{j,0,n/2-1}]+perm+(beta-2)
        qu10 = np.array([-j * beta for j in range(int(n / 2))]) \
               + perm + beta - 2
        # The cost of the rotations.
        # Table[2*(beta-2),{j,0,n/2-1}]
        tof10 = np.array([2 * (beta - 2) for j in range(int(n / 2))])
        # We've erased the data
        perm = perm - beta * n / 2

    # Find the k for the phase fixup for the erasure of the rotations.
    k1 = 2**QI(M + n / 2)[0]

    # Temp qubits used. Data qubits were already erased, so don't change perm.
    qu11 = perm + k1 + np.ceil(np.log2(M / k1))
    tof11 = np.ceil(M / k1) + np.ceil(n / 2 / k1) + k1

    # Swapping based on the spin register.
    qu12 = perm
    tof12 = n / 2

    # Swapping the spin registers.
    qu12a = perm
    tof12a = 1

    # Swapping based on the spin register.
    qu13 = perm
    tof13 = n / 2

    if algorithm == 'half':
        # We use these temp ancillas for the second QROM for the rot angles.
        qu14 = perm + nM - 1 + beta * n / 4
        perm = perm + beta * n / 4
    elif algorithm == 'full':
        # We use these temp ancillas for the second QROM for the rot angles.
        qu14 = perm + nM - 1 + beta * n / 2
        perm = perm + beta * n / 2
    tof14 = M - 2

    # We need a few temporary registers for adding into the phase grad register
    qu15 = perm + (beta - 2)

    if algorithm == 'half':
        # The cost of the rotations.
        tof15 = n * (beta - 2) / 2
    elif algorithm == 'full':
        # The cost of the rotations.
        tof15 = n * (beta - 2)

    # Just one Toffoli to do the controlled Z1.
    qu16 = perm
    tof16 = 1

    if algorithm == 'half':
        # Make a list where we keep subtr the data qubits that can be erased.
        # Table[-j*beta,{j,0,n/4-1}]+perm+(beta-2)
        qu17 = np.array([-j * beta for j in range(int(n / 4))
                        ]) + perm + beta - 2
        # The cost of the rotations.
        # Table[2*(beta-2),{j,0,n/4-1}]
        tof17 = np.array([2 * (beta - 2) for j in range(int(n / 4))])
        # We've erased the data.
        perm = perm - beta * n / 4
    elif algorithm == 'full':
        # Make a list where we keep subtr the data qubits that can be erased.
        # Table[-j*beta,{j,0,n/2-1}]+perm+(beta-2)
        qu17 = np.array([-j * beta for j in range(int(n / 2))]) \
               + perm + beta - 2
        # The cost of the rotations.
        # Table[2*(beta-2),{j,0,n/2-1}]
        tof17 = np.array([2 * (beta - 2) for j in range(int(n / 2))])
        # We've erased the data
        perm = perm - beta * n / 2

    # Find the k for the phase fixup for the erasure of the rotations.
    k1 = 2**QI(M)[0]

    # The temp qubits used. The data qubits were already erased,
    # so don't change perm.
    qu18 = perm + k1 + np.ceil(np.log2(M / k1))
    tof18 = np.ceil(M / k1) + k1

    # Swapping based on the spin register.
    qu19 = perm
    tof19 = n / 2

    # One extra ancilla for the controlled swap of mu and nu because it is
    # controlled on two qubits.
    qu20 = perm + 1
    # One extra Toffoli, because we are controlling on two qubits.
    tof20 = nM + 1

    # We don't need any extra ancillas for the controlled swaps.
    qu21 = perm
    # We are swapping pairs of registers of size nM
    tof21 = 2 * nM

    # The number of ancilla qubits used for the subtraction for the inequal test
    qu22 = perm + chi
    # We can use one of the qubits from the registers that are subtracted as
    # the flag qubit so we don't need an extra flag qubit.
    # The number of Toffolis needed for inverting the inequality test.
    # The number of permanent ancillas is unchanged.
    tof22 = chi
    # We can erase the data for the QROM for inverting the state preparation,
    # then do the phase fixup.
    perm = perm - m

    kt = 2**QI(d)[0]
    # This is the number of qubits needed during the QROM.
    qu23 = perm + kt + np.ceil(np.log2(d / kt))
    # The number of Toffolis for the QROM.
    tof23 = np.ceil(d / kt) + kt
    # The number of temporary qubits used in this computation is the same as
    # the number of Toffolis plus one. We are erasing the contiguous register
    # as we go so can subtract nc.
    qu24 = perm - nc + nM**2 + nM
    # The Toffoli cost of computing the contiguous register.
    tof24 = nM**2 + nM - 1
    # The contiguous register has now been deleted.
    perm = perm - nc
    # This is the maximum number of qubits used while preparing the equal
    # superposition state.
    qu25 = perm + 4 * nM - 1
    # This is the number of Toffolis during this step.
    tof25 = 10 * nM + 2 * br - 9
    # This is increasing the running number of permanent ancillas by 2 for
    # the ν=M+1 flag qubit and the success flag qubit.
    perm = perm - 2

    if algorithm == 'half':
        # We need some ancillas to perform a reflection on multiple qubits.
        # We are including one more Toffoli to make it controlled.
        qu26 = perm + costref + np.ceil(np.log2(iters + 1))
        tof26 = costref + np.ceil(np.log2(iters + 1))
    elif algorithm == 'full':
        # We need some ancillas to perform a reflection on multiple qubits.
        # We are including one more Toffoli to make it controlled.
        qu26 = perm + costref
        tof26 = costref

    qu27 = perm  # Iterate the control register.
    tof27 = 1

    # Labels
    sm = 'small element'
    pq = 'preparation QROM'
    rq = 'rotation QROM'
    ri = r'R$^{\dag}$'
    ro = 'R'
    iq = 'inverse QROM'

    color_dict = {
        sm: '#435CE8',
        pq: '#E83935',
        rq: '#F59236',
        ri: '#E3D246',
        ro: '#36B83E',
        iq: '#E83935'
    }

    if algorithm == 'half':
        tgates = np.hstack((np.array([
            tof1, tof2, tof3, tof4, tof5, tof6, tof7, tof8, tof9, tof8, tof9,
            tof9, tof8
        ]), tof10,
                            np.array([
                                tof11, tof12, tof12a, tof13, tof14, tof15,
                                tof14, tof15, tof16, tof15, tof14
                            ]), tof17,
                            np.array([
                                tof18, tof19, tof20, tof21, tof22, tof23, tof24,
                                tof25, tof26, tof27
                            ])))
        qubits = np.hstack(
            (np.array([
                qu1, qu2, qu3, qu4, qu5, qu6, qu7, qu8, qu9, qu8, qu9, qu9, qu8
            ]), qu10,
             np.array([
                 qu11, qu12, qu12a, qu13, qu14, qu15, qu14, qu15, qu16, qu15,
                 qu14
             ]), qu17,
             np.array(
                 [qu18, qu19, qu20, qu21, qu22, qu23, qu24, qu25, qu26, qu27])))
        labels = [sm, sm, pq, sm, sm, sm, sm, rq, ri, rq, ri, ro, rq] + \
                 [ro] * len(qu10) + \
                 [rq, sm, sm, sm, rq, ri, rq, ri, sm, ro, rq] + \
                 [ro] * len(qu17) + \
                 [rq, sm, sm, sm, sm, iq, sm, sm, sm, sm]

        colors = [color_dict[i] for i in labels]
    elif algorithm == 'full':
        tgates = np.hstack(
            (np.array([tof1, tof2, tof3, tof4, tof5, tof6, tof7, tof8,
                       tof9]), tof10,
             np.array([tof11, tof12, tof12a, tof13, tof14, tof15,
                       tof16]), tof17,
             np.array([
                 tof18, tof19, tof20, tof21, tof22, tof23, tof24, tof25, tof26,
                 tof27
             ])))
        qubits = np.hstack(
            (np.array([qu1, qu2, qu3, qu4, qu5, qu6, qu7, qu8, qu9]), qu10,
             np.array([qu11, qu12, qu12a, qu13, qu14, qu15, qu16]), qu17,
             np.array(
                 [qu18, qu19, qu20, qu21, qu22, qu23, qu24, qu25, qu26, qu27])))
        labels = [sm, sm, pq, sm, sm, sm, sm, rq, ri] + \
                 [ro] * len(qu10) + \
                 [rq, sm, sm, sm, rq, ri, sm] + \
                 [ro] * len(qu17) + \
                 [rq, sm, sm, sm, sm, iq, sm, sm, sm, sm]

        colors = [color_dict[i] for i in labels]

    # check lists are at least consistent
    assert all(
        len(element) == len(tgates) for element in [qubits, labels, colors])

    return tgates, qubits, labels, colors


def plot_qubit_vs_toffoli(tgates,
                          qubits,
                          labels,
                          colors,
                          tgate_label_thresh=100):
    """ Helper function to plot qubit vs toffoli similar to Figs 11 and 12 from
        'Even more efficient quantum...' paper (arXiv:2011.03494)

    Args:
        tgates (list or 1D vector) - list of toffoli values
        qubits (list or 1D vector) - list of qubit values
        labels (list) - list of labels corresponding to different steps of alg
        colors (list) - list of colors corresponding to different steps of alg
        tgate_label_thresh - don't label steps "thinner" than thresh of Toffolis
    """
    # To align the bars on the right edge pass a negative width and align='edge'
    ax = plt.gca()
    plt.bar(np.cumsum(tgates),
            qubits,
            width=-tgates,
            align='edge',
            color=colors)
    plt.bar(0, qubits[-1], width=sum(tgates), align='edge', color='#D7C4F2')
    plt.xlabel('Toffoli count')
    plt.ylabel('Number of qubits')

    # Now add the labels
    # First, group labels and neighboring tgates
    labels_grouped, tgates_grouped, qubits_grouped = group_steps(
        labels, tgates, qubits)
    for step, label in enumerate(labels_grouped):
        if 'small' in label:
            # skip the steps identified as 'small'
            continue
        elif tgates_grouped[step] < tgate_label_thresh:
            # otherwise skip really narrow steps
            continue
        else:
            x = np.cumsum(tgates_grouped)[step] - (tgates_grouped[step] * 0.5)
            y = 0.5 * (qubits_grouped[step] - qubits[-1]) + qubits[-1]
            ax.text(x,
                    y,
                    label,
                    rotation='vertical',
                    va='center',
                    ha='center',
                    fontsize='x-small')

    # Finally add system and control qubit label
    ax.text(0.5*np.sum(tgates), 0.5*qubits[-1], "System and control qubits", \
            va='center', ha='center',fontsize='x-small')

    plt.show()


def table_qubit_vs_toffoli(tgates, qubits, labels, colors):
    """ Helper function to generate qubit vs toffoli table .. text version of
         Fig 11 and Fig 12 in arXiv:2011.03494

    Args:
        tgates (list or 1D vector) - list of toffoli values
        qubits (list or 1D vector) - list of qubit values
        labels (list) - list of labels corresponding to different steps of alg
        colors (list) - list of colors corresponding to different steps of alg
    """

    print("=" * 60)
    print("{:>8s}{:>11s}{:>9s}{:>20s}{:>12s}".format('STEP', 'TOFFOLI',
                                                     'QUBIT*', 'LABEL',
                                                     'COLOR'))
    print("-" * 60)
    for step in range(len(tgates)):
        print('{:8d}{:11d}{:9d}{:>20s}{:>12s}'.format(step, int(tgates[step]),
                                                      int(qubits[step]),
                                                      labels[step],
                                                      colors[step]))
    print("=" * 60)
    print("  *Includes {:d} system and control qubits".format(int(qubits[-1])))


def group_steps(labels, tgates, qubits):
    """ Group similar adjacent steps by label. In addition to the grouped
        labels, also  returning the total Toffoli count and average qubits
        allocated for that grouping.
        Useful for collecting similar steps in the spacetime plots.

        Example:
          Input:
            labels = ['R', 'R', 'QROM', 'QROM, 'I-QROM', 'QROM', 'QROM', 'R']
            tgates = [  5,   8,     20,    10,       14,     30,     10,  20]
            qubits = [ 10,  10,     40,    20,        4,     80,     60,  60]

          Output:
            grouped_labels = ['R', 'QROM', 'I-QROM', 'QROM', 'R']
            grouped_tgates = [ 13,     30,       14,     40,  20]  (sum)
            grouped_qubits = [ 10,     30,        4,     70,  60]  (mean)

    """
    assert len(labels) == len(tgates)
    assert len(labels) == len(qubits)

    # Key function -- group identical nearest neighbors in labels (x[0])
    key_func = lambda x: x[0]

    # create grouped labels and tgates first
    grouped_labels = []
    grouped_tgates = []
    L = zip(labels, tgates)
    for label, group in itertools.groupby(L, key_func):
        grouped_labels.append(label)
        grouped_tgates.append(np.sum([i[1] for i in group]))

    # now do the grouped qubits
    # somehow doing multiple list comprehension the group breaks the grouping?
    # so we have to do this in a separate loop.
    grouped_qubits = []
    L = zip(labels, qubits)
    for label, group in itertools.groupby(L, key_func):
        grouped_qubits.append(np.mean([i[1] for i in group]))

    # sanity check -- shouldn't be losing total value in toffoli
    assert np.sum(tgates) == np.sum(grouped_tgates)
    return grouped_labels, grouped_tgates, grouped_qubits
