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

import numpy
from openfermion.measurements import partition_iterator

MAX_LOOPS = 1e6


def pair_within(labels: list) -> list:
    """
    Generates pairings of labels that contain each pair at least once.

    A pairing of a list is a set of pairs of list elements. E.g. a pairing of

    labels = [1, 2, 3, 4, 5, 6, 7, 8]

    could be

    [(1, 2), (3, 4), (5, 6), (7, 8)]

    (Note that we insist each element only appears in a pairing once; the
    following is not a pairing:

    [(1, 1), (2, 2), (3, 4), (5, 6), (7, 8)]

    This function generates a set of len(labels)-1 such pairings of the input
    list labels, such that each element in labels is paired with each other
    element in at least one pairing

    Args:
        labels (list): list of elements

    Yields:
        pairings (list): list of pairings of elements of labels
    """
    if not labels:
        return
    if len(labels) == 1:
        yield (labels[0],)
        return

    # Determine fragment size
    fragment_size = len(labels) // 2
    frag1 = labels[:fragment_size]
    frag2 = labels[fragment_size:]

    for pairing in pair_between(frag1, frag2, len(frag2) % 2):
        yield pairing

    if len(labels) % 4 == 1:
        frag1.append(None)

    for (pairing1, pairing2) in zip(pair_within(frag1), pair_within(frag2)):

        if len(labels) % 4 == 1:
            if pairing1[-1] is None:
                yield pairing1[:-1] + pairing2
            else:
                extra_pair = ((pairing1[-1], pairing2[-1]),)
                zero_index, = [
                    pair[0] for pair in pairing1[:-1] if pair[1] is None
                ]
                pairing1 = tuple(
                    pair for pair in pairing1[:-1] if pair[1] is not None)
                yield pairing1 + pairing2[:-1] + extra_pair + (zero_index,)

        elif len(labels) % 4 == 2:
            extra_pair = ((pairing1[-1], pairing2[-1]),)
            yield pairing1[:-1] + pairing2[:-1] + extra_pair

        elif len(labels) % 4 == 3:
            yield pairing1[:-1] + pairing2 + (pairing1[-1],)

        else:
            yield pairing1 + pairing2


def pair_between(frag1: list, frag2: list, start_offset: int = 0) -> tuple:
    """Pairs between two fragments of a larger list

    A pairing of a list is a set of pairs of list elements. E.g. a pairing of

    labels = [1, 2, 3, 4, 5, 6, 7, 8]

    could be

    [(1, 2), (3, 4), (5, 6), (7, 8)]

    (Note that we insist each element only appears in a pairing once; the
    following is not a pairing:

    [(1, 1), (2, 2), (3, 4), (5, 6), (7, 8)]

    This function generates a set of pairings between elements of frag1
    and frag2 such that element1 in frag 1 and element2 in frag2,
    the pair (element1, element2) is found exactly once within the pairing.

    Args:
        frag1, frag2 (lists): the elements to be paired
        start_offset (int): prevents the first start_offset pairings
            from being yielded

    Yields:
        pairing tuple: the desired pairings, followed by
            any unpaired elements
    """

    num_iter = max(len(frag1), len(frag2))
    num_pairs = min(len(frag1), len(frag2))

    for index_offset in range(start_offset, num_iter):

        if len(frag1) > len(frag2):
            pairing = tuple(
                (frag1[(index + index_offset) % len(frag1)], frag2[index])
                for index in range(num_pairs))
            pairing += tuple(frag1[index % len(frag1)] for index in range(
                len(frag2) + index_offset,
                len(frag1) + index_offset))
        else:
            pairing = tuple(
                (frag1[index], frag2[(index + index_offset) % len(frag2)])
                for index in range(num_pairs))
        if len(frag2) > len(frag1):
            pairing += tuple(frag2[index % len(frag2)] for index in range(
                len(frag1) + index_offset,
                len(frag2) + index_offset))

        yield pairing


def _loop_iterator(func, *params):
    generator = func(*params)
    looped = False
    num_loops = 0
    while True:
        for res in generator:
            yield res, looped
        looped = True
        num_loops += 1
        if num_loops > MAX_LOOPS:
            raise ValueError(
                'Number of loops exceeded maximum allowed.')  # pragma: no cover
        generator = func(*params)


def _gen_partitions(labels, min_size=4):
    """
    Generates a set of exponentially smaller partitions of a set

    Args:
        labels(list): list to be partitioned
    """
    if len(labels) == 1:
        yield (labels,)
        return
    partitions = (labels[:len(labels) // 2], labels[len(labels) // 2:])
    while True:
        yield partitions
        if len(partitions[-1]) < min_size:
            return
        new_partitions = []
        for part in partitions:
            new_partitions.append(part[:len(part) // 2])
            new_partitions.append(part[len(part) // 2:])
        partitions = new_partitions


def _gen_pairings_between_partitions(parta, partb):
    if len(parta + partb) < 5:
        yield (tuple(parta), tuple(partb))
    splita = [parta[:len(parta) // 2], parta[len(parta) // 2:]]
    splitb = [partb[:len(partb) // 2], partb[len(partb) // 2:]]
    for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)):
        if max(len(splita[a]), len(splitb[b])) < 2:
            continue
        if min(len(splita[1 - a]), len(splitb[1 - b])) < 1:
            continue  # pragma: no cover
        gen_a = _loop_iterator(pair_within, splita[a])
        gen_b = _loop_iterator(pair_within, splitb[b])
        num_iter = max(
            len(splitb[b]) - 1 + len(splitb[b]) % 2,
            len(splita[a]) - 1 + len(splita[a]) % 2)
        for _ in range(num_iter):
            pair_a, _ = next(gen_a)
            pair_b, _ = next(gen_b)
            gen_ab = pair_between(splita[1 - a], splitb[1 - b])
            for pair_ab in gen_ab:
                yield pair_a + pair_b + pair_ab


def pair_within_simultaneously(labels: list) -> tuple:
    """Generates simultaneous pairings between four-element combinations

    A pairing of a list is a set of pairs of list elements. E.g. a pairing of

    labels = [1, 2, 3, 4, 5, 6, 7, 8]

    could be

    [(1, 2), (3, 4), (5, 6), (7, 8)]

    (Note that we insist each element only appears in a pairing once; the
    following is not a pairing:

    [(1, 1), (2, 2), (3, 4), (5, 6), (7, 8)]

    This function generates a set of pairings such that for every four elements
    (i,j,k,l) in 'labels', there exists one pairing containing both (i,j) and
    (k,l)

    Args:
        labels(list): list of elements to be paired

    Yields:
        pairings(tuple of pairs): the desired pairings
    """

    if len(labels) <= 3:
        return

    for partition in _gen_partitions(labels):
        generator_list = [
            _loop_iterator(pair_within, partition[j])
            for j in range(len(partition))
        ]
        for dummy1 in range(len(partition[-2]) - 1 + len(partition[-2]) % 2):
            pairing = tuple()
            for generator in generator_list[::2]:
                pairing = pairing + next(generator)[0]
            for dummy2 in range(
                    len(partition[-1]) - 1 + len(partition[-1]) % 2):
                pairing2 = tuple(pairing)
                for generator in generator_list[1::2]:
                    pairing2 = pairing2 + next(generator)[0]
                yield pairing2

        if len(partition[-1]) < 3:
            continue

        for partition_pairing in pair_within(partition):
            generator_list = [
                _loop_iterator(_gen_pairings_between_partitions, part_a, part_b)
                for part_a, part_b in partition_pairing
            ]
            while True:
                pairing = tuple()
                looped = True
                for generator in generator_list:
                    this_pairing, this_looped = next(generator)
                    pairing += this_pairing
                    if this_looped is False:
                        looped = False
                if looped is True:
                    break
                yield pairing


def _get_padding(num_bins, bin_size):
    """
    For parallel iteration: gets the smallest number L' >= bin_size
    such that num_bins is smaller than the lowest factor of L'.
    """
    trial_size = bin_size
    while True:
        success_flag = True
        for divisor in range(2, num_bins - 1):
            if trial_size % divisor == 0:
                success_flag = False
                break
        if success_flag:
            return trial_size
        trial_size += 1


def _asynchronous_iter(iterators, flatten=False):
    """
    Iterates over a set of K iterators with max L elements to
    generate all pairs between them in O(L^2 + 2L log(L) + log(L)^2),
    assuming L>>K. When appropriate, calls a different iterator
    optimized for small lists.

    Args:
        iterators(list of iterators): iterators to be passed
        flatten(boolean): whether to concatenate or join the results.
    Yields:
        next_result(list of results): the joined/concatenated set of results.
    """
    iterator_lists = [list(iterator) for iterator in iterators]
    num_lists = len(iterator_lists)
    list_size = max([len(lst) for lst in iterator_lists])

    # Edge cases
    if list_size == 1:
        next_res = [
            iterator[0] if iterator else None for iterator in iterator_lists
        ]
        if flatten:
            next_res = [x for result in next_res if result for x in result]
        yield tuple(next_res)
        return
    if numpy.log2(num_lists + 1) * list_size**2 < num_lists**2:
        for next_res in _asynchronous_iter_small_lists(iterator_lists, flatten):
            yield next_res
        return

    new_size = _get_padding(num_lists, list_size)
    for lst in iterator_lists:
        lst += [None] * (new_size - len(lst))

    for j in range(new_size):
        for l in range(new_size):
            next_res = [
                iterator_lists[k][(j * k + l) % new_size]
                for k in range(num_lists - 1)
            ]
            next_res.append(iterator_lists[-1][j])
            if flatten:
                next_res = [x for result in next_res if result for x in result]
            yield tuple(next_res)


def _asynchronous_iter_small_lists(iterator_lists, flatten=False):
    """
    Iterates over a set of K iterators of max L items to generate all pairs
    between them in O(log(K)L^2) time - this is suboptimal when L>>K,
    but does not require list padding, making it better for small L.

    Args:
        iterators(list of iterators): iterators to be passed
        flatten(boolean): whether to concatenate or join the results.
    Yields:
        next_result(list of results): the joined/concatenated set of results.
    """
    for partitions in partition_iterator(iterator_lists, 2):
        for res in _asynchronous_iter(
            [_parallel_iter(partition, flatten) for partition in partitions],
                flatten):
            yield res


def _parallel_iter(iterators, flatten=False):
    """
    Iterates in parallel over a set of iterators.
    Stopped iterators are removed, so the position of any
    result is not conserved.

    Args:
        iterators(list of iterables): iterators to be passed
        flatten(boolean): whether to concatenate or join the results.
    Yields:
        next_result(list of results): the joined/concatenated set of results.
    """
    iterators = [iter(iterator) for iterator in iterators]

    while iterators:
        next_result = []
        for j in range(len(iterators) - 1, -1, -1):
            temp = next(iterators[j], None)
            if temp is None:
                del iterators[j]
            else:
                if flatten:
                    next_result = list(temp) + next_result
                else:
                    next_result = [temp] + next_result
        if next_result:
            yield tuple(next_result)


def pair_within_simultaneously_binned(binned_majoranas: list) -> tuple:
    """Generates symmetry-respecting pairings between four-elements in a list

    A pairing of a list is a set of pairs of list elements. E.g. a pairing of

    labels = [1, 2, 3, 4, 5, 6, 7, 8]

    could be

    [(1, 2), (3, 4), (5, 6), (7, 8)]

    (Note that we insist each element only appears in a pairing once; the
    following is not a pairing:

    [(1, 1), (2, 2), (3, 4), (5, 6), (7, 8)]

    This function generates a pairing of a list of Majoranas that covers
    all 2-RDM elements that conserve a set of symmetry conditions. That is,
    this function guarantees that for any four elements (i,j,k,l) in the input,
    if the corresponding RDM term gamma_igamma_jgamma_kgamma_l satisfies a
    symmetry of the system, the pairs (i,j) and (k,l) will appear simultaneously
    in at least one pairing.

    The constraints are defined by a binning of the Majoranas into bins such
    that Majoranas in bin n commute with symmetry S_i if the ith binary digit
    of n is 0.

    Args:
        binned_majoranas(list of lists of integers): majoranas to be paired,
            separated up by their symmetry bins.

    Yields:
        pairing(tuple): one of the desired pairings
    """

    # Generate all four-fold pairings within bins
    iterators = [pair_within_simultaneously(bn) for bn in binned_majoranas]
    for pairing in _parallel_iter(iterators, flatten=True):
        yield pairing

    # Iterate over all pairs within bins
    num_bins = len(binned_majoranas)
    if max([len(bn) for bn in binned_majoranas]) > 1 and num_bins > 1:
        iterators = [pair_within(bn) for bn in binned_majoranas]
        for pairing in _asynchronous_iter(iterators, flatten=True):
            yield pairing

    # Pair between bins
    for bin_gap in range(1, num_bins // 2):
        iterators = []
        for bin_index in range(num_bins):
            if bin_index < bin_index ^ bin_gap:
                iterators.append(
                    pair_between(binned_majoranas[bin_index],
                                 binned_majoranas[bin_index ^ bin_gap]))
        for pairing in _asynchronous_iter(iterators, flatten=True):
            yield pairing


def pair_within_simultaneously_symmetric(num_fermions: int,
                                         num_symmetries: int) -> tuple:
    """Generates symmetry-respecting pairings between four-elements in a list

    A pairing of a list is a set of pairs of list elements. E.g. a pairing of

    labels = [1, 2, 3, 4, 5, 6, 7, 8]

    could be

    [(1, 2), (3, 4), (5, 6), (7, 8)]

    (Note that we insist each element only appears in a pairing once; the
    following is not a pairing:

    [(1, 1), (2, 2), (3, 4), (5, 6), (7, 8)]

    This function generates a pairing of a list of Majoranas that covers
    all 2-RDM elements that conserve a set of symmetry conditions. That is,
    this function guarantees that for any four elements (i,j,k,l) in the input,
    if the corresponding RDM term gamma_igamma_jgamma_kgamma_l satisfies a
    symmetry of the system, the pairs (i,j) and (k,l) will appear simultaneously
    in at least one pairing.

    We assume in this function that each symmetry divides the set of Majoranas
    in two, indexed by their binary digits.

    Args:
        num_fermions (int) : The number of fermions to be considered (the
            number of Majoranas generated will be twice this size)
        num_symmetries (int): the number of symmetries to be respectd.
    """
    binned_majoranas = [[
        index
        for index in range(2 * num_fermions)
        if index % 2**num_symmetries == bin_index
    ]
                        for bin_index in range(2**num_symmetries)]

    for pairing in pair_within_simultaneously_binned(binned_majoranas):
        yield pairing
