from typing import Callable, List, Tuple


class Bijection:

    def __init__(self, fwd: Callable, rev: Callable, sizes: Callable):
        """
        Bijection holds forward maps and backwards maps

        Args:
            fwd: forward mapping function
            rev: backwards mapping function
            sizes: function getting tuple of domain and codomain element
                   dimensions
        """
        self.fwd = fwd
        self.rev = rev
        self.domain_element_sizes = sizes


def index_index_basis(n: int) -> Bijection:
    """
    Create an index-to-index Bijection

    Args:
        n: length of basis
    Returns: A Bijection object
    """
    idx_dict = dict(zip(range(n), range(n)))

    def forward(i: int) -> int:
        return idx_dict[i]

    def reverse(i: int) -> int:
        return idx_dict[i]

    def sizes() -> Tuple[int, int]:
        return 1, 1

    return Bijection(forward, reverse, sizes)


def index_tuple_basis(codomain_elements: List[Tuple[int, ...]]) -> Bijection:
    """
    Create an index-to-tuple basis

    Args:
        codomain_elements: tuples representing codomain elements
    """
    idx_dict = dict(zip(range(len(codomain_elements)), codomain_elements))
    tuple_dict = dict(zip(codomain_elements, range(len(codomain_elements))))
    codomain_element_size = len(codomain_elements[0])

    def forward(i: int):
        return idx_dict[i]

    def reverse(t: Tuple[int, ...]) -> int:
        return tuple_dict[t]

    def sizes() -> Tuple[int, ...]:
        return (1, codomain_element_size)

    return Bijection(forward, reverse, sizes)
