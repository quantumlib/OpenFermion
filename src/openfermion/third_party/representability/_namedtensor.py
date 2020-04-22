from typing import Iterable, Generator, Optional, Union
from itertools import zip_longest
import numpy as np
from openfermion.third_party.representability._bijections import Bijection, \
    index_index_basis


class Tensor(object):
    """
    Instantiation of named tensor
    """

    def __init__(self,
                 *,
                 tensor: Optional[Union[None, np.ndarray]] = None,
                 basis: Optional[Union[None, Bijection]] = None,
                 name: Optional[Union[None, str]] = None):
        """
        Named Tensor that allows one to label elements with different indices

        For example, a 2-pdm is labeled by geminals (i, j), (k, l) in matrix
        form.  The matrix representing the 2-RDM can now be indexed into with
        tensor notation (i, j, k, l) instead of calling
        [i * dim + j, l * dim + k]

        Args:
            tensor: numpy.ndarray to hold the tensor data
            basis: basis on all the axes
            name: name of the tensor
        """
        if tensor is not None:
            self.dim = tensor.shape[0]
            self.ndim = tensor.ndim
            self.data = np.copy(tensor)
            self.size = self.dim**self.ndim
            self.name = name
            if basis is None:
                self.basis = index_index_basis(self.dim)
            else:
                if not isinstance(basis, Bijection):
                    raise TypeError("Basis must be a Bijection object")
                self.basis = basis
        else:
            self.dim = None
            self.ndim = None
            self.data = None
            self.size = None
            self.basis = basis
            self.name = name

    def __getitem__(self, indices):
        """
        returns the tensor data if loaded
        """
        if self.data is not None:
            return self.data[indices]
        else:
            raise TypeError("data store is not set")

    def __call__(self, *indices):
        """
        Index into the the data by passing through the basis first

        :param indices: indices for the rev_bas
        :return: element of the data
        """
        # I need a way to find out the dimension of an element of the codomain
        codomain_element_size = self.basis.domain_element_sizes()[1]
        index_set = []
        for idx_set in grouper(indices, codomain_element_size):
            if len(idx_set) == 1:
                index_set.append(self.basis.rev(idx_set[0]))
            else:
                index_set.append(self.basis.rev(idx_set))

        return self.data[tuple(index_set)]

    @staticmethod
    def get_obj_size(obj):
        """
        Determine the number of 'elements' an object contains.

        Integers are 1, tuples and lists are len(tuple/list)

        Args:
            obj: object to query for length
        Returns:  length of the object
        """
        if isinstance(obj, (tuple, list)):
            return len(obj)
        elif isinstance(obj, (float, int, complex, bool)):
            return 1
        else:
            raise TypeError("object type doesn't have a recognized length")

    def index_vectorized(self, *indices):
        """
        Perform the canonical index bijection to a scalar

        Note: the start returns a tuple of n-indices. That includes 1
        """
        return self.index_bijection(self.index_transform(indices), self.ndim,
                                    self.dim)

    def index_transform(self, indices):
        """
        Transform the indices to the basis indices

        :param indices: Tuple of tensor indices
        :return:
        """
        codomain_element_size = self.basis.domain_element_sizes()[1]
        index_set = []
        for idx_set in grouper(indices, codomain_element_size):
            index_set.append(
                self.basis.rev(idx_set[0] if len(idx_set) == 1 else idx_set))

        return tuple(index_set)

    @staticmethod
    def index_bijection(indices, ndim, dim):
        """
        calculate the bijection with tensor dim counting
        """
        if len(indices) != ndim:
            raise TypeError(
                "indices are inappriopriate length for the given ndim")

        # C-order canonical vectorization--i.e. right most index in indices
        # changes with the highest frequency
        bijection = 0
        for n in range(ndim):
            bijection += indices[n] * dim**(ndim - n - 1)
        return bijection

    def utri_iterator(self) -> Generator:
        """
        Iterate over the upper triangle (including diagonal)
        and return data value and index
        """
        return self._iterator("upper")

    def ltri_iterator(self) -> Generator:
        """
        Iterate over the lower triangle (including diagonal)
        and return data value and index
        """
        return self._iterator("lower")

    def all_iterator(self) -> Generator:
        """
        Iterate over the lower triangle (including diagonal)
        and return data value and index
        """
        return self._iterator("all")

    def _iterator(self, ultri: str) -> Generator:
        """
        Iterate over the a data store yielding the upper/lower/all values
        """
        if ultri not in ['upper', 'lower', 'all']:
            raise TypeError(
                "iteration type {} is not 'upper', 'lower', or 'all'".format(
                    ultri))

        it = np.nditer(self.data, flags=['multi_index'])
        while not it.finished:
            indices = it.multi_index
            left_idx_set = self.index_bijection(indices[:self.ndim // 2],
                                                self.ndim // 2, self.dim)
            right_idx_set = self.index_bijection(indices[self.ndim // 2:],
                                                 self.ndim // 2, self.dim)

            if ultri == 'upper' and left_idx_set <= right_idx_set:
                yield it[0], map(lambda x: self.basis.fwd(x), it.multi_index)
            elif ultri == 'lower' and left_idx_set >= right_idx_set:
                yield it[0], map(lambda x: self.basis.fwd(x), it.multi_index)
            elif ultri == 'all':
                yield it[0], map(lambda x: self.basis.fwd(x), it.multi_index)

            it.iternext()

    def vectorize(self, order: Optional[str] = 'C') -> np.ndarray:
        """
        Take a multidimensional array and vectorized via C ordering

        :return: a vector of self.size x 1
        """
        return np.reshape(self.data, (-1, 1), order=order)


# from standard library itertools recipe book
def grouper(iterable: Iterable,
            n: int,
            fillvalue: Optional[Union[None, str]] = None):
    """Collect data into fixed-length chunks or blocks"""
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)
