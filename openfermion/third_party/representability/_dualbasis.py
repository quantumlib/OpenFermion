from typing import List, Optional, Union, Tuple
import copy


class DualBasisElement(object):
    """
    This object is named after the algebraic dual space or dual vector space

    Given a vector x in a vector space V we define the dual basis element
    any (A, b, c) that satisfies

    Ax + b = c

    This object allows a user to specify a linear operator on a space of tensors
    that is vectorized.  Each tensor has a name.  The user constructs `A' and
    provides b and c.  For example if I have a tensor that happens to be a
    matrix called `M'.  The trace operation is an operator `A' where A[i, i] = 1
    for all i in [1, dim(`M')].
    """

    def __init__(self,
                 *,
                 tensor_names: Optional[Union[None, List[str]]] = None,
                 tensor_elements: Optional[
                     Union[None, List[Tuple[int, ...]]]] = None,
                 tensor_coeffs: Optional[Union[None, List[float]]] = None,
                 bias: Optional[int] = 0,
                 scalar: Optional[int] = 0):
        """
        Define a linear operator on a tensor `A', a bias `b', and a result `c'
        satisfying

        .. math::
            Ax + b = c

        Args:
            tensor_names: Names of tensor subspace to act on.
            tensor_elements: List of tuples of numerical tensor indices.
            tensor_coeffs: List of the coefficients corresponding to values to
                           put in `A' at locations `tensor_elements'.
            bias: a linear shift to apply
            scalar: the resulting scalar when the inner product is taken with
                    some element of the tensors `{x}'.
        """
        if tensor_names is None:
            self.primal_tensors_names = []
        else:
            self.primal_tensors_names = tensor_names

        if tensor_elements is None:
            self.primal_elements = []
        else:
            self.primal_elements = tensor_elements

        if tensor_coeffs is None:
            self.primal_coeffs = []
        else:
            self.primal_coeffs = tensor_coeffs

        self.constant_bias = bias
        self.dual_scalar = scalar

    def add_element(self, name, element, coeff):
        """
        Load an element into the data structure
        """
        if not isinstance(element, tuple):
            raise TypeError("element needs to be a tuple of indices")
        if not isinstance(name, str):
            raise TypeError("name needs to be a string")
        if not isinstance(coeff, (float, int)):
            raise TypeError("coeff needs to be a non-complex numerical value")

        self.primal_elements.append(element)
        self.primal_coeffs.append(coeff)
        self.primal_tensors_names.append(name)

    def join_elements(self, other):
        """
        Join two DualBasisElements together to form another DualBasisElement

        Args:
            DualBasisElement other: to join with self
        """
        if not isinstance(other, DualBasisElement):
            raise TypeError("I can only join two DualBasisElements together")

        dbe = DualBasisElement()
        dbe.primal_tensors_names.extend(other.primal_tensors_names)
        dbe.primal_tensors_names.extend(self.primal_tensors_names)

        dbe.primal_elements.extend(other.primal_elements)
        dbe.primal_elements.extend(self.primal_elements)

        dbe.primal_coeffs.extend(other.primal_coeffs)
        dbe.primal_coeffs.extend(self.primal_coeffs)

        dbe.constant_bias += self.constant_bias + other.constant_bias
        dbe.dual_scalar += self.dual_scalar + other.dual_scalar

        return dbe.simplify()

    def simplify(self):
        """
        Mutate the DualBasisElement so that non-unique terms get summed together
        """
        id_dict = {}
        for tname, telement, tcoeff in zip(self.primal_tensors_names,
                                           self.primal_elements,
                                           self.primal_coeffs):
            id_str = tname + ".".join([str(x) for x in telement])
            if id_str not in id_dict:
                id_dict[id_str] = (tname, telement, tcoeff)
            else:
                id_dict[id_str] = (tname, telement, id_dict[id_str][2] + tcoeff)

        tnames = []
        telements = []
        tcoeffs = []
        for _, el in id_dict.items():
            tnames.append(el[0])
            telements.append(el[1])
            tcoeffs.append(el[2])

        self.primal_coeffs = tcoeffs
        self.primal_tensors_names = tnames
        self.primal_elements = telements

        return self

    def id(self):
        """
        Get the unique string identifier for the dual basis element
        """
        id_str = ""
        for name, element in zip(self.primal_tensors_names,
                                 self.primal_elements):
            id_str += name + "(" + ",".join([repr(x) for x in element]) + ")\t"
        return id_str

    def __iter__(self):
        for t_label, velement, coeff in zip(self.primal_tensors_names,
                                            self.primal_elements,
                                            self.primal_coeffs):
            yield t_label, velement, coeff

    def __add__(self, other):
        if isinstance(other, DualBasisElement):
            return DualBasis(elements=[self, other])
        elif isinstance(other, DualBasis):
            return other + self
        else:
            raise TypeError(
                "DualBasisElement can be added to same type or DualBasis")


class DualBasis(object):

    def __init__(
            self,
            elements: Optional[Union[None, List[DualBasisElement]]] = None):
        """
        A collection of DualBasisElements

        Args:
            elements: (optional) a list of DualBasisElement objects
        """

        if elements is None:
            self.elements = []
        else:
            if all(map(lambda x: isinstance(x, DualBasisElement), elements)):
                self.elements = elements
            else:
                raise TypeError("elements must all be DualBasisElement objects")

    def __iter__(self):
        return self.elements.__iter__()

    def __getitem__(self, index):
        return self.elements[index]

    def __len__(self):
        return len(self.elements)

    def __add__(self, other):
        if isinstance(other, DualBasisElement):
            new_elements = copy.deepcopy(self.elements)
            new_elements.append(other)
            return DualBasis(elements=new_elements)

        elif isinstance(other, DualBasis):
            new_elements = copy.deepcopy(self.elements)
            new_elements.extend(other.elements)
            return DualBasis(elements=new_elements)

        else:
            raise TypeError(
                "DualBasis adds DualBasisElements or DualBasis only")
