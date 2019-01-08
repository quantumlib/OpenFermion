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


class MajoranaOperator:

    def __init__(self, term=None, coefficient=1.0):
        """Initialize a MajoranaOperator with a single term.

        Args:
            term (Tuple[int]): The indices of a Majorana operator term
                to start off with
            coefficient (complex): The coefficient of the term

        Returns:
            MajoranaOperator
        """
        self.terms = {}
        if term is not None:
            term, parity = _sort_majorana_term(term)
            self.terms[term] = coefficient * (-1)**parity

    def commutes_with(self, other):
        """Test commutation with another MajoranaOperator"""
        if not isinstance(other, type(self)):
            return NotImplemented

        if len(self.terms) == 1 and len(other.terms) == 1:
            return _majorana_terms_commute(list(self.terms.keys())[0],
                                           list(other.terms.keys())[0])
        # TODO
        return False


def _sort_majorana_term(term):
    """Sort a Majorana term.

    Args:
        term (Tuple[int]): The indices of a Majorana operator term

    Returns:
        Tuple[Tuple[int], int]. The first object returned is a sorted list
        representing the indices acted upon. The second object is the parity
        of the term. A parity of 1 indicates that the term should include
        a minus sign.
    """
    if len(term) < 2:
        return term, 0
    center = len(term) // 2
    left_term, left_parity = _sort_majorana_term(term[:center])
    right_term, right_parity = _sort_majorana_term(term[center:])
    merged_term, merge_parity = _merge_majorana_terms(left_term, right_term)
    return merged_term, (left_parity + right_parity + merge_parity) % 2


def _merge_majorana_terms(left_term, right_term):
    merged_term = []
    parity = 0
    i, j = 0, 0
    while i < len(left_term) and j < len(right_term):
        if left_term[i] < right_term[j]:
            merged_term.append(left_term[i])
            i += 1
        elif left_term[i] > right_term[j]:
            merged_term.append(right_term[j])
            j += 1
            parity += len(left_term) - i
        else:
            parity += len(left_term) - i - 1
            i += 1
            j += 1
    if i == len(left_term):
        merged_term.extend(right_term[j:])
    else:
        merged_term.extend(left_term[i:])
    return tuple(merged_term), parity % 2


def _majorana_terms_commute(term_a, term_b):
    """Whether two Majorana terms commute.

    Args:
        term_a (Tuple[int]): The indices of a Majorana operator term
        term_b (Tuple[int]): The indices of a Majorana operator term

    Returns:
        bool. Whether The terms commute.
    """
    intersection = 0
    i, j = 0, 0
    while i < len(term_a) and j < len(term_b):
        if term_a[i] < term_b[j]:
            i += 1
        elif term_a[i] > term_b[j]:
            j += 1
        else:
            intersection += 1
            i += 1
            j += 1
    parity = (len(term_a)*len(term_b) - intersection) % 2
    return not parity
