# MODIFYING QUBIT OPERATOR CLASS

import copy
class DecoderError(Exception):
    pass


class Decoder(object):

    def __init__(self,term=None,coefficient=1.0):

        if not isinstance(coefficient, (int, float, complex)):
            raise ValueError('Coefficient must be a numeric type.')

        self.terms = {}
        if term is None:
            return

        elif isinstance(term,dict):
            self.terms = term

        elif isinstance(term, (tuple, list, int)):
            if isinstance(term, list):
                term = tuple(term)
            if isinstance(term, int):
                term = tuple([term])
            if term is ():
                self.terms[()] = coefficient
                return
            else:
                # Test that input is a tuple of tuples and correct action
                list_ops = []
                for map_idx in term:
                    if isinstance(map_idx,int): # if term is given as (1,2,3)
                        if map_idx < 0:
                            raise DecoderError("Invalid map qubit number "
                                               "must be a non-negative "
                                               "int.")
                        list_ops.append((map_idx,'w'))

                    if isinstance(map_idx,tuple): # if term is given as ((1,'W'),(2,'W'))
                        qidx,op_name = map_idx
                        if op_name not in ['w','W']:
                            raise DecoderError("Invalid vector type, only"
                                               "accepting w")
                        list_ops.append((int(qidx),'w'))

                list_ops.sort(key=lambda loc_operator: loc_operator[0])
                self.terms[tuple(list_ops)] = coefficient

        elif isinstance(term, str):
            for t1 in  term.split(' + '): # given as ('1 + w1 w2 + w3 w4')
                list_ops = []
                constant = False
                for el in t1.split():
                    if constant: raise ValueError('cannot have coefficients within text, declare a separate decoder')
                    if len(el) == 1:
                        if () in self.terms:
                            if abs(self.terms[()]+float(el))>0:
                                self.terms[()] = self.terms[()]+float(el)*coefficient
                            else:
                                self.terms.pop(())
                        else:
                            self.terms[()] = coefficient
                        constant = True

                    elif len(el) == 2:
                        map_idx,op_kind = el[1:],el[0]
                        if not (map_idx>=0 and op_kind in ['W','w']):
                            raise DecoderError("Invalid vector kind, must be 'w' ")
                        list_ops.append((int(map_idx), 'w'))
                    else:
                        raise ValueError('term specified incorrectly.')
                    # Sort and add to self.terms:
                if not constant:
                    list_ops.sort(key=lambda loc_operator: loc_operator[0])
                    self.terms[tuple(list_ops)] = coefficient
        else:
            raise ValueError('term specified incorrectly.')

        for key, val in self.terms.items():
            for var in key:
                if len(var) not in [0, 2]: raise ValueError('every vector must be in the form (idx,w) or () for constants')
                if not isinstance(var[0], int): raise ValueError('qubit indices must be integers')
                if var[1] not in ['w', 'W']: raise ValueError('decoders can only be generated in w/W form')


    def __iadd__(self, addend):
        """
        In-place method for += addition of QubitOperator.

        Args:
          addend: A QubitOperator.

        Raises:
          TypeError: Cannot add invalid type.
        """
        if isinstance(addend,Decoder):
            for term in addend.terms:
                if term in self.terms:
                    if abs(addend.terms[term] + self.terms[term]) > 0.:
                        self.terms[term] += addend.terms[term]
                    else:
                        del self.terms[term]
                else:
                    self.terms[term] = addend.terms[term]
        else:
            raise TypeError('Cannot add invalid type to Decoder.')
        return self

    def __add__(self, addend):
        """ Return self + addend for a QubitOperator. """
        summand = copy.deepcopy(self)
        summand += addend
        return summand

    def to_str(self):
        str_form = ''
        for term,val in self.terms.items():
            str_form += '{}('.format(val)
            if term == ():
                str_form = str_form.strip('(')
                str_form+=' + '
                continue
            for var_i in term:
                str_form += '{}{}'.format(var_i[1],var_i[0])
            str_form+=') + '
        str_form = str_form.rstrip(' + ')
        return str_form


if __name__ == '__main__':
    p = Decoder(1,1.0)
    m = Decoder((),1.0)
    n = Decoder((2,3),1.0)
    print p.terms, '+', m.terms, '+', n.terms, ' + 0.5w1w2w3 = '
    v = Decoder('w1 w2 w3',0.5) + n + m + p
    print v.terms
    q = Decoder('1 + w1 w2',0.5)
    print q.terms
    print [v.to_str(),q.to_str()]
    z = v+q
    print z.to_str()