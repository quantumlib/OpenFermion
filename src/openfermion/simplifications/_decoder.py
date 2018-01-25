# MODIFYING QUBIT OPERATOR CLASS


class DecoderError(Exception):
    pass

class Decoder(object):

    def __init__(self,term=None,coefficient=1.0):

        if not isinstance(coefficient, (int, float, complex)):
            raise ValueError('Coefficient must be a numeric type.')

        self.terms = {}
        if term is None:
            return

        elif isinstance(term, (tuple, list, int)):
            if isinstance(term, list):
                term = tuple(term)
            if isinstance(term, (int)):
                term = tuple([term])
            if term is ():
                self.terms[()] = coefficient
            else:
                # Test that input is a tuple of tuples and correct action
                list_ops = []
                for map_idx in term:
                    if isinstance(map_idx,int):
                        if map_idx < 0:
                            raise DecoderError("Invalid map qubit number "
                                               "must be a non-negative "
                                               "int.")
                        list_ops.append((map_idx,'w'))

                    if isinstance(map_idx,str):
                        op_name,qidx = map_idx
                        if op_name not in ['w','W']:
                            raise DecoderError("Invalid vector type, only"
                                               "accepting w")
                        list_ops.append((qidx,'w'))

                list_ops.sort(key=lambda loc_operator: loc_operator[0])
                self.terms[tuple(list_ops)] = coefficient
        elif isinstance(term, str):
            list_ops = []
            for el in term.split():
                if len(el) < 2:
                    raise ValueError('term specified incorrectly.')
                map_idx,op_kind = el[1:],el[0]
                if not (map_idx>=0 and op_kind in ['W','w']):
                    raise DecoderError("Invalid vector kind, must be 'w' ")
                list_ops.append((map_idx, 'w'))

            # Sort and add to self.terms:
            list_ops.sort(key=lambda loc_operator: loc_operator[0])
            self.terms[tuple(list_ops)] = coefficient
        else:
            raise ValueError('term specified incorrectly.')


if __name__ == '__main__':
    v = Decoder(1,1.0)
    print v.terms
    v = Decoder((),1.0)
    print v.terms
    v = Decoder((1,2,3),1.0)
    print v.terms
    v = Decoder('w1 w2 w3',0.5)
    print v.terms
