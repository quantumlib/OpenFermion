from openfermion.ops._qubit_operator import QubitOperator as QO
from _decoder import Decoder

def _extract(term,coeff):
    if len(term) == 0: #constant
        return QO((),(-1)**(coeff%2))
    if len(term) == 1:
        term = term[0]
        if term[1] not in ['w','W']: raise ValueError('the decoder function type unknown must be w/W')
        return QO('Z'+str(term[0]),coeff)
    if len(term) >1:
        return dissolve(term)

def extractor(decode):
    return_fn = 1
    for term,coeff in decode.terms.items():
        return_fn*=_extract(term,coeff)
    return return_fn

def dissolve(term):
    prod = QO((),2.)
    for var in term:
        prod*=(QO((),0.5) - QO('Z'+str(var[0]),0.5))
    return QO((),1.) - prod


def pauli_action(ham):
    # no need to implement. QubitOperator class already takes care of this
    raise NotImplementedError()



def bin_rules(ham):
    raise NotImplementedError()



def theta_fn(a,b):
    if a>b: return -1
    else: return 1



def delta_fn(a,b):
    if a==b: return -1
    else: return 1


if __name__ == '__main__':
    d = Decoder('1 + w1 + w1 w2',1.0)
    print d.to_str()
    qq = extractor(d)
    print qq.terms