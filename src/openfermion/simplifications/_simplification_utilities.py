from openfermion.ops._qubit_operator import QubitOperator as QO
from _binary_operator import SymbolicBinary

def _extract(term):
    """
    extractor superoperator helper function

    Args:
        term: a single summand of th SymbolicBinary object

    Returns: a QubitOperator object or a constant

    """
    if len(term) == 1:
        term = term[0]
        if term[1]=='W':
            return QO('Z'+str(term[0]))
        elif term[1]=='1':
            return -1   # we only have 1/0s in SymbolicBinaries and zeros are not represented
    if len(term) >1:
        return dissolve(term)


def extractor(decode):
    """
    applies the extraction superoperator

    Args:
        decode: a SymbolicBinary object

    Returns: a QubitOperator object

    """
    return_fn = 1
    for term in decode.terms:
        return_fn*=_extract(term)
    return return_fn


def dissolve(term):
    """
    decomposition helper
    Args:
        term: a multi factor term

    Returns: QubitOperator

    """
    prod = 2.0
    for var in term:
        if var[1]!='W':
            raise ValueError('dissolve works on symbols W')
        prod*=(QO((),0.5) - QO('Z'+str(var[0]),0.5))
    return QO((),1.) - prod


def pauli_action(ham):
    # no need to implement. QubitOperator class already takes care of this
    raise NotImplementedError()


def bin_rules(ham):
    # no need to implement. BinarySymbolic class already takes care of this
    raise NotImplementedError()


def theta_fn(a,b):
    if a>b: return -1
    else: return 1



def delta_fn(a,b):
    if a==b: return -1
    else: return 1


def transform(fermion_op,decoder,encoder):
    raise NotImplementedError()


if __name__ == '__main__':
    d = SymbolicBinary('1 + w1 + w1 w2')
    print d.terms
    qq = extractor(d)
    print qq.terms
