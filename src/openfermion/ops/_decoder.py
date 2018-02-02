from openfermion.config import EQ_TOLERANCE
from openfermion.ops import SymbolicOperator


class DecoderError(Exception):
    pass


class Decoder(SymbolicOperator):

    actions = ('w')
    action_strings = ('w')
    action_before_index = True
    different_indices_commute = True

if __name__ =='__main__':
    p = Decoder(1,1.0)
    m = Decoder((),1.0)
    n = Decoder((2,3),1.0)
