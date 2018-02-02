from _binary_operator import SymbolicBinary
import numpy as np

class BinaryCodeError(Exception):
    pass

class BinaryCode(object):

    def __init__(self, encoding, decoding):

        if not isinstance(encoding, (list,tuple)):
            raise TypeError('encoding must be a list, array or tuple .')

        if not isinstance(decoding, (list,tuple)):
            raise TypeError('decoding must be a list, array or tuple .') 
        
        #transform the encoding into a numpy array extracting the total number qubits and orbits
        self.enc=np.array(map(np.array,encoding))
        self.qubits,self.orbitals = np.shape(np.array(encoding))
        # np.reshape(self.enc,(self.qubits,self.orbits)) casting it to np.array should be enough assuming its a 2D array
        # since we extract the number of orbitals/qubits from this. Also shouldn't we enforce that the user gives
        # qubits x orbitals shaped encoder?

        if(self.orbitals!=len(decoding)):
            raise BinaryCodeError('size mismatch, decoder and encoder should have the same first dimension')

        decoder_qubits = set()
        self.dec = []
        
        # build the decoder - question: what if its given as a matrix... we should be able to handle that
        for symbolic_binary in np.array(decoding):
            # ok there is an issue : do we want to allow for the stuff to be given as [((1,'W'),(2,'W'), ....] or not ?

            # how about this: we only allow components that are acceptable for SymbolicBinary. that way if the user
            # gives a list/text as a component, first we convert to symbolicBinary and do operations later :
            if isinstance(symbolic_binary, (tuple, list, str)):
                symbolic_binary = SymbolicBinary(symbolic_binary)
            if isinstance(symbolic_binary,SymbolicBinary):
                self.dec.append(symbolic_binary)
                decoder_qubits = decoder_qubits | set(symbolic_binary.count_qubits())
            else:
                raise TypeError('decoder component provided is not a suitable for symbolicBinary',symbolic_binary)
        self.decoder_qubits = decoder_qubits



if __name__ == '__main__':
    encoder = [[0,1,0,0],[0,0,0,1],[1,0,0,0]]
    decoder = ['1 + w1', 'w0', 'w1 + w2','w2']
    bc = BinaryCode(encoder,decoder)
    print bc.decoder_qubits
    print bc.dec

                    
        
                    
                       
            
                
        
        
