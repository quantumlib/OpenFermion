from _binary_operator import SymbolicBinary
import numpy as np

class BinaryCodeError(Exception):
    pass

class BinaryCode(object):

    def __init__(self, encoding, decoding):

        if not isinstance(encoding, (np.array,list,tuple)):
            raise TypeError('encoding must be a list, array or tuple .')

        if not isinstance(decoding, (np.array,list,tuple)):
            raise TypeError('decoding must be a list, array or tuple .') 
        
        #transform the encoding into a numpy array extracting the total number qubits and orbits
        self.enc=np.array(map(np.array,encoding))
        self.dec=np.array(decoding)
        self.qubits,self.orbitals = np.shape(np.array(encoding))
        # np.reshape(self.enc,(self.qubits,self.orbits)) casting it to np.array should be enough assuming its a 2D array
        # since we extract the number of orbitals/qubits from this. Also shouldn't we enforce that the user gives
        # qubits x orbitals shaped encoder?
        
        #cast decoding into array form
        
        if(self.orbitals!=len(self.dec)):
            raise BinaryCodeError('size mismatch, decoder and encoder should have the same first dimension')
            # some error that tells you about the dimensional mismatch in the orbit number
        
        for component in np.arange(self.orbitals):
            
            #Case 1: component is an array or a list  ---

            # ok there is an issue : do we want to allow for the stuff to be given as [((1,'W'),(2,'W'), ....] or not ?

            # how about this: we only allow components that are acceptable for SymbolicBinary. that way if the user
            # gives a list/text as a component, first we convert to symbolicBinary and do operations later :
            if isinstance(self.dec[component], (np.array, tuple, list)):
                self.dec[component] = SymbolicBinary(self.dec[component])

            component_qubits = self.dec[component].count_qubits() # do we enforce that the user gives qubit numbers starting
            # from 0. if they didn't, then our decoder will have an all zero-column.
            number_of_qubits = max(component_qubits)+1

                    
        
                    
                       
            
                
        
        
