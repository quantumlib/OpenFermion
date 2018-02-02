from _binary_operator import SymbolicBinary
import numpy as np

class BinaryCode(Exception):
    pass

class BinaryCode(object):

    __init__(self, encoding, decoding):

        if not isinstace(encoding, (np.array,list,tuple)):
            raise TypeError('encoding must be a list, array or tuple .')

        if not isinstace(decoding, (np.array,list,tuple)):
            raise TypeError('decoding must be a list, array or tuple .') 
        
        #transform the encoding into a numpy array extracting the total number qubits and orbits
        self.orbits=len(encoding)
        self.enc=encoding
        self.qubits=len(np.array(self.enc[0]))
        np.reshape(self.enc,(self.qubits,self.orbits))
        
        #cast decoding into array form
        
        self.dec=array(decoding)
        if(self.orbits!=len(self.dec)):
            # some error that tells you about the dimensional mismatch in the orbit number
        
        for component in np.arange(self.orbits):
            
            #Case 1: component is an array or a list  --- ok there is an issue : do we want to allow for the stuff to be given as [((1,'W'),(2,'W'), ....] or not ?
            if isinstance(self.dec[component], (np.array, tuple, list)):
                if(len(component)!=self.qubits):
                     # some error that tells you about the dimensional mismatch in the qubit number
                else:
                    
                    orbitaldecoding=[]
                    for col in self.dec[components]:
                        if(np.array(self.dec[component])[col]): orbitaldecoding+=[((component,'W'))]
                    self.dec[component]=SymbolicBinary(orbitaldecoding)
            #Case 2: component is already SymbolicBinary
            elif(isinstance(self.dec[component], SymbolicBinary)):
                    # do we have a tool to measure to how many qubits it infers ?
            
            #Case 3: component is str       
            elif(isinstance(self.dec[component], str)):
                self.dec[component]=SymbolicBinary(self.dec[component])
                
                # to be continiued ... 
                    
                    
        
                    
                       
            
                
        
        
