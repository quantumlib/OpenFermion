def extractor(term):
    # if term.isinstance(str):
    #     elements = term.split(' ')
    #     extracted = 1
    #     if len(elements)>1: # take care of summation
    #         for element in elements:
    #             extracted*=extractor(element)
    #     else:
    #         element = elements[0]
    #         if element.count('w')>1:
    #             extracted*=dissolve(element)
    #         if element.count('w')==1:
    #             extracted*=
    raise NotImplementedError()




def dissolve(ham):
    raise NotImplementedError()



def pauli_action(ham):
    raise NotImplementedError()



def bin_rules(ham):
    raise NotImplementedError()



def theta_fn(a,b):
    if a>b: return -1
    else: return 1



def delta_fn(a,b):
    if a==b: return -1
    else: return 1

