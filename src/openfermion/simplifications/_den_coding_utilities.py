import numpy as np


def global_decoders(decoder_list,qubits):
    raise NotImplementedError()

def global_encoders(enconder_list):
    num_encoders = len(enconder_list)
    num_global_row = 0
    num_global_col = 0
    for encoding in enconder_list:
        row,col = np.shape(encoding)
        num_global_row+=row
        num_global_col+=col

    global_encoding = np.zeros(shape=(num_global_row,num_global_col))
    raise NotImplementedError()

def f_linear_decoder(mtx):
    mtx = np.array(map(np.array,mtx))
    system_dim,code_dim = np.shape(mtx)
    res = ['']*system_dim
    for a in range(system_dim):
        for b in range(code_dim):
            if mtx[a,b]==1:
                res[a]+='w'+str(b)+' '

    return res


if __name__ == '__main__':
    print global_decoders(1,2)