import numpy as np

# TODO: modify to use the decoder object

def _global_decoder_rule(decode,runner):
    decode_shifted = []
    for entry in decode:
        entry_elements = entry.split(' ')
        tmp_entry = ''
        for element in entry_elements:
            if element.startswith('w'):
                qubit_mapping = int(element[1:])+runner
                tmp_entry+='w'+str(qubit_mapping)+' '
            else:
                tmp_entry+=element+' '
        decode_shifted.append(tmp_entry.rstrip())
    return decode_shifted

def global_decoders(decoder_list,qubits):

    if len(qubits)!=len(decoder_list):
        raise ValueError('size of qubits and decoder list must be the same')

    global_decoder = []
    runner = 0
    for qubit_idx,qubit in enumerate(qubits):
        runner+=qubit
        [global_decoder.append(term) for term in _global_decoder_rule(decoder_list[qubit_idx],runner)]

    return global_decoder

def global_encoders(encoder_list):

    num_global_row,num_global_col = np.sum([np.shape(encoding) for encoding in encoder_list],axis=1)
    global_encoding = np.zeros(shape=(num_global_row,num_global_col))

    row_idx = 0
    col_idx = 0
    for encoding in encoder_list:
        row_enc,col_enc = np.shape(encoding)
        global_encoding[row_idx:row_idx+row_enc,col_idx:col_idx+col_enc] = encoding
        row_idx += row_enc
        col_idx += col_enc
    return global_encoding

def f_linear_decoder(mtx):
    mtx = np.array(map(np.array,mtx))
    system_dim,code_dim = np.shape(mtx)
    res = ['']*system_dim
    for a in range(system_dim):
        for b in range(code_dim):
            if mtx[a,b]==1:
                res[a]+='w'+str(b)+' '
        res[a] = res[a].rstrip()
    return res


if __name__ == '__main__':
    enclist = [[[1,1],[1,1]],[[1,1,0,0],[1,0,0,1]]]
    print global_encoders(enclist)
    decoder_list = [['w1', 'w2', '1 w1 w2'], ['w1', 'w1 w2'], ['w1']]
    qubits = [0, 3, 4]
    print global_decoders(decoder_list,qubits)
    print f_linear_decoder([[0,0,0,1],[1,0,1,0]])