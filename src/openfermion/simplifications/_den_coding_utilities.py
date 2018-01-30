import numpy as np
from _decoder import Decoder

def _global_decoder_rule(decode,runner):
    decode_shifted = []
    for entry in decode:
        if isinstance(entry,str):
            entry = Decoder(entry)
        if isinstance(entry,Decoder):
            list_ops = {}
            for element,coeff in entry.terms.items():
                tmp_element = []
                for var in element:
                    qidx, op_name = var
                    qubit_mapping = qidx + runner
                    tmp_element.append((qubit_mapping,op_name))
                list_ops[tuple(tmp_element)] = coeff
            decode_shifted.append(Decoder(list_ops))

    return decode_shifted

def global_decoders(decoder_list,qubits):

    if len(qubits)!=len(decoder_list):
        raise ValueError('size of qubits and decoder list must be the same')

    global_decoder = []
    runner = 0
    for qubit_idx,qubit in enumerate(qubits):
        [global_decoder.append(term) for term in _global_decoder_rule(decoder_list[qubit_idx],runner)]
        runner+=qubit

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
    res = []*system_dim
    for a in range(system_dim):
        dec_str = ''
        for b in range(code_dim):
            if mtx[a,b]==1:
                dec_str+='w'+str(b)+' + '
        dec_str = dec_str.rstrip(' + ')
        res.append(Decoder(dec_str,1.0))
    return res


if __name__ == '__main__':
    enclist = [[[1,1],[1,1]],[[1,1,0,0],[1,0,0,1]]]
    print global_encoders(enclist)
    decoder_list = [[Decoder('w1'), Decoder('w2'), Decoder('1 + w1 + w2')], [Decoder('w1'), Decoder('w1 w2')], [Decoder('w1')]]
    qubits = [1, 3, 4]
    print [x.to_str() for x in global_decoders(decoder_list,qubits)]
    linear_decoded = f_linear_decoder([[0,0,0,1],[1,0,1,0]])
    print [x.to_str() for x in linear_decoded]