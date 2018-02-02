import numpy as np
from _binary_operator import SymbolicBinary

def _global_decoder_rule(decode,runner):
    """
    global decoder helper
    Args:
        decode: a symbolicBinary object
        runner: the qubit index that corresponds to the decoder

    Returns: a shifted decoder

    """
    decode_shifted = []
    for entry in decode:
        if isinstance(entry,str):
            entry = SymbolicBinary(entry)
        if isinstance(entry,SymbolicBinary):
            list_ops = []
            for element in entry.terms:
                if isinstance(element[0],int): # sole term in symbolic binary
                    qidx, op_name = element
                    qubit_mapping = qidx + runner
                    tmp_element=[(qubit_mapping,op_name)]
                else:
                    tmp_element = []
                    for var in element:
                        qidx, op_name = var
                        qubit_mapping = qidx + runner
                        tmp_element.append((qubit_mapping,op_name))
                list_ops.append(tuple(tmp_element))
            decode_shifted.append(SymbolicBinary(list_ops))

    return decode_shifted

def global_decoders(SymbolicBinary_list,qubits):
    """
        appends a list of decoding functions to a global decoding
        Args:
            SymbolicBinary_list: list of SymbolicBinary object, each decoding is again a list of the components
            qubits: list of integers that give the dimension of the qubit space for the respective decoding

        Returns:

        """
    if len(qubits)!=len(SymbolicBinary_list):
        raise ValueError('size of qubits and SymbolicBinary list must be the same')

    global_SymbolicBinary = []
    runner = 0
    for qubit_idx,qubit in enumerate(qubits):
        [global_SymbolicBinary.append(term) for term in _global_decoder_rule(SymbolicBinary_list[qubit_idx],runner)]
        runner+=qubit

    return global_SymbolicBinary

def global_encoders(encoder_list):
    """
    appends a list of encoding matrices  to a global encoding
    Args:
        encoder_list:list of encoding matrices, stored as 2 - dimensional lists, so the entire object has
        the dimension 3

    Returns: a global encoding matrix

    """
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
    """
    outputs a linear decoding function when given a matrix
    Args:
        mtx: matrix to derive the decoding function from

    Returns: a list of symbolicBinaries

    """
    mtx = np.array(map(np.array,mtx))
    system_dim,code_dim = np.shape(mtx)
    res = []*system_dim
    for a in range(system_dim):
        dec_str = ''
        for b in range(code_dim):
            if mtx[a,b]==1:
                dec_str+='W'+str(b)+' + '
        dec_str = dec_str.rstrip(' + ')
        res.append(SymbolicBinary(dec_str))
    return res


if __name__ == '__main__':
    enclist = [[[1,1],[1,1]],[[1,1,0,0],[1,0,0,1]]]
    print global_encoders(enclist)
    SymbolicBinary_list = [[SymbolicBinary('w1'), SymbolicBinary('w2'), SymbolicBinary('1 + w1 + w2')], [SymbolicBinary('w1'), SymbolicBinary('w1 w2')], [SymbolicBinary('w1')]]
    qubits = [1, 3, 4]
    print [x.terms for x in global_decoders(SymbolicBinary_list,qubits)]
    linear_decoded = f_linear_decoder([[0,0,0,1],[1,0,1,0]])
    print [x.terms for x in linear_decoded]
