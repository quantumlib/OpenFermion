import numpy as np
from _decoder import Decoder

def encoder_bk(d):
    reps = np.ceil(np.log2(d))
    mtx = [[1,0],[1,1]]
    for a in np.arange(1,reps+1):
        mtx = np.kron(np.eye(2),mtx)
        for b in np.arange(0,2**a+1):
            mtx[int(2**a),b]=1
    return mtx[0:d,0:d]

def decoder_bk(d):
    reps = np.ceil(np.log2(d))
    mtx = [[1, 0], [1, 1]]
    for a in np.arange(1,reps+1):
        mtx = np.kron(np.eye(2),mtx)
        mtx[int(2**a),int(2**(a-1))]=1
    return mtx[0:d,0:d]

def decoder_jw(d):
    return [Decoder('w'+str(i),1.0) for i in range(d)]

def encoder_checksum(sites):
    enc = np.zeros(shape=(sites-1,sites))
    for i in range(sites - 1): enc[i,i] = 1
    return enc

def decoder_checksum(sites,odd):
    if odd==1: all_in = Decoder((),1.0)
    else: all_in = Decoder((),0.0)

    for a in range(sites-1):
        all_in += Decoder('w'+str(a),1.0)

    djw = decoder_jw(sites-1)
    djw.append(all_in)
    return djw

if __name__=='__main__':
    print [x.to_str() for x in decoder_checksum(4,1)]


