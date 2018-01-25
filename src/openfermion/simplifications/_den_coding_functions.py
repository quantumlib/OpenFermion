import numpy as np


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
    return ['w'+str(i) for i in range(d)]

def encoder_checksum(sites):
    enc = np.zeros(shape=(sites-1,sites))
    for i in range(sites - 1): enc[i,i] = 1
    return enc

def decoder_checksum(sites,odd):
    if odd==1: all_in = '1 '
    else: all_in = ''
    for a in range(sites-1):
        all_in += ('w'+str(a)+' ')

    all_in = all_in.rstrip()
    djw = decoder_jw(sites-1)
    djw.append(all_in)
    return djw

if __name__=='__main__':
    print decoder_checksum(4,1)


