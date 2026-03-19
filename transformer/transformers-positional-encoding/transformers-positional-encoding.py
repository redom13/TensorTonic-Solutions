import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    #--------------------Naive Implementation-----------------------#
    # pe = np.empty((seq_length,d_model))

    # for pos in range(seq_length):
    #     for i in range(d_model):
    #         if i%2==0:
    #             arg = pos/(10000**(i/d_model))
    #             pe[pos][i] = np.sin(arg)
    #         else:
    #             arg = pos/(10000**(i-1/d_model))
    #             pe[pos][i] = np.cos(arg)

    # return pe
    #---------------------Vectorized Impl----------------------------#
    pos_indices = np.arange(seq_length).reshape(-1,1)
    arg_indices = np.arange(0,d_model,2).reshape(1,-1) # [[0,2,....,d_model//2]] , shape: (1,d_model//2)

    phases = pos_indices/np.power(10000,arg_indices/d_model) # Broadcasting magic

    pe = np.empty((seq_length,d_model)) # np.empty faster than np.zeros or ones

    pe[:,0::2] = np.sin(phases)
    pe[:,1::2] = np.cos(phases)

    return pe