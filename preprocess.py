import numpy as np

def preprocess(x):
    ''' preprocess the input X into a matrix of shape (n,m)
    Arguments:
    X ---- (1,m), a vector of raw data
    
    Returns:
    A0 ---- (n,m).
    '''
    
    A0 = np.ones([1,x.shape[1]])
    
    for i in range(3):
        A0 = np.vstack( (A0,np.power(x,3-i)) )
    
    return A0