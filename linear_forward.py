import numpy as np

def linear_forward(W,A):
    '''
    Arguments:
    W -- (n,1) , n is the number of features in X
    A -- (n,m), the result of previously preprocess of X
    
    Returns:
    Z -- (1,m), the vector of output
    '''

    Z = np.dot(W.T,A)

    return Z 