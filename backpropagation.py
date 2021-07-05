#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np

def backpropagation(y_hat,y,A_prev):
    '''
    This function implyment backpropagation
    Arguments:
    y_hat -- (1,m) the predicted output
    y -- (1,m) the real value
    A_prev -- previous layers neurals
    
    Returns:
    dW1 -- the same dimesions with W1,every element in this vector is gradient for every element in W1
    '''
    y_hat = np.array(y_hat)
    y = np.array(y)
    A_prev = np.array(A_prev)
    m = len(y_hat)
    
    dy = 2 * np.sum(y_hat-y) /m
    dW1 = dy * A_prev
    
    return dW1


# In[7]:





# In[ ]:




