#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np

def cost(y_hat,y):
    '''
    Compute the cost between predict output y_hat and the real value y
    Arguements:
    y_hat -- (1,m), predicted output
    y -- (1,m), real value
    
    Returns:
    cost -- a scalar.
    '''
    
    y_hat = np.array(y_hat)
    y = np.array(y)
    cost = np.sum(np.power((y_hat-y),2)) / len(y)
    
    return cost


# In[17]:





# In[ ]:




