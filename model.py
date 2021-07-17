# coding: utf-8

import preprocess
import numpy as np
import cost
import backpropagation
import linear_forward

def model(x,y,iteration,learning_rate):
    '''
    This function put everything together
    Arguements:
    x -- (1,m) a vector of input
    y -- (1,m) a vector of true label value
    iteration -- a scalar, control how many times will this algorithm learn to fit this function
    learning_rate -- a scalar, control how big the step is in gradient descent 
    
    Returns:
    W1 -- (4,1) the learnt parameters
    cache -- a dictionary stores dW1 and cost after every iteration
    '''
    
    #Do the preprocess step to x
    A0 = preprocess.preprocess(x)
    
    #Randomly initialize parameter W1
    W1 = np.random.rand(4,1)
    
    #create the cache list to store cache
    cache = {'cost':[],
             'dW1':[]}
    
    #train 
    for i in range(iteration):
        y_hat = np.sum(linear_forward.linear_forward(W1,A0),0)
        costs = cost.cost(y_hat,y)
        dW1 = backpropagation.backpropagation(y_hat,y,A0)
        #update W1
        W1 = W1 - learning_rate * dW1
        #store data in cache
        cache['cost'].append(costs)
        cache['dW1'].append(dW1)

    return cache,W1

