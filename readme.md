# REGRESSION FUNCTION FITTING

This repository implements a regression task. I set a neural network to fit a quadratic function. By doing so, I can check if my thoughts of full connected neural network are all right.

***

#### DATA SETS

How do I get the data sets for training and test?

It is much easier than what I think to get the datasets.

I will just use a binary function to generate some (x,y) pairs and add some noise (x,y) pairs as data sets.
Using numpy.randn() to add noise: Y = a1*X^2 + a2*X +a3 +numpy.rand.randn()

#### NEURAL NETWORK STRUCTURE
Input -> Hidden layer -> Output.
##### Details
Optimise it with gradient descent<br>
Forward:X -> X (4,m) -> W (4,1)*X -> Y (1,m)<br>
Backward: cost(MSE) -> gradA<br>
Update: A = A - learning_rate * gradA<br>

##### Get Gradient

Below are my calculate form of how to get W1's gradient,

$ cost=\frac{\sum_{i=1}^{m}{(ŷ_i-y_i)}^2}{m} $

$ \frac{∂cost} {∂ŷ} = \frac{2}{m}\sum_{i=1}^m {(ŷ_i-y_i)}$

$ \frac{∂cost}{∂W1} = \frac{∂cost}{∂ŷ}\frac{∂ŷ}{∂W1} = (\frac{2}{m}\sum_{i=1}^{m}{(ŷ_i-y)} )\cdot A_0 $

so, after every iteration of learning, I will simultaneously update my parameter W1 like follows,

$ W1 := W1 - \alpha\cdot\frac{∂cost}{∂W1}$ 



Above is my BFF project.

***

11th July, 2021

Today I just finished the basic framework of my nearual network, just some linear things in it. Everything goes on as I wanted. But there still some bugs need to be finished.

* Learn how to choose a proper iteration figure and a proper learning_rate. 
* Do error analyze on my framework.
* Standardize my output.

