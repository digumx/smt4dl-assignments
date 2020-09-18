"""

To implement a simple deep neural network class with a regularization parameter

"""

import numpy as np
import random
import math

"""### Few helper functions for the Neural Network"""

def relu(x):
    # input is numpy vector
    return np.maximum(x,0)


vfunc = np.vectorize(lambda x : 1 if x > 0 else 0,  otypes=[np.float])  ## vectorizing allows us to
                                                                        ## apply arbitrary function to all the element of a numpy vector
def relu_derivative(x):
    # input is numpy vector
    return vfunc(x)
    #return (x > 0).astype(int)  # this replaces all the positive value by 1 and rest by 0

def sigmoid(x):
    # input is numpy vector
    return 1.0/(1.0+np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1-sigmoid(x))

def mean_square_derivative(output_activations, y):
    return (output_activations-y)

"""# Network Class
#### Finish the code of the Network Class. You are free to add/remove attributes as required. Make
sure that the signature of **\_\_init\_\_()** and **SGD** methods are not changed as they will be
called in the later part of the notebook

---
After writing the class, run the rest of the code and verify it works properly
"""

class Network:
    def __init__(self, sizes, 
                    activation_func = sigmoid,  
                    activation_derivative = sigmoid_derivative, 
                    regularization = 0):
        """
        Objective: Intialize the network
        Input:sizes - tuple representing number of neurons in each layer from left to right
              Note: sizes[0] is the number of inputs: they are not neurons per se
              E.g., sizes = (5,4,2) means that there are 5 inputs, 4 neurons in the first hidden layer, and 2 neuron in the ouput layer
        """
        self.train_cost_history= []    ## Can be used to store costs w.r.t training data w.r.t iteration number
        self.test_cost_history = []    ## Can be used to store costs w.r.t testing data w.r.t iteration number

        self.activation = activation_func
        self.activation_derivative = activation_derivative
        self.cost_derivative = mean_square_derivative
        
        self.num_layers = len(sizes)   ### Number of layers is length of list sizes
        
        self.biases = [np.random.randn(number_of_neurons, 1) for number_of_neurons in sizes[1:]]
        
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])] 

        # We store lambda/n here
        print("DEBUG: ", sum(x*y for x,y in zip(sizes[:-1], sizes[1:])))
        self.regularization = regularization / sum(x*y for x,y in zip(sizes[:-1], sizes[1:]))
   

    def feedforward(self, x):
        """
        Return the output of the network for the input 'x'.
        """
        for b, w in zip(self.biases, self.weights):
            x = self.activation(np.dot(w, x)+b)
        return x
   

    def SGD(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        """
        Objective: Stochastic Gradient Descent
        1. tranining_data is a python list of tuples of the form (x, y) 
        where x is the input vector (numpy matrix of size (n x 1)) and y is the output label
        2. epochs is the number of epochs
        3. mini_batch_size is the size of single mini-batch on which we will perform the backpropogation
        4. learning_rate is learning rate!
        5. test_data is optional in the same format as training data: can be used to calculate accuracy on each
        iteration. (Ignore test_data while implementing for first time)
        """
        num_batches = len(training_data) // mini_batch_size                     # This must be int
        for it in range(epochs):
            print("Epoch ", it+1, "/", epochs)

            # Shuffle training data
            random.shuffle(training_data)

            # Split into batches
            batches = (training_data[i:i+mini_batch_size] for i in 
                            range(0, len(training_data), mini_batch_size))
            
            # Backprop on each batch
            for batch,batch_num in zip(batches, range(1, num_batches+1)):
                print("Batch ", batch_num, "/", num_batches, end='\r')
                self.apply_backprop_on_batch(batch, learning_rate)
            print()

            # Calculate training and testing costs and add to history
            if(test_data != None):
                cost_train = self.test_accuracy(training_data)
                print("Training cost: ", cost_train) 
                self.train_cost_history.append(cost_train) 
                cost_test = self.test_accuracy(test_data)
                print("Testing cost: ", cost_test) 
                self.test_cost_history.append(cost_test) 
        
    
    def apply_backprop_on_batch(self, batch, learning_rate):
        """
        Applies the backprop on the given batch and updates weights and biases
        
        """
        gradient_biases = [np.zeros(b.shape) for b in self.biases]
        gradient_weights = [np.zeros(w.shape) for w in self.weights]
        
        for x,y in batch:
            # x, y is one training example
            gradient_biases_on_example, gradient_weights_on_example = self.apply_backprop_on_example(x,y)
            gradient_biases = [db + dbi for db,dbi in                                       # Sum up 
                                zip(gradient_biases, gradient_biases_on_example)]
            gradient_weights = [dw + dwi for dw,dwi in 
                                zip(gradient_weights, gradient_weights_on_example)]
        map(lambda m: m * (learning_rate/len(batch)), gradient_biases)                       # Divide
        map(lambda m: m * (learning_rate/len(batch)), gradient_weights)                      # and
                                                                                             # Normalize
        # Update weights and biases
        self.weights = [w - dw - (self.regularization*w) for w,dw in zip(self.weights, gradient_weights)]
        self.biases = [b - db for b,db in zip(self.biases, gradient_biases)]
   

    def apply_backprop_on_example(self, x, y ):
        """
        Applies backpropagation and calculates partial derivates for all the weights and biases
        for one given example (x,y)
        NOTE: It does NOT updates weights/biases. That is done by the caller function
        """
        # First we perform a feed-forward to get the values of the inputs to each layer. We repeat
        # the code here because it also appends to a list, an operation which is unnecessary in the
        # general feed forward. We also store the activation function derivatives.
        layer_inputs = []
        output_activations = x
        activation_function_derivs = []
        for w, b in zip(self.weights, self.biases):
            layer_inputs.insert(0, output_activations)
            tmp = np.dot(w, output_activations) + b
            output_activations = self.activation(tmp)
            activation_function_derivs.insert(0, self.activation_derivative(tmp))


        # Now we do backprop
        biases_gradients = []
        weights_gradients = []
        # The following variable collects the term in the chain rule expansion of dC/dw for the
        # layers before which w occurs in, chained with the derivative of the norm square.
        chain_pre_term = np.transpose(mean_square_derivative(output_activations, y))

        for w,b,i,da in zip(reversed(self.weights), 
                            reversed(self.biases), 
                            layer_inputs,
                            activation_function_derivs):
            #print("DEBUG: ", output_activations, y, chain_pre_term, self.activation_derivative)
            chain_pre_term *= np.transpose(da)                                  # Chain sparse
                                                                                # activation deriv
            biases_gradients.insert(0, np.transpose(chain_pre_term))            # d(Wx + b)/db = 1
            weights_gradients.insert(0, np.outer(chain_pre_term, i))            # Linalg checks out
            #print("DEBUG: ", w.shape, chain_pre_term.shape, da.shape)
            chain_pre_term = np.dot(chain_pre_term, w)                          # Chain d(Wx+b)/dx
                                                                                #       = W

        return biases_gradients, weights_gradients
    
    def test_accuracy(self, test_data):
        err = 0
        for (x,y),i in zip(test_data, range(1, len(test_data)+1)):
            print("Testing accuracy: ", i, "/", len(test_data), end='\r')
            out = self.feedforward(x)
            err += np.linalg.norm(out-y)**2
        print()
        return err / len(test_data)

