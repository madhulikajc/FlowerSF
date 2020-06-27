import numpy as np
import h5py
import time
import os
from PIL import Image
from resizeimage import resizeimage
import matplotlib.pyplot as plt

RES = 100
NUM_PIX = RES * RES
RGB_LEN = NUM_PIX * 3
NUM_FLOWER_CLASSES = 4

# Eventually, test different #layers for improvement to final solution (bias and variance)
# Also change number of hidden nodes / size of each hidden layer (use layer dims for both)

def load_data():
# Later add test set also to this function
    """
    Loads all the image files along with labels into training set
    Classified and labeled based on directory name = flower name
    
    Returns:
    train_set_x -- training set x with R, G, B values in order (flattened)
    train_set_y -- training set y with labels 0, 1, 2, 3, 4 (0000, 1000, 0100, 0010, 0001)
        1 = agapanthus
        2 = ca_poppy
        3 = nasturtium
        4 = alyssum
        0 = not a flower, random picture
    """

    n_x = RGB_LEN
    n_y = NUM_FLOWER_CLASSES


    for flower_class, current_dir in enumerate(["../Data/square_images/01_agapanthus", "../Data/square_images/02_california_poppy", "../Data/square_images/03_nasturtium", "../Data/square_images/04_alyssum", "../Data/square_images/not_flower_random"]):
        m = len([filename for filename in os.listdir(current_dir) if filename.endswith("JPG") or filename.endswith("jpeg") or filename.endswith("jpg")])
        print("Number of training examples in", current_dir, "is", m)

        curr_training_set_x = np.zeros((n_x, m)) # Array training examples as columns
        curr_training_set_y = np.zeros((n_y, m)) # n_y is the number of output classes

        i=0  # Don't enumerate because some non JEPG files might be present, or test in the for statement for JPG
        for filename in os.listdir(current_dir):
            if filename.endswith("JPG") or filename.endswith("jpeg") or filename.endswith("jpg"):
                im = Image.open(current_dir + "/" + filename)
                small = resizeimage.resize_cover(im, [RES, RES])
                pixels = list(small.getdata())
                flatten = [rgb_val for p in pixels for rgb_val in p]  # Extract R G B values from the tuples into a long list
                curr_training_set_x[:, i] = flatten  # Store 3 * RES * RES values per example
                if flower_class < NUM_FLOWER_CLASSES: # Last class is not_flower_random pictures, should be left as 0, 0, 0, 0
                    curr_training_set_y[flower_class, i] = 1
                i=i+1

        if flower_class == 0:
            train_set_x = curr_training_set_x
            train_set_y = curr_training_set_y
        else:
            train_set_x = np.append(train_set_x, curr_training_set_x, axis = 1)
            train_set_y = np.append(train_set_y, curr_training_set_y, axis = 1)

    return train_set_x, train_set_y





def initialize_parameters(n_x, n_h, n_y):
    """
    Arguments:
    n_x: size of the input layer
    n_y: size of the output layer
    n_h: size of the hidden layers (Choose 2 layers for now, with same #hidden nodes - later make L layers, vary nodes)
    
    Returns:
    parameters: Python dictionary of parameters for the whole network -
                  W1: Weight matrix of shape (n_h, n_x)
                  b1: bias matrix of shape (n_h, 1)
                  W2: Weight matrix of shape (n_y, n_h)
                  b2: bias matric of shape (n_h, 1)
    """

    # Initialize bias vectors to zero, and Weight matrices to a standard normal distribution from -0.01 to +0.01
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    # Assert shapes of matrices later here

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def initialize_parameters_deep(layer_dims):
    """
    Argument:
    layer_dims: Python array (list) containing the dimensions (# hidden nodes) of each layer in the network

    Returns:
    parameters: Python dictionary containing the parameters W1, b1, W2, b2, and so on until WL and bL
                 Wl: Weight matrix of dimensions layer_dims[l], layer_dims[l-1]
                 bl: bias vector of shape layer_dims[l], 1

    """

    parameters = {}

    L = len(layer_dims)  # Number of layers in network including input and output layers
                         # Later L refers to the number of layers in the network not including the input layer
                         # In actuality below, we assign parameters to L-1 layers, as the first layer contains the input dim


    for l in range(1, L):
        # Weight and bias matrices are needed for L-1 layers, labeled W1, W2... Wl-1, not counting input layer
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def linear_forward(A, W, b):
    """
    Implement the linear step in forward prop (without activation)

    Arguments:
    A: activation from previous layer (or input data in the case of A0 = X), shape: (size of prev layer, #examples)
    W: Weight matrix, numpy array of shape (size of current layer, size of previous layer)
    b: bias vector, numpy array of shape (size of current layer, 1)

    Returns:
    Z: linear forward prop step without activation, forms the input to the activation 
    cache: Python tuple, containing A, W and b, in order to do back prop later on
    """

    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache



def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z: numpy array of any shape
    
    Returns:
    A: output of sigmoid(z), same shape as Z
    cache: returns Z as well, useful during backprop
    """
    
    A = 1/(1+np.exp(-Z))
    cache = Z    
    return A, cache


def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z: Output of the linear layer, of any shape

    Returns:
    A: Post-activation parameter, of the same shape as Z
    cache: a python dictionary containing "A" ; stored for back prop later
    """
    
    A = np.maximum(0,Z)
    cache = Z 
    return A, cache


    

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the linear forward prop as well as the Activation, ie LINEAR --> ACTIVATION both

    Arguments:
    A_prev: activations from previous layer (or input data): (size of previous layer, number of examples)
    W: Weight matrix, numpy array of shape (size of current layer, size of previous layer)
    b: bias vector, numpy array of shape (size of the current layer, 1)
    activation: the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the linear Weights plus bias step, plus activation function
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing back prop later
    """
    
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)   # linear cache contains A_prev, W and b
        A, activation_cache = sigmoid(Z)   # the Cache contains Z
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)  # linear_cache contains A_prev, W and b
        A, activation_cache = relu(Z)  # the Cache contains Z

    
    cache = (linear_cache, activation_cache)
    return A, cache



def forward_prop(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1) for the first L-1 layers, then->LINEAR->SIGMOID for last layer
    Assumes 4 output classes, but with 5 possible outcomes: 0000, 1000, 0100, 0010, 0001
    This can be changed to add more output classes (more flowers recognized)
    
    Arguments:
    X: numpy array of shape (input features, number of examples)
    parameters: output of initialize_parameters_deep()
    
    Returns:
    AL: final post-activation value of final output layer
    caches: list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 1 to L-1, eg W1, W2)
                each individual cache is a tuple containing the linear cache (A, W, b) and activation cache (Z)
    """

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W'+str(l)], parameters['b' + str(l)], "relu")
        caches = caches + [cache]  # for back prop later

    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.

    AL, cache = linear_activation_forward(A, parameters['W'+str(L)], parameters['b' + str(L)], "sigmoid")
    caches = caches + [cache]

    
    assert(AL.shape == (NUM_FLOWER_CLASSES, X.shape[1]))
        
    return AL, caches


def compute_cost(AL, Y):
    """
    Implement Cost function as a cross entropy (logistic regression style) loss over NUM_FLOWER_CLASSES

    Arguments:
    AL: Vector of probabilities outputs for each class, corresponding to predictions for that label,
            shape(NUM_FLOWER_CLASSES, # examples)
    Y: True label vector, same shape as AL, contains labels like 0100 for 2nd flower class

    Returns:
    cost: Cross entropy cost (can be changed if we want different metrics for measuring loss)
    """

    m = Y.shape[1]

    # Compute individual costs for each flower class in a binary cross entropy / logistic regression way and sum
    cost = 0
    for i in range(NUM_FLOWER_CLASSES):
        cost += (-1/m) * (np.dot(Y[i, :], np.log(AL[i, :]).T) + np.dot(1-Y[i, :], np.log(1-AL[i, :]).T))

    cost = np.squeeze(cost)
    assert(cost.shape == ())
    return cost


## BEGIN BACK PROP FUNCTIONS


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA: post-activation gradient, of any shape
    cache: 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ: Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) 

    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)    
    return dZ


def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA: post-activation gradient, of any shape
    cache: 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ: Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)   #formula for derivative of sigmoid function
    
    assert (dZ.shape == Z.shape)
    return dZ


def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ: Gradient of the cost with respect to the linear output (of current layer l)
    cache: tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev: Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW: Gradient of the cost with respect to W (current layer l), same shape as W
    db: Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)

    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db



def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR and ACTIVATION steps for a given layer.
    
    Arguments:
    dA: post-activation gradient for current layer l 
    cache: tuple of values (linear_cache, activation_cache) stored during fwd prop for computing back prop efficiently
    activation: the activation that was used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev: Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW: Gradient of the cost with respect to W (current layer l), same shape as W
    db: Gradient of the cost with respect to b (current layer l), same shape as b
    """

    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db



def back_prop(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID whole L layer network
    
    Arguments:
    AL: probability vector, output of the forward propagation
    Y: true "label" vector (containing 0 if a given flower class is not present and 1 if it is present in the image)
    caches: list of caches containing:
                every cache of linear_activation_forward() with "relu" (caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (caches[L-1])
    
    Returns:
    grads: A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """

    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))  # Derivative of cost in compute_cost wrt AL

    
    # FINAL Lth layer (SIGMOID -> LINEAR) back prop gradients. 
    # Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")
    
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) back prop gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". 
        # Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        current_cache = caches[l]
        # In the first round grads[dAL-1]has been set above already (L-2+1)
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads



def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l+1)] -= learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l+1)] -= learning_rate * grads["db" + str(l + 1)]
    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X: data, numpy array of shape (RES * RES * 3, number of examples)
    Y: true "label" one-hot vector with number of flower classes (0000, 1000, 0100, etc) (NUM_FLOWER_CLASSES, #examples)

    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    costs = []                         # keep track of cost
    
    parameters = initialize_parameters_deep(layers_dims)
    
    # Gradient Descent
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = forward_prop(X, parameters)
        
        cost = compute_cost(AL, Y)

        # Back prop
        grads = back_prop(AL, Y, caches)
 
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    print(costs)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters





## BEGIN MAIN


tsx, tsy = load_data()
print(tsx.shape)
print(tsy.shape)
print(tsx)
print(tsy)

tsx = tsx/255

layers_dims = [RES*RES*3, 20, 7, 5, 4] #  4-layer model

L_layer_model(tsx, tsy, layers_dims, num_iterations = 3000, print_cost=True)

layers_dims = [RES*RES*3, 10, 4] #  2-layer model

L_layer_model(tsx, tsy, layers_dims, num_iterations = 3000, print_cost=True)


# notes for next steps:
# gradient checking
# simpler networks performance seems better, explore
# bias and variance - check with differing numbers of training data
# save the parameters for later, 
# save .npy files for later for training and testing
# Nan issues
# 0/0 is not 1, so what to do about the sigmoid intial cost for backprop, and derivative calc

