import numpy as np

'''
List that contains activation function selectors for all available functions
'''
activations = ['r', 's', 't']

#%% ReLU
'''
Most widely used activation function for Neural Networks.
'''
def relu_fwd(Z):
    A = Z.copy()
    A[A<0] = 0
    return A

def relu_bwd(Z, A=None):
    dZ = Z.copy()
    dZ[dZ>=0] = 1
    dZ[dZ<0] = 0
    return dZ

#%% Sigmoid
'''
The sigmoid function is used for logistic regression and for the output layer for binary classification problems.
'''
def sigmoid_fwd(Z):
    A = 1 / (1 + np.exp(-Z))
    return A

def sigmoid_bwd(Z, A=None):
    if A is None:
        A = sigmoid_fwd(Z)
    dZ = A * (1 - A)
    return dZ

#%% Tanh
'''
Tanh activation works better than sigmoid in most cases.
'''
def tanh_fwd(Z):
    return np.tanh(Z)

def tanh_bwd(Z, A=None):
    if A is None:
        A = tanh_fwd(Z)
    dZ = 1 - np.power(A, 2)
    return dZ

#%% Leaky ReLU
'''
Leaky ReLU can be used as an alternative for ReLU when many negative values of the activations leads to lots of zero values.
'''
def leaky_relu_fwd(Z):
    A = Z.copy()
    np.maximum(Z, Z * 0.05, A)
    return A

def leaky_relu_bwd(Z, A=None):
    dZ = Z.copy()
    dZ[dZ>=0] = 1
    dZ[dZ<0] = 0.05
    return dZ

#%% Softmax
'''
Similar to sigmoid but for multiclass classification
'''
#def softmax_fwd(Z):
#    A = Z - np.max(Z)
#    exp = np.exp(A)
#    A = exp / np.sum(exp)
#    return A

#def softmax_bwd(Z, A=None):
    