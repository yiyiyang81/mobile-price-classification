import numpy as np

class Layer:
    def __init__(self, n_inputs: int, n_outputs: int):
        '''
        weights is initialized with a random number.
        biases is initialized with 0.
        '''
        self.weights = 0.01 * np.random.randn(n_inputs, n_outputs)
        self.biases = np.zeros((1, n_outputs))


    def forward(self, inputs:[]):
        '''
        Forward passing for the layer.
        '''
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues:[]):
        '''
        Backward passing for the layer.
        '''
        # calculating gradient of the current layer
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
