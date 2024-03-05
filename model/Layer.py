import numpy as np

class Layer:
    def __init__(self, n_inputs: int, n_outputs: int):
        '''
        Initialize the layer with random weights and zero biases.
        
        n_inputs: Number of input features to the layer.
        n_outputs: Number of neurons (units) in the layer.
        '''
        # Initialize weights with small random values
        # This helps to break symmetry and ensures that all neurons initially produce different outputs.
        self.weights = 0.01 * np.random.randn(n_inputs, n_outputs)
        # Initialize biases with zeros
        # Starting with zero biases is a common practice.
        self.biases = np.zeros((1, n_outputs))

    def forward(self, inputs:[]):
        '''
        Perform the forward pass through the layer.
        
        inputs: The input data or the output from the previous layer.
        '''
        # Store inputs for use in the backward pass
        self.inputs = inputs
        # Compute the output of the layer
        # The operation is a dot product of inputs and weights, plus the bias for each neuron.
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues:[]):
        '''
        Perform the backward pass through the layer.
        
        dvalues: The gradient of the loss function with respect to the output of this layer.
        '''
        # Gradient of the loss with respect to weights
        # It is computed as a dot product of the inputs transposed and the gradients with respect to the layer's output.
        self.dweights = np.dot(self.inputs.T, dvalues)
        # Gradient of the loss with respect to biases
        # It is the sum of the gradients, as each bias influences its neuron's output directly by the same amount.
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient of the loss with respect to the layer's inputs
        # This is needed for backpropagation through the previous layer.
        self.dinputs = np.dot(dvalues, self.weights.T)