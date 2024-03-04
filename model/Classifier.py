from .Loss import CategoricalCrossentropy
from .ActivationFunction import ActivationFunction
import numpy as np

class Classifier:

    class Softmax_CategoricalCrossentropy():
        def __init__(self):
            # Initialize the classifier with a softmax activation function and categorical crossentropy loss
            self.activation = ActivationFunction.Softmax()
            self.loss = CategoricalCrossentropy()
            self.losses = self.loss.losses

        def forward(self, inputs:[],y_true:[]):
            # Perform a forward pass through the network
            self.activation.forward(inputs)
            self.output = self.activation.output
            self.loss_value = self.loss.calculate(self.output, y_true)
            return self.loss_value
        
        def backward(self, dvalues:[]=[],y_true:[]=[]):
            # Perform a backward pass through the network
            samples = len(dvalues)
            # If the targets are one-hot encoded, convert them to discrete values
            if len(y_true.shape) == 2:
                y_true = np.argmax(y_true, axis=1)
            self.dinputs = dvalues.copy()
            self.dinputs[range(samples), y_true] -= 1
            self.dinputs = self.dinputs / samples
            return self.dinputs