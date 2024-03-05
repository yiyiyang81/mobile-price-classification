from .Loss import CategoricalCrossentropy
from .ActivationFunction import ActivationFunction
import numpy as np

class Classifier:

    class Softmax_CategoricalCrossentropy():
        def __init__(self):
            # This class combines softmax activation with categorical crossentropy loss.
            # Softmax is used for the forward pass to convert logits to probabilities.
            # CategoricalCrossentropy evaluates how well these probabilities match the true labels.
            self.activation = ActivationFunction.Softmax()
            self.loss = CategoricalCrossentropy()
            self.losses = self.loss.losses  # Stores loss values for monitoring.

        def forward(self, inputs:[], y_true:[]):
            '''
            Performs the forward pass using softmax and calculates the loss.
            
            inputs: Logits from the final layer of the network.
            y_true: Actual labels for the input data.
            '''
            # Apply softmax to input logits.
            self.activation.forward(inputs)
            self.output = self.activation.output  # The probabilities after softmax.
            # Calculate and return the loss value.
            self.loss_value = self.loss.calculate(self.output, y_true)
            return self.loss_value
        
        def backward(self, dvalues:[]=[], y_true:[]=[]):
            '''
            Performs the backward pass, calculating the gradient of the loss with respect to the input logits.
            
            dvalues: The gradient of the loss with respect to the output of the softmax layer.
            y_true: Actual labels for the input data.
            '''
            samples = len(dvalues)
            # Convert one-hot encoded labels to discrete values if necessary.
            if len(y_true.shape) == 2:
                y_true = np.argmax(y_true, axis=1)
            # Copy the softmax output gradients.
            self.dinputs = dvalues.copy()
            # Calculate the gradient of the loss with respect to the inputs of the softmax layer.
            self.dinputs[range(samples), y_true] -= 1
            self.dinputs = self.dinputs / samples  # Normalize by the number of samples for averaging.
            return self.dinputs
