from .Loss import CategoricalCrossentropy
from .ActivationFunction import ActivationFunction
import numpy as np

class Classifier:

    class Softmax_CategoricalCrossentropy():
        def __init__(self):
            self.activation = ActivationFunction.Softmax()
            self.loss = CategoricalCrossentropy()
            self.losses = self.loss.losses

        def forward(self, inputs:[],y_true:[]):
            self.activation.forward(inputs)
            self.output = self.activation.output
            self.loss_value = self.loss.calculate(self.output, y_true)
            return self.loss_value
        
        def backward(self, dvalues:[]=[],y_true:[]=[]):
            samples = len(dvalues)
            if len(y_true.shape) == 2:
                y_true = np.argmax(y_true, axis=1)
            self.dinputs = dvalues.copy()
            self.dinputs[range(samples), y_true] -= 1
            self.dinputs = self.dinputs / samples
            return self.dinputs