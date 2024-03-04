import numpy as np
from .Layer import Layer

class Optimizer:

    class SGD:
        def __init__(self, learning_rate: float = 1.0, decay: float = 0.0, momentum: float = 0.0) -> None:
            # Initialize the optimizer with the given learning rate, decay, and momentum
            self.learning_rate = learning_rate
            self.current_learning_rate = learning_rate
            self.decay = decay
            self.iterations = 0
            self.momentum = momentum

        def pre_update_params(self) -> None:
            # If decay is set, update the current learning rate
            if self.decay:
                self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

        def update_params(self, layer:Layer) -> None:
            # Update the parameters of the layer using the current learning rate and momentum
            if self.momentum:
                # If momentum is set, update the weight and bias momentums
                if not hasattr(layer, 'weight_momentums'):
                    layer.weight_momentums = np.zeros_like(layer.weights)
                    layer.bias_momentums = np.zeros_like(layer.biases)
                weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
                layer.weight_momentums = weight_updates
                bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
                layer.bias_momentums = bias_updates
            else:
                # If momentum is not set, update the weights and biases directly
                weight_updates = -self.current_learning_rate * layer.dweights
                bias_updates = -self.current_learning_rate * layer.dbiases

            layer.weights += weight_updates
            layer.biases += bias_updates

        def post_update_params(self) -> None:
            # Increment the number of iterations after updating the parameters
            self.iterations += 1
