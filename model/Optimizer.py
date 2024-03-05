import numpy as np
from .Layer import Layer

class Optimizer:

    class SGD:
        def __init__(self, learning_rate: float = 1.0, decay: float = 0.0, momentum: float = 0.0) -> None:
            # Initialize the optimizer with the given learning rate, decay, and momentum
            # learning_rate: The step size used for each iteration of the parameter update.
            # decay: The variable used to decrease the learning rate over time, helping to stabilize the updates.
            # momentum: The variable that helps to accelerate SGD in the relevant direction and dampen oscillations.
            self.learning_rate = learning_rate
            self.current_learning_rate = learning_rate
            self.decay = decay
            self.iterations = 0  # Tracks the number of updates (iterations) performed
            self.momentum = momentum

        def pre_update_params(self) -> None:
            # If decay is set, adjust the learning rate based on the number of iterations.
            # This gradual reduction helps in converging to the minimum of the loss function.
            if self.decay:
                self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

        def update_params(self, layer: Layer) -> None:
            # Apply updates to the layer's weights and biases based on calculated gradients and optimization settings.
            if self.momentum:
                # Utilize momentum to update the parameters, which helps in smoothing out the updates
                # and can lead to faster convergence.
                if not hasattr(layer, 'weight_momentums'):
                    # Initialize momentum terms if they don't exist.
                    layer.weight_momentums = np.zeros_like(layer.weights)
                    layer.bias_momentums = np.zeros_like(layer.biases)
                
                # Calculate the momentum-adjusted updates for weights and biases.
                weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
                bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases

                # Apply momentum to the updates.
                layer.weight_momentums = weight_updates
                layer.bias_momentums = bias_updates
            else:
                # If no momentum, directly apply the learning rate to gradients for updates.
                weight_updates = -self.current_learning_rate * layer.dweights
                bias_updates = -self.current_learning_rate * layer.dbiases

            # Update layer weights and biases.
            layer.weights += weight_updates
            layer.biases += bias_updates

        def post_update_params(self) -> None:
            # Increment the iteration count after each parameter update.
            # This is used for adjusting the learning rate when decay is applied.
            self.iterations += 1
