import numpy as np

class Loss:
    def __init__(self):
        # Initializes a list to store the loss values calculated during training.
        self.losses = []

    def calculate(self, output, target) -> float:
        '''
        Calculate the loss by calling the forward method which must be implemented
        by subclasses. It returns the average loss across all samples.
        '''
        # Call the forward method (to be defined in subclasses) to calculate individual losses
        sample_losses = self.forward(output, target)
        # Calculate the mean of these losses to get the average loss for the batch
        data_loss = np.mean(sample_losses)
        return data_loss
    
class CategoricalCrossentropy(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true)->[]:
        '''
        Compute the Categorical Crossentropy loss in the forward pass.
        This function handles both cases where y_true is given as one-hot encoded vectors
        or as discrete class labels.

        The loss for each sample is -log(correct class's predicted probability).
        '''
        samples = len(y_pred)  # Number of samples in the batch
        # Clip the predicted probabilities to avoid log(0) which is undefined
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        # Handling the case of discrete labels for y_true
        if len(y_true.shape) == 1:
            # Select the predicted probabilities of the correct class for each sample
            correct_confidences = y_pred_clipped[range(samples), y_true]
        
        # Handling the case of one-hot encoded labels for y_true
        elif len(y_true.shape) == 2:
            # Multiply each predicted probability by the corresponding one-hot encoded truth value
            # and sum them to get the predicted probability of the correct class
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        
        # Calculate the negative log likelihood for each sample's correct class prediction
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    def backward(self, dvalues, y_true)->None:
        '''
        Compute the gradient of the loss function with respect to its inputs for the backward pass.
        The gradient (dinputs) will be used to update the weights during backpropagation.

        The derivative of the loss with respect to each predicted probability is
        -1 / correct_confidence for the correct class and 0 for others, scaled by the number of samples
        to normalize the gradient across the batch.
        '''
        samples = len(dvalues)  # Number of samples
        labels = len(dvalues[0])  # Number of labels/classes
        
        # Convert discrete labels to one-hot encoded format if necessary
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        # Calculate the gradient of the loss function
        self.dinputs = -y_true / dvalues
        # Normalize the gradients by the number of samples
        self.dinputs = self.dinputs / samples
