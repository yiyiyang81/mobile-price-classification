import numpy as np

class Loss:
    def __init__(self):
        self.losses = []

    def calculate(self, output, target):
        '''
        Calculate the loss using the forward method.
        Return the average loss.
        '''
        sample_losses = self.forward(output, target)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class CategoricalCrossentropy(Loss):
        def __init__(self):
            super().__init__()

        
        def forward(self, y_pred, y_true):
            '''
            Forward passing for Categorical Crossentropy loss function.            
            The loss is given by -log(correct_confidence)
            '''
            samples = len(y_pred)
            # clip data to prevent division by 0
            y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
            # if targets are one-hot encoded, turn them into discrete values
            if len(y_true.shape) == 1:
                # calculate the correct confidences for non-one-hot encoded targets
                correct_confidences = y_pred_clipped[range(samples), y_true]
            # mask values - only for one-hot encoded labels
            elif len(y_true.shape) == 2:
                # calculate the correct confidences for one-hot encoded targets
                correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
            negative_log_likelihoods = -np.log(correct_confidences)
            return negative_log_likelihoods
        
        def backward(self, dvalues, y_true):
            '''
            Backward passing for Categorical Crossentropy loss function.
            The derivative of the loss is given by -1 / correct_confidence
            '''
            samples = len(dvalues)
            labels = len(dvalues[0])
            if len(y_true.shape) == 1:
                y_true = np.eye(labels)[y_true]
            self.dinputs = -y_true / dvalues
            self.dinputs = self.dinputs / samples
