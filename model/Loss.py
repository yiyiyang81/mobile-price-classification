import numpy as np

class Loss:
    def __init__(self):
        self.losses = []

    def calculate(self, output, target):
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
            # if targets are one-hot encoded, 
            if len(y_true.shape) == 1:
                correct_confidences = y_pred_clipped[range(samples), y_true]
            # mask values - only for one-hot encoded labels
            elif len(y_true.shape) == 2:
                # mask the values                
                correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
            negative_log_likelihoods = -np.log(correct_confidences)
            return negative_log_likelihoods
        
        def backward(self, dvalues, y_true):
            samples = len(dvalues)
            labels = len(dvalues[0])
            if len(y_true.shape) == 1:
                y_true = np.eye(labels)[y_true]
            self.dinputs = -y_true / dvalues
            self.dinputs = self.dinputs / samples
