import numpy as np

class ActivationFunction:
    
    class ReLU:

        def forward(self, inputs:[])->None:
            '''
            Forward psasing for ReLU activation function.
            If the input is less than 0, the output is 0, else the output is the input.
            '''
            self.inputs = inputs
            self.output = np.maximum(0, inputs)

        def backward(self, dvalues:[])->None:
            '''
            Backward passing for ReLU activation function.
            If the input is less than 0, the gradient is 0, else the gradient is 1.
            '''
            # making a copy of the original variable since we need to modify the values
            self.dinputs = dvalues.copy()
            # zero gradient where input values were negative
            self.dinputs[self.inputs <= 0] = 0

    class Sigmoid:
        def forward(self, inputs:[])->None:
            '''
            Forward passing for Sigmoid activation function.
            The sigmoid function is given by 1 / (1 + e^-x)
            '''
            self.output = 1 / (1 + np.exp(-inputs))

        def backward(self, dvalues:[])->None:
            '''
            Backward passing for Sigmoid activation function.
            The derivative of the sigmoid function is given by sigmoid(x) * (1 - sigmoid(x))
            '''
            self.dinputs = dvalues * (1 - self.output) * self.output


    class Softmax:
        def forward(self, inputs:[])->None:
            '''
            Forward passing for Softmax activation function.
            The softmax function is given by e^x / sum(e^x)
            '''
            exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
            probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
            self.output = probabilities

        def backward(self, dvalues:[])->None:
            '''
            Backward passing for Softmax activation function.
            The derivative of the softmax function is given by the Jacobian matrix.
            Jacobian matrix is just taking the derivative of the softmax function with respect to the input.
            '''
            self.dinputs = np.empty_like(dvalues)
            for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
                single_output = single_output.reshape(-1, 1)
                jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
                self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

