from .Base import SpamClassifierBase
from ..plots.Graph import LossGraph
from ..types.types import *
import numpy as np

class SpamClassifierGradientDescent(SpamClassifierBase):
    """
    Spam classifying model which trains its weights using gradient descent
    """

    def __init__(self):
        super().__init__()
        self.config_directory = "./configs/gradient_descent.json"

    def forward_prop_store_outputs(self, input_array: NDArrayInt) -> tuple[list[NDArrayFloat], list[NDArrayFloat]]:
        """
        Propagates an input sample through all the layers of the neural network
        Saves the outputs of the hidden layers and activation layers
        """

        try:

            hidden_outputs = [self.weights[0].dot(input_array) + self.biases[0]] 
            activation_outputs = []

            for i in range(self.number_of_hidden_layers): # Forward propagation algorithm repeats and stores all the hidden output and activation output layers
                activation_outputs.append(self.relu(hidden_outputs[i]))
                hidden_outputs.append(self.weights[i+1].dot(activation_outputs[i]) + self.biases[i+1])
            
            activation_outputs.append(self.sigmoid(hidden_outputs[-1])) # Final activation output uses a sigmoid function
            
            return hidden_outputs, activation_outputs

        except ValueError as e:
            print("The size of the input layer did not match that of the data")
            raise e

    def train(self, training_labels: NDArrayInt, training_data: NDArrayInt, iterations: int, visualise: bool):
        """
        Method which for training the model via gradient descent
        Iterations are dependant on user input
        Learning rate is changed in the config file
        """

        m = training_data.shape[1] # The number of training examples

        configs = self.load_config()

        learning_rate = configs["learning_rate"] # Sets the learning rate from the config file

        if visualise: # Create a graph for live plotting if requested
            graph = LossGraph("Gradient Descent")

        for i in range(iterations):

            hidden_outputs, activation_outputs = self.forward_prop_store_outputs(training_data) # Gtes the layer outputs

            dZ = activation_outputs[-1] - training_labels # The derivative of the first hidden layer

            for j in range(-1, -self.number_of_hidden_layers-1, -1): # Step here is negative as the algorithm starts from the last weight and propagates backwards

                dA = np.dot(self.weights[j].T, dZ)
                dW = np.dot(dZ, activation_outputs[j-1].T) / m
                db = np.sum(dZ, axis=1, keepdims=True) / m

                self.weights[j] -= learning_rate * dW # Weights and biases are updated
                self.biases[j] -= learning_rate * db

                dZ = dA * self.derivative_relu(hidden_outputs[j-1])

            dW = np.dot(dZ, training_data.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            self.weights[0] -= learning_rate * dW
            self.biases[0] -= learning_rate * db

            loss = self.log_loss(self.forward_prop(training_data), training_labels) # Get the current loss, just for logging purposes

            print(f"iteration:{i}, loss:{loss:.10f}")

            if visualise: # Update the graph if it exists
                graph.update(i, loss)

    
    def derivative_relu(self, x: NDArrayFloat) -> NDArrayInt:
        """
        The derivative of the ReLU function
        Used to calculate the gradient between activation and hidden layers
        """

        return (x > 0).astype(int)