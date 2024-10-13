from ..types.types import *
import json
import numpy as np
import matplotlib.pyplot as plt

class SpamClassifierBase:
    """
    Base class for all spam classifiers
    """

    def __init__(self):

        self.number_of_hidden_layers = 0 # The number of hidden layers the model contains
        self.weights = []                # Contains all the weights for each layer
        self.biases = []                 # Contains all the biases for each layer
        self.config_directory = ""       # The directory of the config

    def save_weights(self, file_path: str):
        """
        Saves the model's weights and biases as an npz to the given directory
        """

        try:
            np.savez_compressed(file_path, *(self.weights + self.biases)) # The weights are stored first in the file followed by the biases
        except IOError as e:
            raise e
            
    def load_weights(self, file_path: str):
        """
        Loads the model's weights and biases from a given directory
        """

        try:
            data = np.load(file_path)
        except FileNotFoundError as e:
            print("Could not find weights")
            raise e

        self.number_of_hidden_layers = (len(data) // 2) - 1 # The file contains 

        self.weights = [data[key] for i, key in enumerate(data) if i <= self.number_of_hidden_layers] # The first half of the arrays in the file are the weights
        self.biases = [data[key] for i, key in enumerate(data) if i > self.number_of_hidden_layers]   # The latter half are the biases

    def initialise_layers(self, input_layer_size: int, layer_sizes: list[int]):
        """
        Creates the layers for the model
        Initially, weights and biases are chosen randomly and modified with kaiming initialisation
        """

        self.number_of_hidden_layers = len(layer_sizes)

        self.weights = [np.random.randn(layer_size, next_input_size) * self.kaiming(next_input_size) for layer_size, next_input_size in zip(layer_sizes + [1], [input_layer_size] + layer_sizes)] # A final layer of 1 neuron is added as the output must be binary
        self.biases = [np.random.randn(layer_size, 1) * self.kaiming(next_input_size) for layer_size, next_input_size in zip(layer_sizes + [1], [input_layer_size] + layer_sizes)]

    def kaiming(self, input_units: int) -> float:
        """
        Initialisation method for improving convergence rates of models with ReLU activation functions
        Modifies the random starting weights slightly
        Also known as He initialisation
        """

        return np.sqrt(2 / input_units)
        
    def predict(self, data: NDArrayInt) -> NDArrayInt:
        """
        Classifies all given samples in a numpy array
        """

        return np.round(self.forward_prop(data)).astype(int) # The outputs are rounded up or down to represent one of two classes
                                                                                
        
    def forward_prop(self, input_array: NDArrayInt) -> NDArrayFloat:
        """
        Propagates an input sample through all the layers of the neural network
        """

        try:

            for i in range(self.number_of_hidden_layers):
                input_array = self.relu(np.dot(self.weights[i], input_array) + self.biases[i])
            
            return self.sigmoid(np.dot(self.weights[-1], input_array) + self.biases[-1]) # A sigmoid activation function is applied at the end to keep the output between 0 and 1
        
        except ValueError as e:
            print("The size of the input layer did not match that of the data")
            raise e
    

    def relu(self, x: NDArrayFloat) -> NDArrayFloat:
        """
        Activation function that sets negative values to 0
        """

        return np.maximum(0, x)

    def sigmoid(self, x: NDArrayFloat) -> NDArrayFloat:
        """
        Activation function which outputs in a range between 0 and 1
        """
        try:
            return 1 / (1 + np.exp(-x))
        except FloatingPointError as e:
            print(x)
            raise e


    def log_loss(self, output_array: NDArrayFloat, label_array: NDArrayInt) -> NDArrayFloat:
        """
        Calculates loss value using logarithms
        """

        output_array = np.clip(output_array, 1e-7, 1-1e-7) # The array is clipped slightly above 0 and below 1 to prevent division by zero
        return -(1/label_array.shape[0]) * np.sum(label_array * np.log(output_array) + (1 - label_array) * np.log(1 - output_array))

    def copy_weights(self, weights: list[NDArrayFloat]) -> list[NDArrayFloat]:
        """
        Creates value copies of given weights or biases
        """

        return [weight.copy() for weight in weights]

    def set_config_directory(self, directory: str):
        """
        Method for changing defualt config path
        """
        self.config_directory = directory

    def load_config(self):
        """
        Function which loads the configs for the model
        """
        try:
            with open(self.config_directory, 'r') as f:
                return json.load(f)

        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise e

    def train(self, training_labels: NDArrayInt, training_data: NDArrayInt, iterations: int, visualise: bool):
        return None