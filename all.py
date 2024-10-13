import json
import argparse
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


MODELS = ["simulated_annealing", "gradient_descent"] # This is a list of the names of the models that can be selected
NDArrayInt = npt.NDArray[np.int_]                    # These are defined types for np arrays of integers and floats, useful for type hinting
NDArrayFloat = npt.NDArray[np.float_]



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
    


class SpamClassifierSimulatedAnnealing(SpamClassifierBase):
    """
    Spam classifying model which trains its weights using simulated annealing
    """

    def __init__(self):
        super().__init__()
        self.config_directory = "./configs/simulated_annealing.json"

    def train(self, training_labels: NDArrayInt, training_data: NDArrayInt, iterations: int, visualise: bool):
        """
        Method which trains the weights and biases of the model using simulated annealing
        Iterations are dependant on user input
        Temperature and shift ratio depend on config file
        """

        lowest_loss = self.log_loss(self.forward_prop(training_data), training_labels) # This tracks the lowest loss achieved
        improvements = 0                                                               # Increments whenever the current best weights and biases are improved on
        acceptances = 0                                                                # Increments whenever a set of weights and biases are accepted and used, even if they are not better

        best_weights = self.copy_weights(self.weights)
        best_biases = self.copy_weights(self.biases)

        configs = self.load_config()

        learning_rate = configs["learning_rate"]
        temperature = configs["temperature"]

        cooling_rate = np.power(1000,-1/iterations) # Cooling rate is an expression of the number of iterations to prevent temperature reaching near 0 in longer runs

        if visualise: # Create a graph for live plotting if requested
            graph = LossGraph("Simulated Annealing")

        for i in range(iterations):

            self.weights = [weight + np.random.randn(weight.shape[0], weight.shape[1]) * learning_rate for weight in self.weights] # Weights and biases are modified slightly
            self.biases = [bias + np.random.randn(bias.shape[0], bias.shape[1]) * learning_rate for bias in self.biases]

            loss = self.log_loss(self.forward_prop(training_data), training_labels)

            if loss < lowest_loss: # If the loss is lower than the lowest loss, the model has improved and the new eights are accepted
                best_weights = self.copy_weights(self.weights)
                best_biases = self.copy_weights(self.biases)
                lowest_loss = loss
                improvements += 1

            elif np.random.rand() > self.acceptance_probability(loss - lowest_loss, temperature): # The weights will then  be rejected if they fail a probability check
                self.weights = self.copy_weights(best_weights)
                self.biases = self.copy_weights(best_biases)

            else: # Otherwise, the weights are accepted, even though they are not the best
                acceptances += 1

            temperature *= cooling_rate # Temperature is reduced
                            
            print(f"iteration:{i}, improvements:{improvements}, acceptances:{acceptances}, loss:{loss:.10f}, lowest loss:{lowest_loss:.10f}, temperature:{temperature:.10f}")

            if visualise: # Update the graph if it exists
                graph.update(i, loss)

    def acceptance_probability(self, delta_loss: float, temperature: float) -> float:
        """
        This function determines the probability a non-optimal weight set is accepted based on the difference in loss between it and the best weights
        """

        return np.exp(-delta_loss / temperature)
    


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
    

    
class LossGraph():
    """
    Class used to plot the data from the classifers' training process in real time
    Plots loss against iteration number
    """

    def __init__(self, title: str):

        plt.ion() # Must be in interactive mode to plot in real time

        self.fig, self.ax = plt.subplots()

        self.xs = [] # These hold all of the x and y values respectively for plotting onto the graph
        self.ys = []

        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 1)

        plt.title(title)
        plt.ylabel("Loss")
        plt.xlabel("Iteration")
        
        self.line, = self.ax.plot(self.xs, self.ys, color='r')

    def update(self, iteration: int, loss: float):
        """
        Plots a new point onto the graph
        """

        self.xs.append(iteration)
        self.ys.append(loss)

        self.line.set_data(self.xs, self.ys) # Updates the line

        if iteration > 100: # If the number of iterations exceeds 100, the x axis extends to accomodate
            self.ax.set_xlim(0, iteration)

        self.fig.canvas.draw() # Draws the changes to the screen
        self.fig.canvas.flush_events()

def main():
    """
    Main function of the program
    Either begins training a model or tests a model based on user input
    """

    args = get_arguments()

    classifier = get_model(args.model)

    if args.config_directory is not None:
        classifier.set_config_directory(args.config_directory)

    match args.action:

        case "train":

            input_layer_size = args.layers[0]     # The first number in the layers argument list is used to set the number of neurons for the input layer
            hidden_layers_sizes = args.layers[1:] # Subsequent numbers set the number of neurons for the hidden layers, as well as the number of layers

            training_labels, training_data = load_spam(args.training_data_directory) # Unpacks the training spam into the labels and features

            classifier.initialise_layers(input_layer_size, hidden_layers_sizes) # The layers of the model are created
            classifier.train(training_labels, training_data, args.iterations, args.graph)   # The model begins training on the training data

            if not args.dont_save_weights: # Weights are saved by default
                classifier.save_weights(args.weights_save_directory)
                print(f"Weights saved to {args.weights_save_directory}")

            if args.graph:
                plt.ioff()
                plt.show()

        case "test":

            classifier.load_weights(args.weights_load_directory)
            
            test_labels, test_data = load_spam(args.test_data_directory) # Unpacks the testing spam

            predictions = classifier.predict(test_data)                                  # Begins forward propagation on the testing data
            accuracy = np.count_nonzero(predictions == test_labels)/test_labels.shape[0] # The generated 1 dimension array of answers is compared wwith the labels. Matching values are counted and then divided by the number of samples to get an accuracy 
            
            print(f"Tests completed with an accuracy of {accuracy*100}%.")

def get_arguments() -> argparse.Namespace:
    """
    Function used for reading and interpreting any given command line arguments
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--action", type=str, choices=["train", "test"], required=True, help="Choose to either train a model or test one")
    parser.add_argument("-m", "--model", type=str, choices=MODELS, required=True, help="Choose a model to use")
    parser.add_argument("-l", "--layers", type=int,  help=f"The numbers of neurons in the input and subsequent hidden layers", nargs='+', default=[54, 54])
    parser.add_argument("-i", "--iterations", type=int, help="Number of iterations to use for training", default=10000)
    parser.add_argument("-g", "--graph", help="Plots variables related to the training process in real time", action="store_true")
    parser.add_argument("-cdr", "--config_directory", type=str, help="Directory of the config file for the model", default=None)
    parser.add_argument("-dsw", "--dont_save_weights", help="Don't save the weights when training is complete", action="store_true")
    parser.add_argument("-trndr", "--training_data_directory", type=str, help="Directory of the data to train on", default="./data/training_spam.csv")
    parser.add_argument("-tstdr", "--test_data_directory", type=str, help="Directory of the data to test on", default="./data/testing_spam.csv")
    parser.add_argument('-wsdr', "--weights_save_directory", type=str, help="Directory to save the weights to", default="./data/weights.npz")
    parser.add_argument('-wldr', "--weights_load_directory", type=str, help="Directory to load the weights from", default="./data/weights.npz")

    return parser.parse_args()

def get_model(model: str) -> SpamClassifierBase:
    """
    Takes the user's input and returns the classifying model they requested
    """

    match model:
        case "simulated_annealing":
            return SpamClassifierSimulatedAnnealing()
        case "gradient_descent":
            return SpamClassifierGradientDescent()

def load_spam(file_path: str) -> tuple[NDArrayInt, NDArrayInt]:
    """
    Reads files that contain spam data for testing or training
    """  

    try:
        training_spam = np.loadtxt(open(file_path), delimiter=",").astype(int).T # Transposes the input so it can be multiplied with the weights
    except FileNotFoundError as e:
        print("Could not find spam data")
        raise e

    return training_spam[0, :], training_spam[1:, :] # This splits the labels and features into two separate arrays


if __name__ == "__main__":
    main()