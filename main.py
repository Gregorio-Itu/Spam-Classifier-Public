from modules.classifiers import *
from modules.types.types import *
import argparse
import numpy as np
import matplotlib.pyplot as plt



MODELS = ["simulated_annealing", "gradient_descent"] # This is a list of the names of the models that can be selected


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
        training_spam = np.loadtxt(open(file_path), delimiter=",").astype(int).T # transposes the input so it can be multiplied with the weights
    except FileNotFoundError as e:
        print("Could not find spam data")
        raise e

    return training_spam[0, :], training_spam[1:, :] # This splits the labels and features into two separate arrays

if __name__ == "__main__":
    main()