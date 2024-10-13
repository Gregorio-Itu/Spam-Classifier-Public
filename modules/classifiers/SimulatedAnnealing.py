from .Base import SpamClassifierBase
from ..plots.Graph import LossGraph
from ..types.types import *
import numpy as np

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
    