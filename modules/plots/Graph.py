import matplotlib.pyplot as plt

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