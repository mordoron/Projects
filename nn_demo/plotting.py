# Plotting. curtisy of: https://github.com/kapil-varshney/utilities/tree/master/training_plot
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from tensorflow import keras

class TrainingPlotFunc(keras.callbacks.Callback):

    def __init__(self, V, R, F, mean, std, title):
        self.V = V
        self.I = np.arange(0,R,0.01)
        self.mean = mean
        self.std = std
        self.F = F

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        clear_output(wait=True)
        plt.ylim([np.min(self.F(self.I))-0.5,np.max(self.F(self.I))+0.5])
        plt.plot(self.V*self.std+self.mean, self.model.predict(self.V), c='r', label='Prediction')
        plt.plot(self.I, self.F(self.I), c='b', label='sin')
        plt.legend(loc='lower left')
        plt.show()

class TrainingPlot(keras.callbacks.Callback):

    def __init__(self, title):
        self.title = title

    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))

        # Before plotting ensure at least 1 epochs have passed
        if len(self.losses) >= 1:
            # Clear the previous plot
            clear_output(wait=True)
            N = np.arange(0, len(self.losses))

            # You can chose the style of your preference
            # print(plt.style.available) to see the available options
            plt.style.use("seaborn")

            # Plot train loss, train acc, val loss and val acc against epochs passed
            fig, ax1 = plt.subplots()
            ax1.plot(N, self.losses, label="train_loss", color='r')
            ax1.plot(N, self.val_losses, label="val_loss", color='g')
            ax2 = ax1.twinx()
            ax2.grid(False)
            ax2.plot(N, self.acc, label="train_acc", color='b')
            ax2.plot(N, self.val_acc, label="val_acc", color='c')
            plt.title("Training Loss and Accuracy {}".format(self.title))
            plt.xlabel("Epoch #")
            ax1.set_ylabel("Loss")
            ax2.set_ylabel("Accuracy")
            fig.legend(loc="upper right", bbox_to_anchor=(0.5, 1), bbox_transform=ax1.transAxes)
            plt.show()