"""

Digit recognition using Network from network.py

The code here is taken from the notebook and modified so that it may be run as a script from the
command line with arguments for the number of epochs and regularization parameter. The modifications
include removal of certain comments and the code testing the neural network manually on the 99th
testing data. Instead, a final accuracy is calculated and recorded. The data generated for plotting
is also saved at the end of the training process, the file name is obtained as command line input.
The data is saved as the string value of (test_hist, train_hist) as python lists, and can be read
and interpreted via `eval()`.

Arg 1:  Name of output file to write plotting data to.
Arg 2:  Number of epochs
Arg 3:  Regularization parameter

All three arguements must be present when launching from the command line.

We import the data from mnist.
"""

from tensorflow.keras.datasets import mnist
from network import *
import sys



# HELPER FUNCTIONS
# following function will be used to transfor the training lables for digits recognition
def vectorized_result(j): 
    # Returns 10-size column with 1 at jth position and rest are 0
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e



(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train_vectorized = list(map(vectorized_result, y_train))   # This will convert digit 0 to [1.0,0,0,0..0], 1 to [0,1.0,0,0...0] and so on...
x_train_vectorized = list(map(lambda x: np.reshape(x, (784,1))/255, x_train)) 

training_data = list(zip(x_train_vectorized, y_train_vectorized))

x_test_vectorized = list(map(lambda x: np.reshape(x, (784,1))/255, x_test)) 

vectorized_test_data = list(zip(x_test_vectorized, map(vectorized_result, y_test)))

net = Network([784, 16, 10], regularization = float(sys.argv[3])  # activation_func = relu, activation_derivative = relu_derivative)
n_epochs = sys.argv[2]
net.SGD(training_data, n_epochs, 10, 0.1, test_data=vectorized_test_data)   


# We print out final accuracy
accuracy = 0
for x,y in zip(x_test_vectorized, y_test):
    accuracy += 1 if y == np.argmax(net.feedforward(x)) else 0
accuracy *= 100 / len(y_test)
print("Percent accuracy", accuracy)


# Save the plotting data
with open(sys.argv[1], 'w') as f:
   f.write(str(net.test_cost_history, net.train_cost_history))


# Plot graph
plt.plot(range(n_epochs), net.train_cost_history, 'g',  label = "Training")
plt.plot(range(n_epochs), net.test_cost_history, 'r',  label = "Testing")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Mean Square Cost")
plt.title("DigiNet, lambda = " + sys.argv[3] + ", epochs = " + sys.argv[2])
plt.grid()
plt.show()


