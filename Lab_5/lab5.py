# #input layer hidden layer and output layer
# #Implement a 2-layer (input layer,
# # hidden layer and output layer) neural network from scratch for the XOR operation. This includes implementing forward and backward passes from scratch.
# # The truth table for XOR is given by
# import numpy as np
# #
# #
# X = [[0, 0],
#      [0, 1],
#      [1, 0],
#      [1, 1]]
# Y = [[0],
#      [1],
#      [1],
#      [0]]
# alpha = float(input("Enter alpha: "))
#
# def relu(x):
#     return np.maximum(0, x)
# def sigmoid(x):
#    return 1 / (1 + np.exp(-x))
# def tanh(x):
#    return np.tanh(x)
#
#
# def xavier_initialization(n_in, n_out):
#    std = np.sqrt(2 / (n_in + n_out))
#    W = np.random.randn(n_in, n_out) * std
#    return W
#
# def he_initialization(n_in, n_out):
#    std = np.sqrt(2 / n_in)
#    W = np.random.randn(n_in, n_out) * std
#    return W
#
# no_of_layers = int(input("Enter number of layers: "))
#
# no_of_neurons = []
#
# for i in range(no_of_layers):
#     no_of_neurons.append(int(input("Enter number of neurons: ")))
# layer_size = [len(X[0])]+no_of_neurons+[len(Y[0])]
#
# weights = []
# for i in range(len(layer_size)-1):
#     weight = he_initialization(layer_size[i], layer_size[i+1])
#     weights.append(weight)
# print(weights)
#
# y_cap=[]
#
# current_input=X
# Z=[]
# A=[]
# for i in range(len(weights)):
#     z = np.dot(current_input, weights[i])
#     Z.append(z)
#     activated = relu(z)
#     A.append(activated)
#     current_input = activated
#
# print("Relu",A)
# print("Z",Z)
#
# y_cap = current_input
# print("y_cap",y_cap)
#
# loss=((y_cap-Y)**2)/2
# print("loss",loss)
#
# #X → [W1] → Z1 → relu → A1 → [W2] → Z2 → relu → A2(=y_cap) → Loss
# #dL/dW2 = dL/dA2 * dA2/dZ2 * dZ2/dW2
# dLdA=np.subtract(y_cap,Y)
# print("dLdA",dLdA)
# dAdZ=dLdA*(Z[1]>0)
# print("dAdZ",dAdZ)
# dzdW=np.dot(A[0].T, dAdZ)
# print("dzdW",dzdW)
#
#
#
# #epochs = int(input("Enter number of epochs: "))
#
# #try to write in class
# # from torch.utils.data import Dataset
# #
# # class XORData(Dataset):
# #     def __init__(self,X,Y):
# #         self.X = X
# #         self.Y = Y
# #
# #     def __len__(self):
# #         return len(self.X)
# #
# #     def __getitem__(self, idx):
# #         return self.X[idx], self.Y[idx]
#
#
import numpy as np


class NeuralNetwork:

    def __init__(self, X, Y, alpha, no_of_layers, no_of_neurons):

        self.X = np.array(X)
        self.Y = np.array(Y)
        self.alpha = alpha

        # [input, hidden layers..., output]
        self.layer_size = [len(X[0])] + no_of_neurons + [len(Y[0])]

        # Initialize weights
        self.weights = []

        for i in range(len(self.layer_size) - 1):
            weight = self.he_initialization(
                self.layer_size[i],
                self.layer_size[i + 1]
            )
            self.weights.append(weight)

        self.Z = []
        self.A = []
        self.y_cap = None
        self.loss = None

    # -----------------------------
    # Activation functions
    # -----------------------------

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    # -----------------------------
    # Weight initialization
    # -----------------------------

    def xavier_initialization(self, n_in, n_out):

        std = np.sqrt(2 / (n_in + n_out))

        W = np.random.randn(n_in, n_out) * std

        return W

    def he_initialization(self, n_in, n_out):

        std = np.sqrt(2 / n_in)

        W = np.random.randn(n_in, n_out) * std

        return W

    # -----------------------------
    # Forward propagation
    # -----------------------------

    def forward(self):

        current_input = self.X

        self.Z = []
        self.A = []

        for i in range(len(self.weights)):

            # Z = XW
            z = np.dot(current_input, self.weights[i])

            self.Z.append(z)

            # Activation
            activated = self.relu(z)

            self.A.append(activated)

            # Output of current layer becomes
            # input to next layer
            current_input = activated

        self.y_cap = current_input

        print("Relu:")
        print(self.A)

        print("\nZ:")
        print(self.Z)

        print("\ny_cap:")
        print(self.y_cap)

        return self.y_cap

    # -----------------------------
    # Loss function
    # -----------------------------

    def calculate_loss(self):

        self.loss = ((self.y_cap - self.Y) ** 2) / 2

        print("\nLoss:")
        print(self.loss)

        return self.loss

    # -----------------------------
    # Backpropagation
    # -----------------------------

    def backward(self):

        # dL/dA2
        dLdA = np.subtract(self.y_cap, self.Y)

        print("\ndL/dA:")
        print(dLdA)

        # dA2/dZ2
        dAdZ = dLdA * (self.Z[1] > 0)

        print("\ndA/dZ:")
        print(dAdZ)

        # dZ2/dW2
        dzdW = np.dot(self.A[0].T, dAdZ)

        print("\ndZ/dW:")
        print(dzdW)

        return dzdW


# =====================================
# Input data
# =====================================

X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

Y = [
    [0],
    [1],
    [1],
    [0]
]


# =====================================
# User input
# =====================================

alpha = float(input("Enter alpha: "))

no_of_layers = int(input("Enter number of layers: "))

no_of_neurons = []

for i in range(no_of_layers):

    n = int(input(f"Enter number of neurons in layer {i + 1}: "))

    no_of_neurons.append(n)


# =====================================
# Create neural network
# =====================================

nn = NeuralNetwork(
    X,
    Y,
    alpha,
    no_of_layers,
    no_of_neurons
)


# =====================================
# Display weights
# =====================================

print("\nWeights:")

for i, weight in enumerate(nn.weights):

    print(f"\nW{i + 1}:")
    print(weight)


# =====================================
# Forward propagation
# =====================================

nn.forward()


# =====================================
# Loss
# =====================================

nn.calculate_loss()


# =====================================
# Backpropagation
# =====================================

nn.backward()