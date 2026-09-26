#input layer hidden layer and output layer
#Implement a 2-layer (input layer,
# hidden layer and output layer) neural network from scratch for the XOR operation. This includes implementing forward and backward passes from scratch.
# The truth table for XOR is given by
import numpy as np


X = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]
Y = [[0],
     [1],
     [1],
     [0]]
alpha = float(input("Enter alpha: "))

def relu(x):
    return np.maximum(0, x)
def sigmoid(x):
   return 1 / (1 + np.exp(-x))
def tanh(x):
   return np.tanh(x)


def xavier_initialization(n_in, n_out):
   std = np.sqrt(2 / (n_in + n_out))
   W = np.random.randn(n_in, n_out) * std
   return W

def he_initialization(n_in, n_out):
   std = np.sqrt(2 / n_in)
   W = np.random.randn(n_in, n_out) * std
   return W

no_of_layers = int(input("Enter number of layers: "))

no_of_neurons = []

for i in range(no_of_layers):
    no_of_neurons.append(int(input("Enter number of neurons: ")))
layer_size = [len(X[0])]+no_of_neurons+[len(Y[0])]

weights = []
for i in range(len(layer_size)-1):
    weight = he_initialization(layer_size[i], layer_size[i+1])
    weights.append(weight)
print(weights)

y_cap=[]

current_input=X
Z=[]
A=[]
for i in range(len(weights)):
    z = np.dot(current_input, weights[i])
    Z.append(z)
    activated = relu(z)
    A.append(activated)
    current_input = activated

print("Relu",A)
print("Z",Z)

y_cap = current_input
print("y_cap",y_cap)

loss=((y_cap-Y)**2)/2
print("loss",loss)

#X → [W1] → Z1 → relu → A1 → [W2] → Z2 → relu → A2(=y_cap) → Loss
#dL/dW2 = dL/dA2 * dA2/dZ2 * dZ2/dW2
dLdA=np.subtract(y_cap,Y)
print("dLdA",dLdA)
dAdZ=dLdA*(Z[1]>0)
print("dAdZ",dAdZ)
dzdW=np.dot(A[0].T, dAdZ)
print("dzdW",dzdW)



#epochs = int(input("Enter number of epochs: "))

    




