import numpy as np
import matplotlib.pyplot as plt

z=np.arange(-10,10,0.2)

def Sigmoid(x):
    return 1/(1+np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0,x)

def Softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

A=input("Enter Sigmoid or Tanh or relu or Softmax: ")
if A=="Sigmoid" or A=="sigmoid":
    sigmoid=Sigmoid(z)
    plt.plot(z,sigmoid)
    plt.xlabel("z")
    plt.ylabel("Sigmoid")
    plt.title("Sigmoid_curve")
    plt.grid(True)
    plt.show()
    print("The min is -1 & max is 1")
    print('Tanh is not zero centered')
    print('if the input is very small the sigmoid function returns 0.5 \n if the input is very large it returns 1')
elif A=="Tanh" or A=="tanh":
    Tanh=tanh(z)
    plt.plot(z,Tanh)
    plt.xlabel("z")
    plt.ylabel("Tanh")
    plt.title("Tanh_curve")
    plt.grid(True)
    plt.show()
    print("The min is -1 & max is 1")
    print('Tanh is zero centered')
elif A=="relu":
    relu=relu(z)
    plt.plot(z,relu)
    plt.xlabel("z")
    plt.ylabel("ReLU")
    plt.title("ReLU_curve")
    plt.grid(True)
    plt.show()
    print("The min is 0 & max is 9.8")
    print('ReLU is zero centered')
elif A=="Softmax" or A=="softmax":
    softmax=Softmax(z)
    plt.plot(z,softmax)
    plt.xlabel("z")
    plt.ylabel("Softmax")
    plt.title("Softmax_curve")
    plt.grid(True)
    plt.show()
    print("The min is 0 & max is 0.18138")
    print('ReLU is zero centered')
else:
    print("Invalid input")



print("=============================")
a=[0.0000001,0.000001,0.00001]
print(tanh(a))
print(relu(a))
print(Softmax(a))
print("============================")
b=[10,100,1000]
print(tanh(b))
print(relu(b))
print(Softmax(b))