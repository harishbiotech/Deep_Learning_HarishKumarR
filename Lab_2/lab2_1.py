import numpy as np
X=np.random.randint(1,10,4)
W=np.random.randint(1,5,4)
b=int(input('Enter the bias value: '))
z=np.dot(X,W)+b
# d=np.dot(X,W)
# print(d)
print('X values',X)
print('Weight',W)
print('Dot product is equal to the summation of Xi*Wi:',z)
def ReLU(x):
    return np.maximum(0,x)
print('y^ =',ReLU(z))
