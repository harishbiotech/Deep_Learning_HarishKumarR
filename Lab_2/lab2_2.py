import numpy as np
# X=np.random.randint(1,10,4)
X=[5,8,7,9]
print(X)
W=np.array([[0.1,-0.1,0.2],
            [-1.1,0.4,1.1],
            [0.2,0.3,0.1],
            [0.7,0.5,-1.2]])
W1=np.array([[-0.1,-0.2],
             [-0.1,0.2],
             [0.3,0.1]])
W2=np.array([[-0.2],
             [-0.1]])
b=1
z=(np.dot(X,W))+b
def relu(x):
    return np.maximum(0,x)

for i in z:
