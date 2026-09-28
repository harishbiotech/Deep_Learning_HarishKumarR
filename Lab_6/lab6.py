import torch

X = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]
Y = [[0], [1], [1], [0]]
X_tensor = torch.tensor(X)
X_tensor_float = X_tensor.float()
Y_tensor = torch.tensor(Y,dtype=torch.float)
# print(X_tensor)
# print(X_tensor.dtype)
# print(X_tensor.shape)
# print(X_tensor_float)
# print(X_tensor_float.dtype)
# print(Y_tensor)
# print(Y_tensor.dtype)
# print(Y_tensor.shape)

#============ gradient ==============

w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)

y = w * x
print("w",w) #
print("x",x)
print("y",y)
y.backward()
print("back propagation",w.grad)

import torch.nn as nn

class XorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2,4)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(4,1)

    def forward(self, x):
         x = self.layer1(x)
         x = self.relu(x)
         x = self.layer2(x)
         output = x
         return output
model = XorNet()
X_tensor = torch.tensor(X, dtype=torch.float)
output = model(X_tensor)
print(output)
print(output.shape)