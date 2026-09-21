import torch
#backward is for the backpropagation
x=torch.randn(1,requires_grad=True)
print(x)
y=x+2
print(y)
y.backward()
print(x.grad)
weights = torch.ones(4,requires_grad=True)
for epoch in range(1):
    model_output=(weights*3).sum()
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_()
