import numpy as np

X = [[0, 0, 1],
     [1, 1, 1],
     [1, 0, 1],
     [0, 1, 1]]
Y = [0,1,1,0]
weight = [0.1,0.1,0.1]
print("Weight =>", weight)
alpha = float(input("Enter alpha: "))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

for t in range(1001):

    print("=======================")
    print("Number of iteration:", t)

    dot = np.dot(X, weight)
    Y_cap = []

    for i in range(len(dot)):
        Y_cap.append(sigmoid(dot[i]))
    print("Predicted Y_cap =>", Y_cap)

    L = []
    for i in range(len(Y_cap)):
        loss = -1 * (Y[i] * np.log(Y_cap[i])+ (1 - Y[i]) * np.log(1 - Y_cap[i]))
        L.append(loss)
    print("Loss =>", L)

    updated_weight = []

    for i in range(len(weight)):
        s = 0
        for j in range(len(Y_cap)):
            s += (Y_cap[j] - Y[j]) * X[j][i]
        new_weight = weight[i] - alpha * s
        updated_weight.append(new_weight)

    weight = updated_weight

    print("Updated weight =>", weight)