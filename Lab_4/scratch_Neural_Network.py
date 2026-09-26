import numpy as np
import matplotlib.pyplot as plt


X = [[0, 0, 1],
     [1, 1, 1],
     [1, 0, 1],
     [0, 1, 1]]
Y = [0, 1, 1, 0]
weight = [0.1, 0.1, 0.1]
print("Weight =>", weight)
alpha = float(input("Enter alpha: "))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
loss_history = []
for t in range(1001):
    print("=======================")
    print("Number of iteration:", t)
    # Forward propagation
    dot = np.dot(X, weight)
    Y_cap = []
    for i in range(len(dot)):
        Y_cap.append(sigmoid(dot[i]))
    print("Predicted Y_cap =>", Y_cap)
    # Calculate loss for every sample
    L = []
    for i in range(len(Y_cap)):
        loss = (Y_cap[i] - Y[i])**2 / 2
        L.append(loss)
        print(f"Sample {i+1} | "f"Actual = {Y[i]} | "f"Predicted = {Y_cap[i]:.4f} | "f"Loss = {loss:.4f}")
    # Average loss
    average_loss = sum(L) / len(L)
    print("Average Loss =>", average_loss)
    # Store loss for plotting
    loss_history.append(average_loss)
    # Weight update
    updated_weight = []
    for i in range(len(weight)):
        s = 0
        for j in range(len(Y_cap)):
            s += (Y_cap[j] - Y[j]) * X[j][i]
        new_weight = weight[i] - alpha * s
        updated_weight.append(new_weight)
    weight = updated_weight
    print("Updated weight =>", weight)


# =========================
# Plot Loss Curve
# =========================

plt.plot(range(1001), loss_history)

plt.xlabel("Iteration")
plt.ylabel("Average Loss")
plt.title("Loss vs Iteration")

plt.show()