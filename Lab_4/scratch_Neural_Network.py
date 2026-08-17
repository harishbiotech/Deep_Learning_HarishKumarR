import numpy as np
X=[[0,0,1],
   [1,1,1],
   [1,0,1],
   [0,1,1]]

Y=[0,
   1,
   1,
   0]
weight=[0.1,
        0.1,
        0.1]

print("weight => ",weight)

def sigmoid(z):
   return 1/(1+np.exp(-z))

alpha=float(input("Enter alpha : "))

dot=np.dot(X,weight)
print("Weight * X value => ",dot)

Y_cap=[]

for i in range(0,len(dot)):
   Y_cap.append(sigmoid(dot[i]))
print("Predicted Y_cap => ",Y_cap)

L=[]

for i in range(0,len(Y_cap)):
   L.append(-1*(Y[i]*np.log(Y_cap[i])+(1-Y[i])*np.log(1-Y_cap[i])))

print("Loss => ",L)

updated_weight=[]
s=[]
for i in range(0,len(X[0])):
   for j in range(0,len(Y_cap)):
      s.append((Y_cap[j]-Y[j])*X[i][j])
   updated_weight.append(weight[i]-alpha*(s[i]))

print("Updated weight => ",updated_weight)