#===================================
#writing a function to semi-automate it
#====================================
import sympy as sp
import math as mt
print('Enter 4 values and 4 weights')
XW1=[]
XW2=[]
XW3=[]
XW4=[]
h=-1
k=1
m=1
XW1.append(float(input("Enter X1 value")))
XW1.append(float(input("Enter W1 weight")))
XW2.append(float(input("Enter X2 value")))
XW2.append(float(input("Enter W2 weight")))
XW3.append(float(input("Enter X3 value")))
XW3.append(float(input("Enter W3 weight")))
XW4.append(float(input("Enter X4 value")))
XW4.append(float(input("Enter W4 weight")))
print(XW1)
print(XW2)
print(XW3)
print(XW4)
a=mt.prod(XW1)
b=mt.prod(XW2)
c=mt.prod(XW3)
d=mt.prod(XW4)
print('a',a)
print('b',b)
print('c',c)
print('d',d)
e=(a+b)
f=c+d
print('e',e)
print('f',f)
g=e+f
print('g',g)
i=g*h
print('i',i)
j=mt.exp(i)
print('j',j)
l=j+k
print('l',l)
n=m/l
print('n',n)

f=c+d
print('f=c+d')
df_dc=sp.diff(f, c)
df_dd=sp.diff(f, d)
print('df_dc',df_dc)
print('df_dd',df_dd)

def differentiate(expression, variable, value=None):
    variable = sp.Symbol(variable)
    expression = sp.sympify(expression)
    derivative = sp.diff(expression, variable,value=None)
    if value is not None:
        derivative = derivative.subs(variable, value)
    return derivative

variabless=["l","j","i","g","f","e","c","d","a","b"]
expressions=["1/l","j+k","E**i","g*h","e+f","e+f","c+d","c+d","a+b","a+b"]
dif=[]
for i in range(len(expressions)):
    print(differentiate(expressions[i], variabless[i]))
    print(expressions[i])
    print("===============================")

