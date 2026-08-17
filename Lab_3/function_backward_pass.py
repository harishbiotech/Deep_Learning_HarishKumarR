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
XW1.append(float(input("Enter XW1 value")))
XW1.append(float(input("Enter XW1 weight")))
XW2.append(float(input("Enter XW2 value")))
XW2.append(float(input("Enter XW2 weight")))
XW3.append(float(input("Enter XW3 value")))
XW3.append(float(input("Enter XW3 weight")))
XW4.append(float(input("Enter XW4 value")))
XW4.append(float(input("Enter XW4 weight")))
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

def differentiate(d,x,y,z):
    x=sp.Symbol('x')
    y=sp.Symbol('y')
    dy_dx=sp.diff(x,y)
    val=dy_dx.subs(x,z)
    return val

