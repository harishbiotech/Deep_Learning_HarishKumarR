#backward propagation of 4 input layer and 1 output layer
# I am hard-coding this question and i will learn class and try to automate it


import sympy as sp
import math as mt

from sympy.polys.densebasic import dup_from_dict

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
differentiation_sigma=n*(1-n)
print('differentiation_sigma',differentiation_sigma)
l=sp.Symbol('l')
n=1/l
df_dl = sp.diff(n,l)
print('df_dl',df_dl)
value_df_dl=df_dl.subs(l,1.0497)
print('value',value_df_dl*1)
k=sp.Symbol('k')
l=1+k
df_dk = sp.diff(l,k)
print('df_dk',df_dk)
value_df_dk=df_dk.subs(l,0.0497)
print('value',value_df_dk*value_df_dl)
i=sp.Symbol('i')
j=sp.exp(i)
df_di=sp.diff(j,i)
print('df_di',df_di)
value_df_di=df_di.subs(i,-3)
print('value',value_df_di.evalf()*value_df_dk)
g=