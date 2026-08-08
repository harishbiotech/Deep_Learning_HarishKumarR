#backward propagation of 4 input layer and 1 output layer
# I am hard-coding this question and i will learn class and try to automate it
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
differentiation_sigma=n*(1-n)
print('differentiation_sigma',differentiation_sigma)
l=sp.Symbol('l')
n=1/l
print('n=1/l')
df_dl = sp.diff(n,l)
print('df_dl',df_dl)
value_df_dl=df_dl.subs(l,1.0497)
glo1=value_df_dl*1
print('glo1',glo1)
k=sp.Symbol('k')
l=1+k
print('l=1+k')
df_dk = sp.diff(l,k)
print('df_dk',df_dk)
value_df_dk=df_dk.subs(l,0.0497)
glo2=value_df_dk*glo1
print('glo2',glo2)
i=sp.Symbol('i')
j=sp.exp(i)
print('j=e^i')
df_di=sp.diff(j,i)
print('df_di',df_di)
value_df_di=df_di.subs(i,-3)
glo3=value_df_di.evalf()*glo2
print('glo3',glo3)
g=sp.Symbol('g')
i=-1*g
print('i=-1*g')
df_dg=sp.diff(i,g)
print('df_dg',df_dg)
value_df_dg=df_dg.subs(i,3)
glo4=value_df_dg*glo3
print('glo4',glo4)
e = sp.Symbol('e')
f = sp.Symbol('f')
g = e + f
print('g=e+f')
df_de=sp.diff(g, e)
df_df=sp.diff(g, f)
print('df_de',df_de)
print('df_df',df_df)
value_df_de=df_de.subs(e,0.5)
glo5=value_df_de*glo4
print('glo5',glo5)
value_df_df=value_df_de.subs(f,2.5)
glo6=value_df_df*glo4
print('glo6',glo6)
c = sp.Symbol('c')
d = sp.Symbol('d')
f=c+d
print('f=c+d')
df_dc=sp.diff(f, c)
df_dd=sp.diff(f, d)
print('df_dc',df_dc)
print('df_dd',df_dd)
value_df_dc=df_dc.subs(c,0.888)
glo7=value_df_dc*glo6
print('glo7',glo7)
value_df_dd=df_dd.subs(d,1.6)
glo8=value_df_dd*glo6
print('glo8',glo8)
a=sp.Symbol('a')
b=sp.Symbol('b')
e=a+b
print('e=a+b')
df_da=sp.diff(e, a)
df_db=sp.diff(e, b)
value_df_da=df_da.subs(a,0.1)
glo9=value_df_da*glo5
print('glo9',glo9)
value_df_db=df_db.subs(b,0.4)
glo10=value_df_db*glo5
print('glo10',glo10)
W1=sp.Symbol('W1')
X1=sp.Symbol('X1')
a=W1*X1
df_W1=sp.diff(a, W1)
print('df_W1',df_W1)
value_df_W1=df_W1.subs(W1,0.1)
print('value',value_df_W1*glo9)
df_X1=sp.diff(a,X1)
value_df_X1=df_X1.subs(X1,1)
df_X1=sp.diff(a, W1)
print('df_X1',df_X1)
print('value_df_X1',value_df_X1*glo9)
W2=sp.Symbol('W2')
X2=sp.Symbol('X2')
b=W2*X2
df_W2=sp.diff(b, W2)
print('df_W2',df_W2)
value_df_W2=df_W2.subs(W1,0.2)
print('value_df_W1',value_df_W2*glo10)
df_X2=sp.diff(b,X2)
value_df_X2=df_X2.subs(X2,2)
df_X2=sp.diff(a, W1)
print('df_X1',df_X2)
print('value_df_X1',value_df_X2*glo10)
W3=sp.Symbol('W3')
X3=sp.Symbol('X3')
c=W3*X3
df_X3=sp.diff(c, X3)
print('df_x3',df_X3)
value_df_X3=df_X3.subs(X3,3)
print('value_df_x3',value_df_X3*glo7)
df_W3=sp.diff(c, W3)
print('df_W3',df_W3)
value_df_W3=df_W3.subs(W3,0.3)
print('value_df_W3',value_df_W3*glo7)
W4=sp.Symbol('W4')
X4=sp.Symbol('X4')
d=W4*X4
df_X4=sp.diff(c, X4)
print('df_X4',df_X4)
value_df_X4=df_X4.subs(X4,4)
print('value_df_X4',value_df_X4*glo8)
df_W4=sp.diff(d, W4)
print('df_W4',df_W4)
value_df_X4=df_X4.subs(W4,0.4)
print('value_df_X4',value_df_X4*glo8)
