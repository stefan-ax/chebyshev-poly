'''
    DOCSTRING
'''
__author__ = 'Stefan A. Obada'
__date__ = 'Tue Mar 10 19:47:08 2020'

import numpy as np
import matplotlib.pyplot as plt

# Set N
N = 20
# Create vector of evaluations
ev = np.zeros(shape=(N+1,1))
ev[-1] = 40

# Define functions T and its derivative, T_dot recursive
def T(x, k):
    if k==0:
        return 1
    elif k==1:
        return x
    else:
        return 2*T(x, k-1)*x - T(x, k-2)

def T_dot(x, k):
    if k==0:
        return 0
    elif k==1:
        return 1
    else:
        return 2*T(x, k-1) + 2*x*T_dot(x, k-1) - T_dot(x, k-2)
    
# Create the Gauss-Lobatto linear space
X = []
for j in range(N+1):
    X.append(np.cos(j * np.pi / N))
    
# Create D - matrix
def D_jk(j, k):
    return ( (k**2 / (1 - X[j]**2)) - 40**3 * X[j] ) * T(X[j], k) + ( X[j] / (1 - X[j]**2) ) * T_dot(X[j], k)

D = np.zeros(shape=(N+1, N+1)) 

for j in range(N+1):
    print(f'Row {j}')
    if j==0:
        for k in range(N+1):
            D[j, k] = T(X[j], k)
    elif j==N:
        for k in range(N+1):
            D[j, k] = T_dot(X[j], k)
    else:
        for k in range(N+1):
            D[j, k] = D_jk(j, k)
            
# LSE for solution
D = np.array(D)

# a = np.linalg.inv(D.T.dot(D)).dot(D.T).dot(ev)
a = np.linalg.inv(D).dot(ev)

# Define the solution function y
def y(x):
    Tk_vector = np.array([T(x/40, k) for k in range(N+1)])
    return Tk_vector.dot(a)[0]

# Plot the solution
linsp = np.linspace(start=-40, stop=40, num=300)
plt.plot(linsp, [y(x) for x in linsp], c='red')


