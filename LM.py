

import math
import numpy as np

pi = np.pi

def det(x):
    return np.linalg.det(x)


def inv(x):
    return np.linalg.inv(x)


def e(x):
    return np.e**(x)

def dot(a,b):
    return np.dot(a,b)

def SVD(A):
    return  np.linalg.svd(A,full_matrices=True)

def eye(n):
    return np.eye(n)

def Gauss(X):
    value = -1*e(-0.5* dot(X.T,X)[0,0])/(2*pi)
    return value

def Gauss_first_derivaive(X):
    return -1*Gauss(X)*X

def Gauss_second_derivaive(X):

    return Gauss(X)*(-1.0*eye(2) + dot(X,X.T))


# LM for Gauss

#init
dim = 2
X = np.matrix([[1],[1]])
u =1
threshold = e(-20)
g_norm = 100
k = 0

while(g_norm>threshold):
    # print "Iteration:",k
    g = Gauss_first_derivaive(X)
    G = Gauss_second_derivaive(X)
    [S,V,D] = SVD(G + u*eye(dim))
    while( 0 in V):
        u = u * 4
        [S, V, D] = SVD(G + u * eye(dim))

    s = dot(inv(G+u*eye(dim)),-g)
    delta_f = Gauss(X+s) - Gauss(X)

    delta_q = dot(g.T,s) + 0.5*dot(dot(s.T,G),s)

    r = delta_f/delta_q

    if r < 0.25:
        u=4*u
    elif r>0.75:
        u=u*0.5

    if r > 0:
        X = X + s
    k = k + 1
    g_norm = dot(g.T,g)[0,0]



print "Iterations:",k
print "Result:X.T = ",X.T

