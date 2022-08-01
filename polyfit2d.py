from re import I
import torch
import numpy as np
import itertools
from tables import py_lookup
from tqdm import tqdm
import matplotlib.pyplot as plt

"""

"""

x = py_lookup.parse.axes['ALPHA1'] # ALPHA1
y = py_lookup.parse.axes['BETA1'] # BETA1
#z = 

x_order = 3
y_order = 2

def f(x, y):
    return py_lookup.interp_2d(torch.tensor([x, y]), 'Cy')

def polyfit2d(f, x, y, x_order, y_order):

    ncols = (x_order+1)*(y_order+1)
    #nrows = max(len(x), len(y))

    powers = []
    ij = itertools.product(range(x_order+1), range(y_order+1))
    for (i,j) in ij:
        powers.append((i,j))

    # form X
    print('forming X...')
    X = torch.zeros([x.shape[0] * y.shape[0], ncols])
    for col in range(ncols):
        xy = itertools.product(range(len(x)),range(len(y)))
        for row, (x_idx,y_idx) in enumerate(xy):
            X[row,col] = x[x_idx]**powers[col][0] * y[y_idx]**powers[col][1]

    # form Y 
    print('forming Y...')
    Y = torch.zeros([X.shape[0],1])
    xy = itertools.product(range(len(x)),range(len(y)))
    for row, (x_idx,y_idx) in enumerate(xy):
        Y[row,:] = f(x[x_idx], y[y_idx])

    # using least squares find the coefficients
    b = torch.linalg.inv(X.T @ X) @ X.T @ Y

    return b

"""
Johns Thoughts:

Ok SO, we now have a candidate b vector, which should be our coefficients.
This must be tested however so we need a little function now that produces the 
rows of X for a given point such that we can interrogate the function.

This will also let us plot it
"""

b = polyfit2d(f, x, y, x_order, y_order)

# query point
xp = 90.
yp = 30.

# form the correct powers of this point
def polyval2d(xp, yp, b, x_order, y_order):
    ncols = (x_order+1)*(y_order+1)
    point = torch.zeros([1,ncols])
    xy = itertools.product(range(x_order+1),range(y_order+1))
    for k, (i,j) in enumerate(xy):
        point[:,k] = xp**i * yp**j

    zp = point @ b
    return zp

zp = polyval2d(xp, yp, b, x_order, y_order)

"""
Time to plot out in 3D

find out what A is in the surface_gen.py script -> replicATE
"""
X, Y = torch.meshgrid(x, y)

print('forming surface visualisation...')
output = torch.zeros([X.shape[0],X.shape[1]])
nx = X.shape[0]*X.shape[1] # number of x values
ny = Y.shape[0]*Y.shape[1] # number of y values
for xcol in tqdm(range(X.shape[0])):
    for xrow in range(X.shape[1]):
        for ycol in range(Y.shape[0]):
            for yrow in range(Y.shape[1]):
                xp = X[xcol,xrow]
                yp = Y[ycol,yrow]
                output[xcol,yrow] = f(xp,yp)

"""
plot the surface that has been created
"""
fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.plot_surface(X, Y, output, rstride=10, cstride=10, alpha=0.2)
ax.scatter(X, Y, output, c='r', s=10) # correct! :)
plt.xlabel('alpha')
plt.ylabel('beta')
ax.set_zlabel('Z')
#ax.axis('equal')
#ax.axis('tight')
plt.show()

import ipdb
ipdb.set_trace()