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
z = py_lookup.parse.axes['DH1'] # DH1

# normalise the data
#norm = max(\
#    x.abs().max(),
#    y.abs().max())
#
#x_norm = x/norm
#y_norm = y/norm 

x_order = 3
y_order = 3
z_order = 2

def f(x, y, z):
    return py_lookup.interp_3d(torch.tensor([x, y, z]), 'Cx')

def polyfit3d(f, x, y, z, x_order, y_order, z_order):

    ncols = (x_order+1)*(y_order+1)*(z_order+1)
    #nrows = max(len(x), len(y))

    powers = []
    ijk = itertools.product(range(x_order+1), range(y_order+1), range(z_order+1))
    for (i,j,k) in ijk:
        powers.append((i,j,k))

 

    # form X
    print('forming X...')
    X = torch.zeros([x.shape[0] * y.shape[0] * z.shape[0], ncols])
    for col in range(ncols):
        xyz = itertools.product(range(len(x)),range(len(y)),range(len(z)))
        for row, (x_idx,y_idx,z_idx) in enumerate(xyz):
            X[row,col] = \
                x[x_idx]**powers[col][0] * \
                y[y_idx]**powers[col][1] * \
                z[z_idx]**powers[col][2]

    # form Y 
    print('forming Y...')
    Y = torch.zeros([X.shape[0],1])
    xyz = itertools.product(range(len(x)),range(len(y)),range(len(z)))
    for row, (x_idx,y_idx,z_idx) in enumerate(xyz):
        Y[row,:] = f(x[x_idx], y[y_idx], z[z_idx])

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

b = polyfit3d(f, x, y, z, x_order, y_order, z_order)

# query point
xp = 90.
yp = 30.
zp = 0.

# form the correct powers of this point
def polyval2d(xp, yp, zp, b, x_order, y_order, z_order):
    ncols = (x_order+1)*(y_order+1)*(z_order+1)
    point = torch.zeros([1,ncols])
    xyz = itertools.product(range(x_order+1),range(y_order+1),range(z_order+1))
    for l, (i,j,k) in enumerate(xyz):
        point[:,l] = xp**i * yp**j * zp**k

    zp = point @ b
    return zp

out_p = polyval2d(xp, yp, zp, b, x_order, y_order, z_order)

import ipdb
ipdb.set_trace()

"""
Time to plot out in 3D

find out what A is in the surface_gen.py script -> replicATE
"""

def poly_test2d(x, y, b, x_order, y_order, f, polyval2d):
    print('forming surface visualisation...')
    X, Y = torch.meshgrid(x, y)
    true_output = torch.zeros([X.shape[0],X.shape[1]])
    poly_output = torch.zeros([X.shape[0],X.shape[1]])
    nx = X.shape[0]*X.shape[1] # number of x values
    ny = Y.shape[0]*Y.shape[1] # number of y values
    for xcol in tqdm(range(X.shape[0])):
        for xrow in range(X.shape[1]):
            for ycol in range(Y.shape[0]):
                for yrow in range(Y.shape[1]):
                    xp = X[xcol,xrow]
                    yp = Y[ycol,yrow]
                    true_output[xcol,yrow] = f(xp,yp)
                    poly_output[xcol,yrow] = polyval2d(xp,yp,b,x_order,y_order)

    """
    form a new set of points to test the surface at
    """ 

    """
    plot the surface that has been created
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, poly_output, rstride=1, cstride=1, alpha=0.2)
    ax.scatter(X, Y, true_output, c='r', s=10) # correct! :)
    plt.xlabel('alpha')
    plt.ylabel('beta')
    ax.set_zlabel('Z')
    #ax.axis('equal')
    #ax.axis('tight')
    plt.show()

#poly_test2d(x, y, b, x_order, y_order, f, polyval2d)
