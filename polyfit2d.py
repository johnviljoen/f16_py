import torch
import numpy as np
import itertools
from tables import py_lookup

"""
Johns thoughts:

ok SO if I just take the 1D x and y values as the query points, they will produce a diagonal line in the
actual lookup table, missing most of the points. Therefore x and y must be meshgrids, which are then
flattened, so that we get all possible points, with all possible lookups.

Idea 2: we instead use an itertools.product to find all combinations of (x,y), which will then be used to
have all z values. This will produce a set of 3D points which can be fit!

this works but a different number of x and y values screws things... must think about the matrices
in the morning!
"""

x = py_lookup.parse.axes['ALPHA1'] # ALPHA1
y = py_lookup.parse.axes['BETA1'] # BETA1
#z = 

x_order = 3
y_order = 2

ij = itertools.product(range(x_order+1), range(y_order+1))
ncols = (x_order+1)*(y_order+1)

X = torch.zeros([max(x.shape, y.shape)[0], ncols])

powers = []
for (i,j) in ij:
    powers.append((i,j))

for idx in range(len(powers)):
    print(idx)



import ipdb
ipdb.set_trace()

for k, (i,j) in enumerate(ij):
    print('------------------------')
    print(f'k is {k}')
    print(f'i is {i}')
    print(f'j is {j}')
    

    
    try:
        X[:,k] = x**i * y**j # only works for equal sizes in dimensions
    except:
        import ipdb
        ipdb.set_trace()


import ipdb
ipdb.set_trace()

def polyfit2d(x, y, z, x_order=3, y_order=3): # 9 for perfect
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(x_order+1), range(y_order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(G, z) # == np.linalg.inv(G.T @ G) @ G.T @ z
    return m

import ipdb
ipdb.set_trace()