from tkinter import W
import numpy as np
from tables import py_lookup
import matplotlib.pyplot as plt
import torch
from scipy.optimize import curve_fit
import itertools

# alpha axis array
ALPHA1 = py_lookup.parse.axes['ALPHA1']
ALPHA2 = py_lookup.parse.axes['ALPHA2']

# beta axis array
BETA1 = py_lookup.parse.axes['BETA1']

# el axis array
DH1 = py_lookup.parse.axes['DH1']
DH2 = py_lookup.parse.axes['DH2']

# meshgrid a set of coordinates
X, Y = torch.meshgrid(ALPHA1, BETA1)
XX = X.flatten().unsqueeze(1) # alpha
YY = Y.flatten().unsqueeze(1) # beta

# initialise Cx
Cy = torch.zeros([ALPHA1.shape[0], BETA1.shape[0]])

# Cx is the data that we will be fitting
for i, alpha in enumerate(ALPHA1):
    for j, beta in enumerate(BETA1):
        Cy[i,j] = py_lookup.interp_2d(torch.tensor([alpha, beta]), 'Cy')

def f(alpha, beta):
    return py_lookup.interp_2d(torch.tensor([alpha, beta]), 'Cy')

# form nth order matrices automatically
alpha_n = 20 # order of alpha
beta_n = 19 # order of beta
alpha_list = []
beta_list = []
norm_val = XX.max()
XX = XX/norm_val
YY = YY/norm_val
for i, val in enumerate(range(alpha_n)):
    alpha_list.append(XX**i)
for i, val in enumerate(range(beta_n)):
    beta_list.append(YY**i)
# goes in the order 1, y, y^2, x, xy, xy^2, x^2, x^2y, x^2y^2
test = list(itertools.product(alpha_list, beta_list))

# test is now a list of tuples of 2 elements. We wish to multiply each of these two elements
output = []
for elem in test:
    temp = elem[0]*elem[1]
    output.append(temp)

A = torch.cat(output, axis=1)
B = Cy.flatten()

#import pdb
#pdb.set_trace()
#A = torch.cat(temp_list, axis=1)
#B = Cy.flatten()
####### COMBINATIONS ########
# cartesian product itertools
# or combinations
# itertools product, range alpha_n, range beta_n
# if that fails normalise

##### HPC
# USE torch random inputs random seeded
# try find random seeds and SEND IT on the HPC - order of 100

###### longitudinal f16, inverted pendulum
# minutes - 
# 1. low order polynomial (normalisation optional) fits using combination terms as well
# 2. 3pm friday for meet next week
# 3. longitudinal F16 NLC, MPC comparison in 2 weeks - systems done so that numerical results can be gathered as required
# 4. Constrained NLC applied in the expected fashion - maybe works maybe doesnt. 

#test = curve_fit(f, ALPHA1, BETA1)
#import pdb
#pdb.set_trace()

import ipdb
ipdb.set_trace()

coeff, r, rank, s = torch.linalg.lstsq(A, B)

Z = (A @ coeff).reshape(X.shape)

#Z = np.dot(np.c_[XX*0+1, XX, YY, XX**2, XX**2*YY, XX**2*YY**2, YY**2, XX*YY**2, XX*YY], coeff).reshape(X.shape)

# plot points and fitted surface
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
ax.scatter(X, Y, Cy, c='r', s=10)
plt.xlabel('alpha')
plt.ylabel('beta')
ax.set_zlabel('Z')
#ax.axis('equal')
#ax.axis('tight')
plt.show()


import ipdb
ipdb.set_trace()
Cx = py_lookup.interp_3d(torch.tensor([0.0,0.0,0.0]), 'Cx')

     


