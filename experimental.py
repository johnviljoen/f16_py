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
x_order = 20 # order of alpha
y_order = 19 # order of beta
x_list = []
y_list = []
norm_val = max(XX.abs().max(),YY.abs().max())
XX = XX/norm_val
YY = YY/norm_val
for i, val in enumerate(range(x_order)):
    x_list.append(XX**i)
for i, val in enumerate(range(y_order)):
    y_list.append(YY**i)
# goes in the order 1, y, y^2, x, xy, xy^2, x^2, x^2y, x^2y^2
comb = list(itertools.product(x_list, y_list))

# test is now a list of tuples of 2 elements. We wish to multiply each of these two elements
output = []
for elem in comb:
    output.append(elem[0]*elem[1])

A = torch.cat(output, axis=1)
B = Cy.flatten()

C,_,_,_ = torch.linalg.lstsq(A,B) # torch.linalg.inv(A.T @ A) @ A.T @ B

Z = (A @ C).reshape(X.shape)

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