import numpy as np
from tables import py_lookup
import matplotlib.pyplot as plt
import torch
from scipy.optimize import curve_fit

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

# form the 2nd order matrices manually
A = torch.cat([XX*0+1, XX, YY, XX**2, XX**2*YY, XX**2*YY**2, YY**2, XX*YY**2, XX*YY], axis=1)
B = Cy.flatten()

# form nth order matrices automatically
alpha_n = 20 # order of alpha
beta_n = 20 # order of beta
temp_list = []
for i, val in enumerate(range(alpha_n)):
    temp_list.append(XX**i)
for i, val in enumerate(range(beta_n-1)):
    temp_list.append(YY**(i+1))
A = torch.cat(temp_list, axis=1)
B = Cy.flatten()


#test = curve_fit(f, ALPHA1, BETA1)
#import pdb
#pdb.set_trace()

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

import pdb
pdb.set_trace()

Cx = py_lookup.interp_3d(torch.tensor([0.0,0.0,0.0]), 'Cx')

     


