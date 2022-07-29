import numpy as np

x = np.linspace(0, 1, 20)
y = np.linspace(0, 1, 20)
X, Y = np.meshgrid(x, y, copy=False)
Z = X**2 + Y**2 + np.random.rand(*X.shape)*0.01

X = X.flatten()
Y = Y.flatten()

A = np.array([X*0+1, X, Y, X**2, X**2*Y, X**2*Y**2, Y**2, X*Y**2, X*Y]).T
B = Z.flatten()

coeff, r, rank, s = np.linalg.lstsq(A, B, rcond=-1)


import pdb
pdb.set_trace()