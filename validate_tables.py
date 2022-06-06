import numpy as np
import torch
import matplotlib.pyplot as plt
from parse_dat import py_lookup
from tqdm import tqdm
from tables import c_lookup

# one inference example
alpha = np.array([0.])
beta = np.array([0.])
el = np.array([0.])
inp = torch.tensor([alpha, beta, el])

Cx, Cz, Cm, Cy, Cn, Cl = c_lookup.hifi_C(inp)

# get LUT axes
AXES = torch.load('aerodata_pt/AXES.pt')

# 1D comparison w/ hifi_damping_lef

# 3D comparison w/ hifi_C
Cx = torch.zeros([20,19,5])
Cz = torch.zeros([20,19,5])
Cm = torch.zeros([20,19,5])
Cy = torch.zeros([20,19,5])
Cn = torch.zeros([20,19,5])
Cl = torch.zeros([20,19,5])
for i, alpha in enumerate(tqdm(AXES['ALPHA1'])):
    for j, beta in enumerate(AXES['BETA1']):
        for k, dh in enumerate(AXES['DH1']):
            inp = torch.tensor([alpha, beta, el])

            Cx[i,j,k] = c_lookup.hifi_C(inp)[0]

    print(alpha)

Cx_py = py_lookup.tables['CX0120_ALPHA1_BETA1_DH1_201.dat']

plt.plot(Cx[:,:,0])
plt.plot(Cx_py[:,:,0])

plt.show()

import pdb
pdb.set_trace()
