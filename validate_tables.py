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
Cy = torch.zeros([20,19])
Cn = torch.zeros([20,19,5])
Cl = torch.zeros([20,19,5])
for i, alpha in enumerate(tqdm(AXES['ALPHA1'])):
    for j, beta in enumerate(AXES['BETA1']):
        for k, dh in enumerate(AXES['DH1']):
            inp = torch.tensor([alpha, beta, dh])

            Cx[i,j,k] = c_lookup.hifi_C(inp)[0]
        Cy[i,j] = c_lookup.hifi_C(inp)[3]

    print(alpha)

# 2D comparison w/ hifi_C_lef
Cx_lef = torch.zeros([14,19])
Cz_lef = torch.zeros([14,19])
Cm_lef = torch.zeros([14,19])
Cy_lef = torch.zeros([14,19])
Cn_lef = torch.zeros([14,19])
Cl_lef = torch.zeros([14,19])
CX0820 = torch.zeros([14,19])
for i, alpha in enumerate(tqdm(AXES['ALPHA2'])):
    for j, beta in enumerate(AXES['BETA1']):
        inp = torch.tensor([alpha, beta])
        inp2 = torch.tensor([alpha,beta,0])
        Cx_lef[i,j] = c_lookup.hifi_C_lef(inp)[0]

Cx_py = py_lookup.tables['CX0120_ALPHA1_BETA1_DH1_201.dat']
CX0820_py = py_lookup.tables['CX0820_ALPHA2_BETA1_202.dat']
Cx_lef_py = py_lookup.tables['CX0820_ALPHA2_BETA1_202.dat'] - Cx_py[0:14,:,2]

diff_3D = Cx - Cx_py
diff_2D = Cx_lef_py - Cx_lef

import pdb
pdb.set_trace()

plt.plot(Cx[:,:,0])
plt.plot(Cx_py[:,:,0])

plt.show()

import pdb
pdb.set_trace()
