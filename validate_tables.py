import numpy as np
import torch
import matplotlib.pyplot as plt
#from parse_dat import py_lookup
from tqdm import tqdm
from tables import c_lookup, py_parse, py_tables

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

Cx_py = py_parse.tables['CX0120_ALPHA1_BETA1_DH1_201.dat']
CX0820_py = py_parse.tables['CX0820_ALPHA2_BETA1_202.dat']
Cx_lef_py = py_parse.tables['CX0820_ALPHA2_BETA1_202.dat'] - Cx_py[0:14,:,2]

diff_3D = Cx - Cx_py
diff_2D = Cx_lef_py - Cx_lef

#import pdb
#pdb.set_trace()

#plt.plot(Cx[:,:,0])
#plt.plot(Cx_py[:,:,0])
#
#plt.show()

# coefficient to filename (c2f)
c2f = { 'Cx':'CX0120_ALPHA1_BETA1_DH1_201.dat',         # hifi_C
        'Cz':'CZ0120_ALPHA1_BETA1_DH1_301.dat',
        'Cm':'CM0120_ALPHA1_BETA1_DH1_101.dat',
        'Cy':'CY0320_ALPHA1_BETA1_401.dat',
        'Cn':'CN0120_ALPHA1_BETA1_DH2_501.dat',
        'Cl':'CL0120_ALPHA1_BETA1_DH2_601.dat',
        'Cx_lef':'CX0820_ALPHA2_BETA1_202.dat',         # hifi_C_lef
        'Cz_lef':'CZ0820_ALPHA2_BETA1_302.dat',
        'Cm_lef':'CM0820_ALPHA2_BETA1_102.dat',
        'Cy_lef':'CY0820_ALPHA2_BETA1_402.dat',
        'Cn_lef':'CN0820_ALPHA2_BETA1_502.dat',
        'Cl_lef':'CL0820_ALPHA2_BETA1_602.dat',
        'CXq':'CX1120_ALPHA1_204.dat',                  # hifi_damping
        'CYr':'CY1320_ALPHA1_406.dat',
        'CYp':'CY1220_ALPHA1_408.dat',
        'CZq':'CZ1120_ALPHA1_304.dat',
        'CLr':'CL1320_ALPHA1_606.dat',
        'CLp':'CL1220_ALPHA1_608.dat',
        'CMq':'CM1120_ALPHA1_104.dat',
        'CNr':'CN1320_ALPHA1_506.dat',
        'CNp':'CN1220_ALPHA1_508.dat',
        'delta_CXq_lef':'CX1420_ALPHA2_205.dat',        # hifi_damping_lef
        'delta_CYr_lef':'CY1620_ALPHA2_407.dat',
        'delta_CYp_lef':'CY1520_ALPHA2_409.dat',
        'delta_CZq_lef':'CZ1420_ALPHA2_305.dat',
        'delta_CLr_lef':'CL1620_ALPHA2_607.dat',
        'delta_CLp_lef':'CL1520_ALPHA2_609.dat',
        'delta_CMq_lef':'CM1420_ALPHA2_105.dat',
        'delta_CNr_lef':'CN1620_ALPHA2_507.dat',
        'delta_CNp_lef':'CN1520_ALPHA2_509.dat',
        'Cy_r30':'CY0720_ALPHA1_BETA1_405.dat',         # hifi_rudder
        'Cn_r30':'CN0720_ALPHA1_BETA1_503.dat',
        'Cl_r30':'CL0720_ALPHA1_BETA1_603.dat',
        'Cy_a20':'CY0620_ALPHA1_BETA1_403.dat',         # hifi_ailerons
        'Cy_a20_lef':'CY0920_ALPHA2_BETA1_404.dat',
        'Cn_a20':'CN0620_ALPHA1_BETA1_504.dat',
        'Cn_a20_lef':'CN0920_ALPHA2_BETA1_505.dat',
        'Cl_a20':'CL0620_ALPHA1_BETA1_604.dat',
        'Cl_a20_lef':'CL0920_ALPHA2_BETA1_605.dat',
        'delta_CNbeta':'CN9999_ALPHA1_brett.dat',       # hifi_other_coeffs
        'delta_CLbeta':'CL9999_ALPHA1_brett.dat',
        'delta_Cm':'CM9999_ALPHA1_brett.dat',
        'eta_el':'ETA_DH1_brett.dat',
        'delta_Cm_ds':None}

# filename to coefficient inverted table (f2c)
f2c = {value: key for key, value in c2f.items()}

########## testing py_tables
Cx, Cz, Cm, Cy, Cn, Cl = py_tables.hifi_C()

import pdb
pdb.set_trace()
