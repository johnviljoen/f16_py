import numpy as np
import torch
import matplotlib.pyplot as plt
#from parse_dat import py_lookup
from tqdm import tqdm
from tables import c_lookup, py_parse, py_lookup

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
Cx, Cz, Cm, Cy, Cn, Cl = py_lookup.hifi_C()


inp = torch.tensor([-2.5, 1.0, 1.0])



# pass alpha, beta, el
Cx, Cz, Cm, Cy, Cn, Cl = c_lookup.hifi_C(inp)

# pass alpha
Cxq, Cyr, Cyp, Czq, Clr, Clp, Cmq, Cnr, Cnp = c_lookup.hifi_damping(inp[0:1])

# pass alpha, beta
delta_Cx_lef, delta_Cz_lef, delta_Cm_lef, delta_Cy_lef, delta_Cn_lef, \
    delta_Cl_lef = c_lookup.hifi_C_lef(inp[0:2])

# pass alpha
delta_Cxq_lef, delta_Cyr_lef, delta_Cyp_lef, delta_Czq_lef, \
    delta_Clr_lef, delta_Clp_lef, delta_Cmq_lef, delta_Cnr_lef, \
        delta_Cnp_lef = c_lookup.hifi_damping_lef(inp[0:1])

# pass alpha, beta
delta_Cy_r30, delta_Cn_r30, delta_Cl_r30 = c_lookup.hifi_rudder(inp[0:2])

# pass alpha, beta
delta_Cy_a20, delta_Cy_a20_lef, delta_Cn_a20, delta_Cn_a20_lef, \
    delta_Cl_a20, delta_Cl_a20_lef = c_lookup.hifi_ailerons(inp[0:2])

# pass alpha, el
delta_Cnbeta, delta_Clbeta, delta_Cm, eta_el, delta_Cm_ds = c_lookup.hifi_other_coeffs(inp[::2])

# hifi_C
# GOOD
diff_Cx = py_lookup.interp_3d(inp, 'Cx') - Cx
diff_Cz = py_lookup.interp_3d(inp, 'Cz') - Cz
diff_Cm = py_lookup.interp_3d(inp, 'Cm') - Cm
diff_Cy = py_lookup.interp_2d(inp[0:2], 'Cy') - Cy
diff_Cn = py_lookup.interp_3d(inp, 'Cn') - Cn
diff_Cl = py_lookup.interp_3d(inp, 'Cl') - Cl

# hifi_damping
# GOOD
diff_Cxq = py_lookup.interp_1d(inp[0:1], 'CXq') - Cxq
diff_Cyr = py_lookup.interp_1d(inp[0:1], 'CYr') - Cyr
diff_Cyp = py_lookup.interp_1d(inp[0:1], 'CYp') - Cyp
diff_Czq = py_lookup.interp_1d(inp[0:1], 'CZq') - Czq
diff_Clr = py_lookup.interp_1d(inp[0:1], 'CLr') - Clr
diff_Clp = py_lookup.interp_1d(inp[0:1], 'CLp') - Clp
diff_Cmq = py_lookup.interp_1d(inp[0:1], 'CMq') - Cmq
diff_Cnr = py_lookup.interp_1d(inp[0:1], 'CNr') - Cnr
diff_Cnp = py_lookup.interp_1d(inp[0:1], 'CNp') - Cnp

# hifi_C_lef
# GOOD
temp = torch.cat([inp[0:2], torch.tensor([0])])
diff_delta_Cx_lef = py_lookup.interp_2d(inp[0:2], 'Cx_lef') - py_lookup.interp_3d(temp, 'Cx') - delta_Cx_lef
diff_delta_Cz_lef = py_lookup.interp_2d(inp[0:2], 'Cz_lef') - py_lookup.interp_3d(temp, 'Cz') - delta_Cz_lef
diff_delta_Cm_lef = py_lookup.interp_2d(inp[0:2], 'Cm_lef') - py_lookup.interp_3d(temp, 'Cm') - delta_Cm_lef
diff_delta_Cy_lef = py_lookup.interp_2d(inp[0:2], 'Cy_lef') - py_lookup.interp_2d(inp[0:2], 'Cy') - delta_Cy_lef
diff_delta_Cn_lef = py_lookup.interp_2d(inp[0:2], 'Cn_lef') - py_lookup.interp_3d(temp, 'Cn') - delta_Cn_lef
diff_delta_Cl_lef = py_lookup.interp_2d(inp[0:2], 'Cl_lef') - py_lookup.interp_3d(temp, 'Cl') - delta_Cl_lef

# hifi_damping_lef
# GOOD
diff_delta_Cxq_lef = py_lookup.interp_1d(inp[0:1], 'delta_CXq_lef') - delta_Cxq_lef
diff_delta_Cyr_lef = py_lookup.interp_1d(inp[0:1], 'delta_CYr_lef') - delta_Cyr_lef
diff_delta_Cyp_lef = py_lookup.interp_1d(inp[0:1], 'delta_CYp_lef') - delta_Cyp_lef
diff_delta_Czq_lef = py_lookup.interp_1d(inp[0:1], 'delta_CZq_lef') - delta_Czq_lef # this being unused is not an error, it is as the original C was written, you can delete if you like.
diff_delta_Clr_lef = py_lookup.interp_1d(inp[0:1], 'delta_CLr_lef') - delta_Clr_lef
diff_delta_Clp_lef = py_lookup.interp_1d(inp[0:1], 'delta_CLp_lef') - delta_Clp_lef
diff_delta_Cmq_lef = py_lookup.interp_1d(inp[0:1], 'delta_CMq_lef') - delta_Cmq_lef
diff_delta_Cnr_lef = py_lookup.interp_1d(inp[0:1], 'delta_CNr_lef') - delta_Cnr_lef
diff_delta_Cnp_lef = py_lookup.interp_1d(inp[0:1], 'delta_CNp_lef') - delta_Cnp_lef

# hifi_rudder
# GOOD
diff_delta_Cy_r30 = py_lookup.interp_2d(inp[0:2], 'Cy_r30') - py_lookup.interp_2d(inp[0:2], 'Cy') - delta_Cy_r30
diff_delta_Cn_r30 = py_lookup.interp_2d(inp[0:2], 'Cn_r30') - py_lookup.interp_3d(temp, 'Cn') - delta_Cn_r30
diff_delta_Cl_r30 = py_lookup.interp_2d(inp[0:2], 'Cl_r30') - py_lookup.interp_3d(temp, 'Cl') - delta_Cl_r30

# hifi_ailerons
# DISCREPANCIES HERE
diff_delta_Cy_a20     = py_lookup.interp_2d(inp[0:2], 'Cy_a20') - py_lookup.interp_2d(inp[0:2], 'Cy') - delta_Cy_a20
diff_delta_Cy_a20_lef = py_lookup.interp_2d(inp[0:2], 'Cy_a20_lef') - py_lookup.interp_2d(inp[0:2], 'Cy_lef') - delta_Cy_a20_lef # discrepancies
diff_delta_Cn_a20     = py_lookup.interp_2d(inp[0:2], 'Cn_a20') - py_lookup.interp_3d(temp, 'Cn') - delta_Cn_a20
diff_delta_Cn_a20_lef = py_lookup.interp_2d(inp[0:2], 'Cn_a20_lef') - py_lookup.interp_2d(inp[0:2], 'Cn_lef') - delta_Cn_a20_lef # discrepancies
diff_delta_Cl_a20     = py_lookup.interp_2d(inp[0:2], 'Cl_a20') - py_lookup.interp_3d(temp, 'Cl') - delta_Cl_a20
diff_delta_Cl_a20_lef = py_lookup.interp_2d(inp[0:2], 'Cl_a20_lef') - py_lookup.interp_2d(inp[0:2], 'Cl_lef') - delta_Cl_a20_lef # discrepancies

# hifi_other_coeffs
# GOOD
diff_delta_Cnbeta = py_lookup.interp_1d(inp[0:1], 'delta_CNbeta') - delta_Cnbeta
diff_delta_Clbeta = py_lookup.interp_1d(inp[0:1], 'delta_CLbeta') - delta_Clbeta
diff_delta_Cm = py_lookup.interp_1d(inp[0:1], 'delta_Cm') - delta_Cm
diff_eta_el = py_lookup.interp_1d(inp[2:3], 'eta_el') - eta_el
diff_delta_Cm_ds = 0.


import pdb
pdb.set_trace()
