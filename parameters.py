import torch

'''
I will replace the state selection with C matrix multiplies, because my list
comprehension code was HORRENDOUSly complicated. It did work but it is
utterly unmaintainable
'''

# Simulation Parameters
dt, time_start, time_end = 0.001, 0., 10.

# constants
g = 9.81
pi = 3.14159265359

# set an 'infinity' rather than use the torch.inf to ensure that inf * 0 = 0 and not nan
inf = 1e8

# In[initial_conditions]  
''' states in m/s, rad, rad/s '''
npos        = 0.                # m
epos        = 0.                # m
h           = 3048.             # m
phi         = 0.                # rad
theta       = 0.                # rad
psi         = 0.                # rad

vt          = 213.36            # m/s
alpha       = 1.0721 * pi/180   # rad
beta        = 0.                # rad
p           = 0.                # rad/s
q           = 0.                # rad/s
r           = 0.                # rad/s

''' control states in lbs, deg '''
T           = 2886.6468         # lbs
dh          = -2.0385           # deg
da          = -0.087577         # deg
dr          = -0.03877          # deg
lef         = 0.3986            # deg

# In[limits]

npos_min        = -inf       # (m)
epos_min        = -inf       # (m)
h_min           = 0             # (m)
phi_min         = -inf       # (deg)
theta_min       = -inf       # (deg)
psi_min         = -inf       # (deg)
V_min           = 0             # (m/s)
alpha_min       = -20.          # (deg)
beta_min        = -30.          # (deg)
p_min           = -300          # (deg/s)
q_min           = -100          # (deg/s)
r_min           = -50           # (deg/s)

T_min           = 1000          # (lbs)
dh_min          = -25           # (deg)
da_min          = -21.5         # (deg)
dr_min          = -30.          # (deg)
lef_min         = 0.            # (deg)

npos_max        = inf        # (m)
epos_max        = inf        # (m)
h_max           = 100000        # (m)
phi_max         = inf        # (deg)
theta_max       = inf        # (deg)
psi_max         = inf        # (deg)
V_max           = 900           # (m/s)
alpha_max       = 90            # (deg)
beta_max        = 30            # (deg)
p_max           = 300           # (deg/s)
q_max           = 100           # (deg/s)
r_max           = 50            # (deg/s)

T_max           = 19000         # (lbs)
dh_max          = 25            # (deg)
da_max          = 21.5          # (deg)
dr_max          = 30            # (deg)
lef_max         = 25            # (deg)


# the Equations of Motion are written in imperial units so converting to them...
m2f = 3.28084 # metres to feet conversion
f2m = 1/m2f # feet to metres conversion

x0 = torch.tensor([npos*m2f, epos*m2f, h*m2f, phi, theta, psi, vt*m2f, alpha, beta, p, q, r, T, dh, da, dr, lef, -alpha*180/pi])#[np.newaxis].T
x0 = x0.reshape([len(x0),1])

u0 = torch.clone(x0[12:16,:])

states = ['npos','epos','h','phi','theta','psi','V','alpha','beta','p','q','r','T','dh','da','dr','lf2','lf1']
inputs = ['T','dh','da','dr']

observed_states = ['phi','theta','alpha','beta','p','q','r','dh','da','dr','lf2','lf1']
observed_state_idx = [3,4,7,8,9,10,11,13,14,15,16,17]

controlled_inputs = ['dh', 'da', 'dr']
controlled_inputs_idx = [1,2,3]

x_units = ['ft','ft','ft','rad','rad','rad','ft/s','rad','rad','rad/s','rad/s','rad/s','lb','deg','deg','deg','deg','deg']
u_units = ['lb','deg','deg','deg']

x_ub = [npos_max, epos_max, h_max, phi_max, theta_max, psi_max, V_max, alpha_max, beta_max, p_max, q_max, r_max, T_max, dh_max, da_max, dr_max, lef_max, inf]
x_lb = [npos_min, epos_min, h_min, phi_min, theta_min, psi_min, V_min, alpha_min, beta_min, p_min, q_min, r_min, T_min, dh_min, da_min, dr_min, lef_min, -inf]

u_ub = [T_max, dh_max, da_max, dr_max]
u_lb = [T_min, dh_min, da_min, dr_min]

x_ub = torch.tensor(x_ub).unsqueeze(1)
x_lb = torch.tensor(x_lb).unsqueeze(1)

u_ub = torch.tensor(u_ub).unsqueeze(1)
u_lb = torch.tensor(u_lb).unsqueeze(1)

udot_ub = [10000, 60, 80, 120]
udot_lb = [-10000, -60, -80, -120]

udot_ub = torch.tensor(udot_ub).unsqueeze(1)
udot_lb = torch.tensor(udot_lb).unsqueeze(1)

# observed states matrix C
C = torch.tensor([])
for idx in observed_state_idx:
    row = torch.zeros(len(x0)).unsqueeze(0)
    row[0,idx] = 1.
    C = torch.cat((C, row), dim=0)

# assume D matrix to be zeros
D = torch.zeros([12,4])

# initial observation vector using C
y0 = C @ x0

# excludes the thrust
r0 = u0[1:]

# state selection matrix
SSM = torch.tensor([])
for idx in observed_state_idx:
    row = torch.zeros(len(x0)).unsqueeze(0)
    row[0,idx] = 1.
    SSM = torch.cat((SSM, row), dim=0)

# input selection matrix
ISM = torch.tensor([])
for idx in controlled_inputs_idx:
    row = torch.zeros(len(u0)).unsqueeze(0)
    row[0,idx] = 1.
    ISM = torch.cat((ISM, row), dim=0)
