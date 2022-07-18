import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from parameters import x0, u0, y0, r0, C, D, dt, SSM, ISM, x_ub, x_lb, u_ub, u_lb, udot_ub, udot_lb, ltt
from tables import c_lookup
from dynamics import Nlplant
from control import Linmod, LMPC

class F16(nn.Module):
    def __init__(self, device, dtype):
        super().__init__()
        
        self.device = device
        self.dtype = dtype

        self.x = x0.to(device).type(dtype)
        self.x0 = torch.clone(self.x)
        self.u = u0.to(device).type(dtype)
        self.u0 = torch.clone(self.u)
        self.y = y0.to(device).type(dtype)
        self.y0 = torch.clone(self.y)
        self.r = r0.to(device).type(dtype)
        self.r0 = torch.clone(self.r)
        self.C = C.to(device).type(dtype)
        self.D = D.to(device).type(dtype)
        self.dt = torch.tensor(dt, device=device, dtype=dtype)
        self.SSM = SSM.to(device).type(dtype)
        self.ISM = ISM.to(device).type(dtype)
        self.x_ub = x_ub.to(device).type(dtype)
        self.x_lb = x_lb.to(device).type(dtype)
        self.u_ub = u_ub.to(device).type(dtype)
        self.u_lb = u_lb.to(device).type(dtype)
        self.udot_ub = udot_ub.to(device).type(dtype)
        self.udot_lb = udot_lb.to(device).type(dtype)

        # number of inputs
        self.p = len(self.u)

        # number of outputs
        self.q = len(self.y)

        # number of states
        self.n = len(self.x)

        # dynamics
        self.nlplant = Nlplant(device, dtype, ltt)

        # linearisation
        self.linmod = Linmod(
                device, 
                dtype,
                self.nlplant,
                self.n,
                self.p,
                self.C,
                self.D,
                self.dt,
                eps=1e-02) # with C matrix of 1's, large EPS needed for B matrix (units)

        # placeholder linearisation
        self.A, self.B, _, _ = self.linmod(self.x, self.u)

        # assuming that C is only 1's and 0's, this selects the correct rows and cols from A
        Ar = self.SSM @ self.A @ self.SSM.T
        Br = self.SSM @ self.B @ self.ISM.T

        y_lb = self.SSM @ self.x_lb
        y_ub = self.SSM @ self.x_ub
        
        r_ub = self.ISM @ self.u_ub
        r_lb = self.ISM @ self.u_lb

        rdot_ub = self.ISM @ self.udot_ub
        rdot_lb = self.ISM @ self.udot_lb

        self.lmpc = LMPC(
                device,
                dtype,
                self.dt,
                self.y,
                self.r,
                Ar,
                Br,
                self.C,
                self.D,
                y_lb,
                y_ub,
                r_lb,
                r_ub,
                rdot_lb,
                rdot_ub,
                hzn=10)

    def step(self, x, r):
        
        y = self.obs(x)
        u = self.mpc(y, r)
        xdot = self.nlplant(x, u)[0]
        x += xdot * self.dt
        return x
        

    def obs(self, x):
        return self.C @ self.x
         
    def mpc(self, y, r):
        # y is the states observed by the MPC
        # r is the reference input

        return r
        

# cuda is slow when using c_lookups as need to go to .detach().cpu().numpy() 
# to call the c lookup
# further than this in Linmod the scipy implementation requires conversion to numpy and back
device = torch.device('cpu')
dtype = torch.float32

############# testing as I go #############

from parameters import observed_states, states, pi

# instantiation of f16
f16 = F16(device, dtype)

# testing the production of the dlqr K matrix
f16.lmpc.dlqr()

# testing the CLP simulation of the dlqr K matrix on the linearised system
ts = 1000
out = torch.zeros([12,ts])
rp = 0.
rq = 2. 
rr = 0.
#y0 = torch.clone(f16.y0)
f16.y0[4:7] = torch.tensor([rp,rq,rr], device=device, dtype=dtype).unsqueeze(1)
#for i in tqdm(range(ts)):
#    f16.r = - f16.lmpc.K @ (f16.y - f16.y0)
#    f16.y = f16.lmpc.A @ f16.y + f16.lmpc.B @ f16.r
#    metric_units_y = torch.clone(f16.y)
#
#    metric_units_y[:7] = metric_units_y[:7]*180/pi
#    out[:,i:i+1] = metric_units_y
#
#t = torch.linspace(0,ts/1000,ts)
#fig,axs = plt.subplots(12,1)
#for i in range(12):
#    axs[i].plot(t, out[i,:])
#    axs[i].set(ylabel=f'{observed_states[i]}')
#fig.suptitle(f'linear system closed loop with DLQR controller \n p_cmd={rp}, q_cmd={rq}, r_cmd={rr}')
#plt.show()

# testing the CLP simulation of the LMPC implementation on the linearised system
#for i in tqdm(range(ts)):
#    f16.r = f16.lmpc(f16.y, f16.y0)
#    f16.y = f16.lmpc.A @ f16.y + f16.lmpc.B @ f16.r
#    metric_units_y = torch.clone(f16.y)
#
#    metric_units_y[:7] = metric_units_y[:7]*180/pi
#    out[:,i:i+1] = metric_units_y
#
#t = torch.linspace(0,ts/1000,ts)
#fig,axs = plt.subplots(12,1)
#for i in range(12):
#    axs[i].plot(t, out[i,:])
#    axs[i].set(ylabel=f'{observed_states[i]}')
#fig.suptitle(f'linear system closed loop with LMPC controller \n p_cmd={rp}, q_cmd={rq}, r_cmd={rr}')
#plt.show()


# f16 lmpc testing
f16.lmpc.calc_MC()

ro = f16.lmpc(f16.y, f16.y*1.001)

# instantiation of nlplant
nlplant = Nlplant(device, dtype, ltt)

# example call of nlplant to find xdot
nlplant(f16.x, f16.u)

# example single time step
print(f16.step(f16.x, f16.u))

# test linearisation and cont2discrete (which is verified)
# note that the way that I am treating with imperial units directly
# means that a large EPS is required to capture the B matrix dynamics
# and likely other matrices, but I havent explicitly seen them.
print(f16.linmod(f16.x, f16.u))


# example simulation
out = torch.zeros([18,ts])
for i in tqdm(range(ts)):
    f16.u[1:] = -f16.lmpc(f16.y, f16.y0) + f16.u0[1:]
    #f16.y = f16.lmpc.A @ f16.y + f16.lmpc.B @ f16.u[1:]
    f16.x = f16.step(f16.x, f16.u)
    f16.y = f16.SSM @ f16.x
    metric_units_x = torch.clone(f16.x)
    metric_units_x[9] = metric_units_x[9]
    metric_units_x[10] = metric_units_x[10]
    metric_units_x[11] = metric_units_x[11]

    out[:,i:i+1] = f16.x


t = torch.linspace(0,ts/1000,ts)
fig,axs = plt.subplots(18,1)
for i in range(18):
    axs[i].plot(t, out[i,:])
    axs[i].set(ylabel=f'{states[i]}')
fig.suptitle(f'nonlinear system closed loop with LMPC controller \n p_cmd={rp}, q_cmd={rq}, r_cmd={rr}')
#plt.show()

plt.savefig('test.png')

import pdb
pdb.set_trace()

# example c_lookup test
print(c_lookup.hifi_C(torch.tensor([0.1,0.1,0.1])))

#########################################

import pdb
pdb.set_trace()
