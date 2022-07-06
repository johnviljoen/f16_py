import torch
import torch.nn as nn
import scipy
import scipy.linalg
import osqp

def dmom(mat, num_mats):
    # diagonal matrix of matrices -> dmom
    
    # dimension extraction
    nrows = mat.shape[0]
    ncols = mat.shape[1]
    
    # matrix of matrices matomats -> I thought it sounded cool
    matomats = np.zeros((nrows*num_mats,ncols*num_mats))
    
    for i in range(num_mats):
        for j in range(num_mats):
            if i == j:
                matomats[nrows*i:nrows*(i+1),ncols*j:ncols*(j+1)] = mat
                
    return matomats

def cont2discrete(A, B, C, D, dt):
    # this is a port of the 'zoh' scipy cont2discrete to pytorch
    em_upper = torch.hstack((A, B))

    em_lower = torch.hstack((torch.zeros((B.shape[1], A.shape[0])),
                             torch.zeros((B.shape[1], B.shape[1]))))

    em = torch.vstack((em_upper, em_lower))
    ms = torch.linalg.matrix_exp(em * dt)

    # Dispose of the lower rows
    ms = ms[:A.shape[0], :]

    Ad = ms[:, 0:A.shape[1]]
    Bd = ms[:, A.shape[1]:]

    return Ad, Bd, C, D

# In[ discrete linear quadratic regulator ]
# from https://github.com/python-control/python-control/issues/359:
def dlqr(A,B,Q,R):
    """
    Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]


    Discrete-time Linear Quadratic Regulator calculation.
    State-feedback control  u[k] = -K*(x_ref[k] - x[k])
    select the states that you want considered and make x[k] the difference
    between the current x and the desired x.

    How to apply the function:
        K = dlqr(A_d,B_d,Q,R)

    Inputs:
      A_d, B_d, Q, R  -> all numpy arrays  (simple float number not allowed)

    Returns:
      K: state feedback gain

    """
    # first, solve the ricatti equation
    P = np.array(scipy.linalg.solve_discrete_are(A, B, Q, R))
    # compute the LQR gain
    K = np.array(scipy.linalg.inv(B.T @ P @ B+R) @ (B.T @ P @ A))
    return K

class Linmod(nn.Module):

    """
    Function to linearise any general input output system numerically. This should
    generally be considered a last resort as significant parts of systems can usually be
    linearised analytically, which is much faster. However, in this case I just
    need something that I know works.

    after instantiation, the forward function takes the states and the inputs, and
    returns the discrete time A,B,C,D matrices
    """

    def __init__(self, device, dtype, nlplant, num_states, num_inps, C, D, dt, eps):
        super().__init__()

        # continuous nonlinear 'plant', the input output system to be linearised
        self.calc_xdot = nlplant

        # the disturbance to be used to numerically find gradients
        self.eps = eps

        # the timestep to be used for discretisation 
        self.dt = dt

        # number of states
        self.num_states = num_states
        
        # number of inputs
        self.num_inps = num_inps
       
        # instantiate everything before forward pass
        self.A = torch.zeros([num_states, num_states], device=device, dtype=dtype)
        self.B = torch.zeros([num_states, num_inps], device=device, dtype=dtype)

        self.dx = torch.zeros([num_states, 1], device=device, dtype=dtype)
        self.du = torch.zeros([num_inps, 1], device=device, dtype=dtype)
        self.C = C # assumed constant here 
        self.D = D # assumed constant here

        self.I = torch.eye(num_states, device=device, dtype=dtype)

    def __call__(self, x, u):
        return self.forward(x, u)

    def forward(self, x, u):
        # Perturb each of the state variables and compute linearisation
        for i in range(self.num_states):

            self.dx *= 0
            self.dx[i] = self.eps

            self.A[:,i:i+1] = (self.calc_xdot(x + self.dx, u)[0] - self.calc_xdot(x, u)[0]) / self.eps

        for i in range(self.num_inps):

            self.du *= 0
            self.du[i] = self.eps

            self.B[:,i:i+1] = (self.calc_xdot(x, u + self.du)[0] - self.calc_xdot(x, u)[0]) / self.eps

        return cont2discrete(self.A, self.B, self.C, self.D, self.dt)

class LMPC(nn.Module):

    """
    This is a linear MPC (LMPC) implementation. This means that it uses a single state space model to
    perform predictions with. This assumes that the linear model is representative of whatever nonlinear
    model it is being applied to.

    This is in contrast to NLMPC (nonlinear MPC), and CLMPC (continuous linearisation MPC)
    """

    def __init__(self, device, dtype, dt, y, r, A, B, C, D, y_lb, y_ub, r_lb, r_ub, rdot_lb, rdot_ub, hzn=10):
        super().__init__()
        self.device = device
        self.dtype = dtype


        self.dt = dt
        self.y = y
        self.r = r

        self.A = A
        self.B = B
        self.C = C
        self.D = D

        #self.Q = C @ C.T
        self.Q = torch.zeros([C.shape[0]]*2)
        self.Q[4,4] = 1.    # p
        self.Q[5,5] = 10.   # q
        self.Q[6,6] = 1.    # r
        #self.Q = torch.eye(C.shape[0])
        self.R = torch.eye(B.shape[1])
        self.QQ = torch.block_diag(*[self.Q]*hzn)
        self.RR = torch.block_diag(*[self.R]*hzn)

        self.nstates = A.shape[0]
        self.ninputs = B.shape[1]
        self.hzn = hzn

        self.CC = torch.zeros([self.nstates*hzn, self.ninputs*hzn], device=device, dtype=dtype)
        self.MM = torch.zeros([self.nstates*hzn, self.nstates], device=device, dtype=dtype)
        self.Bz = torch.zeros([self.nstates, self.ninputs], device=device, dtype=dtype)
        
        # as we are not doing anything but linear MPC MM, CC, and K can be calculated offline

        # calculate MM, CC, assign to self.MM, self.CC
        self.calc_MC()

        # calculate K, assign to self.K
        self.K = self.dlqr()

        """
        There are three constraints to be enforced on the system:

            state constraints:
                x(n+1) = Ax(n) + Bu(n)

            input command limit constraints:
                u_min <= u <= u_max

            input command rate limit constraints:
                udot_min <= udot <= udot_max
        """
        # calculate state constraint limits vector

            
        self.y_lb = torch.tile(y_lb,(hzn,1))
        self.y_ub = torch.tile(y_ub,(hzn,1))
        
        self.state_constr_lower = self.y_lb - self.MM @ y
        self.state_constr_upper = self.y_ub - self.MM @ y
        
        # the state constraint input sequence matrix is just CC
        
        # calculate the command saturation limits vector
        
        self.cmd_constr_lower = torch.tile(r_lb,(hzn,1))
        self.cmd_constr_upper = torch.tile(r_ub,(hzn,1))

        # calculate the command saturation input sequence matrix -> just eye 
        self.cmd_constr_mat = torch.eye(self.ninputs*hzn,device=device,dtype=dtype)

        self.rdot_lb = rdot_lb
        self.rdot_ub = rdot_ub


    def __call__(self, y, r):
        return self.forward(y, r)
    
    def dlqr(self):
        # first, solve the ricatti equation
        P = torch.tensor(scipy.linalg.solve_discrete_are(self.A, self.B, self.Q, self.R),device=self.device,dtype=self.dtype)
        # compute the LQR gain
        K = torch.tensor(scipy.linalg.inv(self.B.T @ P @ self.B+self.R),device=self.device, dtype=self.dtype) @ (self.B.T @ P @ self.A)
        return K

    def calc_MC(self):
        # sets the self.MM and self.CC matrices from self.A and self.B and self.Bz
        for i in range(self.hzn):
            self.MM[self.nstates*i:self.nstates*(i+1),:] = torch.linalg.matrix_power(self.A,i+1)
            for j in range(self.hzn):
                if i-j >= 0:
                    self.CC[self.nstates*i:self.nstates*(i+1),self.ninputs*j:self.ninputs*(j+1)] = torch.matmul(torch.linalg.matrix_power(self.A,(i-j)),self.B)
                else:
                    self.CC[self.nstates*i:self.nstates*(i+1),self.ninputs*j:self.ninputs*(j+1)] = self.Bz

    def OSQP_setup(self, y, yr):
        """
        y represents the current state vector seen by the MPC
        
        yr represents the reference state vector that the MPC wishes to achieve
        """

        # stack yr for the prediction horizon
        yr = torch.tile(yr, (self.hzn,1))

        # calculate and assign terminal weighting matrix
        self.QQ[-self.nstates:,-self.nstates:] = torch.tensor(scipy.linalg.solve_discrete_lyapunov((self.A + self.B @ self.K).T, self.Q + self.K.T @ self.R @ self.K),device=self.device,dtype=self.dtype)

        
        OSQP_P = 2 * (self.CC.T @ self.QQ @ self.CC + self.RR)
        OSQP_q = -2 * ((yr - self.MM @ y).T @ self.QQ @ self.CC).T 

        # calculate the command rate saturation limits vector
   
        act_states = y[7:10]


        u0_rate_constr_lower = act_states + self.rdot_lb * self.dt
        u0_rate_constr_upper = act_states + self.rdot_ub * self.dt

        cmd_rate_constr_lower = torch.cat((u0_rate_constr_lower,torch.tile(self.rdot_lb,(self.hzn-1,1))))
        cmd_rate_constr_upper = torch.cat((u0_rate_constr_upper,torch.tile(self.rdot_ub,(self.hzn-1,1))))

        # calculate the command rate saturation input sequence matrix
        cmd_rate_constr_mat = torch.eye(self.ninputs*self.hzn)
        for i in range(self.ninputs*self.hzn):
            if i >= self.ninputs:
                cmd_rate_constr_mat[i,i-self.ninputs] = -1

        # assemble the complete matrices to send to OSQP
        OSQP_A = torch.cat((self.CC, self.cmd_constr_mat, cmd_rate_constr_mat), axis=0)
        OSQP_l = torch.cat((self.state_constr_lower, self.cmd_constr_lower, cmd_rate_constr_lower))
        OSQP_u = torch.cat((self.state_constr_upper, self.cmd_constr_upper, cmd_rate_constr_upper))
       
        OSQP_P = scipy.sparse.csc_matrix(OSQP_P)
        OSQP_A = scipy.sparse.csc_matrix(OSQP_A)
        OSQP_q, OSQP_l, OSQP_u = OSQP_q.numpy(), OSQP_l.numpy(), OSQP_u.numpy()

        return OSQP_P, OSQP_A, OSQP_q, OSQP_l, OSQP_u

    def forward(self, y, yr):

        OSQP_P, OSQP_A, OSQP_q, OSQP_l, OSQP_u = self.OSQP_setup(y, yr)
        
        opt = osqp.OSQP()
        opt.setup(P=OSQP_P, q=OSQP_q, A=OSQP_A, l=OSQP_l, u=OSQP_u, verbose=False)
        res = opt.solve()
        ro = res.x[0:self.ninputs]
        return torch.tensor(ro, device=self.device, dtype=self.dtype).unsqueeze(1)

class Trim(nn.Module):
    def __init__(self, device, dtype):
        super().__init__()
        self.device = device
        self.dtype = dtype

    def __call__(self, h_t, v_t, x, u):
        return self.trim(h_t, v_t, x, u)

    def trim(self, h_t, v_t, x, u):

        """ Function for trimming the aircraft in straight and level flight. The objective
        function is built to be the same as that of the MATLAB version of the Nguyen
        simulation.

        Args:
            h_t:
                altitude above sea level in ft, float
            v_t:
                airspeed in ft/s, float

        Returns:
            x_trim:
                trim state vector, 1D numpy array
            opt:
                scipy.optimize.minimize output information
        """

        print('Trim state being calculated...')

        # initial guesses
        thrust = 5000           # thrust, lbs
        elevator = -0.09        # elevator, degrees
        alpha = 8.49            # AOA, degrees
        rudder = -0.01          # rudder angle, degrees
        aileron = 0.01          # aileron, degrees

        UX0 = [thrust, elevator, alpha, rudder, aileron]

        #################### convert everything to numpy ####################

        #x_np = x.values.numpy()
        #u_np = u.values.numpy()

        opt = minimize(obj_func, UX0, args=((h_t, v_t, x, u)), method='Nelder-Mead',tol=1e-10,options={'maxiter':5e+04})

        P3_t, dstab_t, da_t, dr_t, alpha_t  = opt.x

        rho0 = 2.377e-3
        tfac = 1 - 0.703e-5*h_t

        temp = 519*tfac
        if h_t >= 35000:
            temp = 390

        rho = rho0*tfac**4.14
        qbar = 0.5*rho*v_t**2
        ps = 1715*rho*temp

        dlef = 1.38*alpha_t*180/pi - 9.05*qbar/ps + 1.45

        #################### back to torch tensors #####################
        x_trim = torch.tensor([0, 0, h_t, 0, alpha_t, 0, v_t, alpha_t, 0, 0, 0, 0, P3_t, dstab_t, da_t, dr_t, dlef, -alpha_t*180/pi])

        print('SUCCESS: Trim complete')
        return x_trim, opt


    def obj_func(self, UX0, h_t, v_t, x, u):

        V = v_t
        h = h_t
        P3, dh, da, dr, alpha = UX0
        npos = 0
        epos = 0
        phi = 0
        psi = 0
        beta = 0
        p = 0
        q = 0
        r = 0
        rho0 = 2.377e-3
        tfac = 1 - 0.703e-5*h
        temp = 519*tfac
        if h >= 35000:
            temp = 390
        rho = rho0*tfac**4.14
        qbar = 0.5*rho*V**2
        ps = 1715*rho*temp
        dlef = 1.38*alpha*180/pi - 9.05*qbar/ps + 1.45
        x.values = torch.tensor([npos, epos, h, phi, alpha, psi, V, alpha, beta, p, q, r, P3, dh, da, dr, dlef, -alpha*180/pi])

        # thrust limits
        x.values[12] = torch.clip(x.values[12], u.lower_cmd_bound[0], u.upper_cmd_bound[0])
        # elevator limits
        x.values[13] = torch.clip(x.values[13], u.lower_cmd_bound[1], u.upper_cmd_bound[1])
        # aileron limits
        x.values[14] = torch.clip(x.values[14], u.lower_cmd_bound[2], u.upper_cmd_bound[2])
        # rudder limits
        x.values[15] = torch.clip(x.values[15], u.lower_cmd_bound[3], u.upper_cmd_bound[3])
        # alpha limits
        x.values[7] = torch.clip(x.values[7], x.lower_bound[7]*pi/180, x.upper_bound[7]*pi/180)

        u = x.values[12:16]
        xdot,_,_ = calc_xdot(x.values, u)
        xdot = xdot.reshape([18,1])

        phi_w = 10
        theta_w = 10
        psi_w = 10

        weight = torch.tensor([0, 0, 5, phi_w, theta_w, psi_w, 2, 10, 10, 10, 10, 10], dtype=torch.float32).reshape([1,12])
        cost = torch.mm(weight,xdot[0:12]**2)

