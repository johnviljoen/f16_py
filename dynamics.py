import torch
import torch.nn as nn

from parameters import x_lb, x_ub, u_lb, u_ub, pi
from tables import c_lookup
from tables import py_lookup

class Nlplant(nn.Module):
    def __init__(self, device, dtype, lookup_type):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.acts = Actuators(device, dtype)
        self.eom = EoM(device, dtype, lookup_type=lookup_type)

    def __call__(self, x, u):
        return self.forward(x, u)

    def forward(self, x, u):
        """
        args:
            x:
                {xe, ye, h, phi, theta, psi, V, alpha, beta, p, q, r, T, dh, da, dr, lf2, lf1}
            u:
                {T, dh, da, dr}

        returns:
            xdot:
                time derivates of x, in same order
            accelerations:
                {anx_cg, any_cg, anz_cg}
            atmospherics:
                {mach, qbar, ps}
        """
        # initialise variables
        actuator_xdot = torch.zeros([6,1], device=self.device, dtype=self.dtype)
        # Thrust Model
        actuator_xdot[0] = self.acts.upd_thrust(u[0], x[12])
        # Dstab Model
        actuator_xdot[1] = self.acts.upd_dstab(u[1], x[13])
        # aileron model
        actuator_xdot[2] = self.acts.upd_ail(u[2], x[14])
        # rudder model
        actuator_xdot[3] = self.acts.upd_rud(u[3], x[15])
        # leading edge flap model
        actuator_xdot[5], actuator_xdot[4] = self.acts.upd_lef(x[2], x[6], x[7], x[17], x[16])
        # run nlplant for xdot
        xdot, accelerations, atmospherics = self.eom(x)
        # assign actuator xdots
        xdot[12:18] = actuator_xdot
        return xdot, accelerations, atmospherics

class Atmos(nn.Module):
    def __init__(self, device, dtype):
        super().__init__()
        self.device = device
        self.dtype = dtype

    def __call__(self, alt, vt):
        rho0 = 2.377e-3
        
        tfac =1 - .703e-5*(alt)
        temp = 519.0*tfac
        if alt >= 35000.0: temp=390

        rho=rho0*torch.pow(tfac,4.14)
        mach = vt/torch.sqrt(1.4*1716.3*temp)
        qbar = .5*rho*torch.pow(vt,2)
        ps   = 1715.0*rho*temp

        if ps == 0: ps = 1715
        
        return mach, qbar, ps

class Actuators(nn.Module):
    def __init__(self, device, dtype):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.atmos = Atmos(device, dtype)

    def upd_lef(self, h, V, alpha, lef_state_1, lef_state_2):
        
        coeff = self.atmos(h, V)
        atmos_out = coeff[1]/coeff[2] * 9.05
        alpha_deg = alpha*180/pi
        
        LF_err = alpha_deg - (lef_state_1 + (2 * alpha_deg))
        #lef_state_1 += LF_err*7.25*time_step
        LF_out = (lef_state_1 + (2 * alpha_deg)) * 1.38
        
        lef_cmd = LF_out + 1.45 - atmos_out
        
        # command saturation
        lef_cmd = torch.clip(lef_cmd,x_lb[16],x_ub[16])
        # rate saturation
        lef_err = torch.clip((1/0.136) * (lef_cmd - lef_state_2),-25,25)
        
        return LF_err*7.25, lef_err

    def upd_thrust(self, T_cmd, T_state):
        # command saturation
        T_cmd = torch.clip(T_cmd,u_lb[0],u_ub[0])
        # rate saturation
        return torch.clip(T_cmd - T_state, -10000, 10000)

    def upd_dstab(self, dstab_cmd, dstab_state):
        # command saturation
        dstab_cmd = torch.clip(dstab_cmd,u_lb[1],u_ub[1])
        # rate saturation
        return torch.clip(20.2*(dstab_cmd - dstab_state), -60, 60)

    def upd_ail(self, ail_cmd, ail_state):
        # command saturation
        ail_cmd = torch.clip(ail_cmd,u_lb[2],u_ub[2])
        # rate saturation
        return torch.clip(20.2*(ail_cmd - ail_state), -80, 80)

    def upd_rud(self, rud_cmd, rud_state):
        # command saturation
        rud_cmd = torch.clip(rud_cmd,u_lb[3],u_ub[3])
        # rate saturation
        return torch.clip(20.2*(rud_cmd - rud_state), -120, 120)

class EoM(nn.Module):
    def __init__(self, device, dtype, lookup_type='C'):
        super().__init__()

        self.device = device
        self.dtype = dtype
        self.lookup_type = lookup_type

        self.atmos = Atmos(self.device, self.dtype)

    def __call__(self, x):
        return self.forward(x)


    def forward(self, xu):
        g    = 32.17                            # gravity, ft/s^2
        m    = 636.94                           # mass, slugs
        B    = 30.0                             # span, ft
        S    = 300.0                            # planform area, ft^2
        cbar = 11.32                            # mean aero chord, ft
        xcgr = 0.35                             # reference center of gravity as a fraction of cbar
        xcg  = 0.25                             # center of gravity as a fraction of cbar
        
        Heng = 0.0                              # turbine momentum along roll axis
        pi   = torch.acos(torch.tensor(-1, device=self.device, dtype=self.dtype))
        r2d  = torch.rad2deg(torch.tensor(1, device=self.device, dtype=self.dtype))   # radians to degrees
        
        # NasaData translated via eq. 2.4-6 on pg 80 of Stevens and Lewis
        
        Jy  = 55814.0                           # slug-ft^2
        Jxz = 982.0                             # slug-ft^2   
        Jz  = 63100.0                           # slug-ft^2
        Jx  = 9496.0                            # slug-ft^2
        
        # instantiate xdot
        xdot = torch.zeros(xu.shape, device=self.device, dtype=self.dtype)
        
        # In[states]
        
        npos  = xu[0]   # north position
        epos  = xu[1]   # east position
        alt   = xu[2]   # altitude
        phi   = xu[3]   # orientation angles in rad
        theta = xu[4]
        psi   = xu[5]
        
        vt    = xu[6]     # total velocity
        alpha = xu[7] * r2d # angle of attack in degrees
        beta  = xu[8] * r2d # sideslip angle in degrees
        P     = xu[9]    # Roll Rate --- rolling  moment is Lbar
        Q     = xu[10]    # Pitch Rate--- pitching moment is M
        R     = xu[11]    # Yaw Rate  --- yawing   moment is N
        
        sin = torch.sin
        cos = torch.cos
        tan = torch.tan
        
        sa    = sin(xu[7]) # sin(alpha)
        ca    = cos(xu[7]) # cos(alpha)
        sb    = sin(xu[8]) # sin(beta)
        cb    = cos(xu[8]) # cos(beta)
        tb    = tan(xu[8]) # tan(beta)
        
        st    = sin(theta)
        ct    = cos(theta)
        tt    = tan(theta)
        sphi  = sin(phi)
        cphi  = cos(phi)
        spsi  = sin(psi)
        cpsi  = cos(psi)
        
        if vt < 0.01: vt = 0.01
        
        # In[Control inputs]
        
        T     = xu[12]   # thrust
        el    = xu[13]   # Elevator setting in degrees
        ail   = xu[14]   # Ailerons mex setting in degrees
        rud   = xu[15]   # Rudder setting in degrees
        lef   = xu[16]   # Leading edge flap setting in degrees
        
        # dail  = ail/20.0;   aileron normalized against max angle
        # The aileron was normalized using 20.0 but the NASA report and
        # S&L both have 21.5 deg. as maximum deflection.
        # As a result...
        dail  = ail/21.5
        drud  = rud/30.0  # rudder normalized against max angle
        dlef  = (1 - lef/25.0)  # leading edge flap normalized against max angle
        
        # In[Atmospheric effects]
        mach, qbar, ps = self.atmos(alt, vt)
        
        # In[Navigation equations]
        U = vt*ca*cb  # directional velocities
        V = vt*sb
        W = vt*sa*cb
        
        # nposdot
        xdot[0] = U*(ct*cpsi) + \
                    V*(sphi*cpsi*st - cphi*spsi) + \
                    W*(cphi*st*cpsi + sphi*spsi)
                    
        # eposdot
        xdot[1] = U*(ct*spsi) + \
                    V*(sphi*spsi*st + cphi*cpsi) + \
                    W*(cphi*st*spsi - sphi*cpsi)
                    
        # altdot
        xdot[2] = U*st - V*(sphi*ct) - W*(cphi*ct)
        
        # In[Kinematic equations]
        
        # phidot
        xdot[3] = P + tt*(Q*sphi + R*cphi)


        # theta dot
        xdot[4] = Q*cphi - R*sphi

        # psidot
        xdot[5] = (Q*sphi + R*cphi)/ct
        
        # In[Table Lookup]
        inp = torch.tensor([alpha, beta, el], device=self.device, dtype=self.dtype)

        if self.lookup_type == 'Py':
            # hifi_C
            Cx = py_lookup.interp_3d(inp, 'Cx')
            Cz = py_lookup.interp_3d(inp, 'Cz')
            Cm = py_lookup.interp_3d(inp, 'Cm')
            Cy = py_lookup.interp_2d(inp[0:2], 'Cy')
            Cn = py_lookup.interp_3d(inp, 'Cn')
            Cl = py_lookup.interp_3d(inp, 'Cl')
       
            # hifi_damping
            Cxq = py_lookup.interp_1d(inp[0:1], 'CXq')
            Cyr = py_lookup.interp_1d(inp[0:1], 'CYr')
            Cyp = py_lookup.interp_1d(inp[0:1], 'CYp')
            Czq = py_lookup.interp_1d(inp[0:1], 'CZq')
            Clr = py_lookup.interp_1d(inp[0:1], 'CLr')
            Clp = py_lookup.interp_1d(inp[0:1], 'CLp')
            Cmq = py_lookup.interp_1d(inp[0:1], 'CMq')
            Cnr = py_lookup.interp_1d(inp[0:1], 'CNr')
            Cnp = py_lookup.interp_1d(inp[0:1], 'CNp')

            # hifi_C_lef
            temp = torch.cat([inp[0:2], torch.tensor([0])])
            delta_Cx_lef = py_lookup.interp_2d(inp[0:2], 'Cx_lef') - py_lookup.interp_3d(temp, 'Cx')
            delta_Cz_lef = py_lookup.interp_2d(inp[0:2], 'Cz_lef') - py_lookup.interp_3d(temp, 'Cz')
            delta_Cm_lef = py_lookup.interp_2d(inp[0:2], 'Cm_lef') - py_lookup.interp_3d(temp, 'Cm')
            delta_Cy_lef = py_lookup.interp_2d(inp[0:2], 'Cy_lef') - py_lookup.interp_2d(inp[0:2], 'Cy')
            delta_Cn_lef = py_lookup.interp_2d(inp[0:2], 'Cn_lef') - py_lookup.interp_3d(temp, 'Cn')
            delta_Cl_lef = py_lookup.interp_2d(inp[0:2], 'Cl_lef') - py_lookup.interp_3d(temp, 'Cl')

            # hifi_damping_lef
            delta_Cxq_lef = py_lookup.interp_1d(inp[0:1], 'delta_CXq_lef')
            delta_Cyr_lef = py_lookup.interp_1d(inp[0:1], 'delta_CYr_lef')
            delta_Cyp_lef = py_lookup.interp_1d(inp[0:1], 'delta_CYp_lef')
            delta_Czq_lef = py_lookup.interp_1d(inp[0:1], 'delta_CZq_lef') # this being unused is not an error, it is as the original C was written, you can delete if you like.
            delta_Clr_lef = py_lookup.interp_1d(inp[0:1], 'delta_CLr_lef')
            delta_Clp_lef = py_lookup.interp_1d(inp[0:1], 'delta_CLp_lef')
            delta_Cmq_lef = py_lookup.interp_1d(inp[0:1], 'delta_CMq_lef')
            delta_Cnr_lef = py_lookup.interp_1d(inp[0:1], 'delta_CNr_lef')
            delta_Cnp_lef = py_lookup.interp_1d(inp[0:1], 'delta_CNp_lef')

            # hifi_rudder
            delta_Cy_r30 = py_lookup.interp_2d(inp[0:2], 'Cy_r30') - py_lookup.interp_2d(inp[0:2], 'Cy')
            delta_Cn_r30 = py_lookup.interp_2d(inp[0:2], 'Cn_r30') - py_lookup.interp_3d(temp, 'Cn')
            delta_Cl_r30 = py_lookup.interp_2d(inp[0:2], 'Cl_r30') - py_lookup.interp_3d(temp, 'Cl')

            # hifi_ailerons
            delta_Cy_a20     = py_lookup.interp_2d(inp[0:2], 'Cy_a20') - py_lookup.interp_2d(inp[0:2], 'Cy')
            delta_Cy_a20_lef = py_lookup.interp_2d(inp[0:2], 'Cy_a20_lef') - py_lookup.interp_2d(inp[0:2], 'Cy_lef')
            delta_Cn_a20     = py_lookup.interp_2d(inp[0:2], 'Cn_a20') - py_lookup.interp_3d(temp, 'Cn')
            delta_Cn_a20_lef = py_lookup.interp_2d(inp[0:2], 'Cn_a20_lef') - py_lookup.interp_2d(inp[0:2], 'Cn_lef')
            delta_Cl_a20     = py_lookup.interp_2d(inp[0:2], 'Cl_a20') - py_lookup.interp_3d(temp, 'Cl')
            delta_Cl_a20_lef = py_lookup.interp_2d(inp[0:2], 'Cl_a20_lef') - py_lookup.interp_2d(inp[0:2], 'Cl_lef')
            
            # hifi_other_coeffs
            delta_Cnbeta = py_lookup.interp_1d(inp[0:1], 'delta_CNbeta')
            delta_Clbeta = py_lookup.interp_1d(inp[0:1], 'delta_CLbeta')
            delta_Cm = py_lookup.interp_1d(inp[0:1], 'delta_Cm')
            eta_el = py_lookup.interp_1d(inp[2:3], 'eta_el')
            delta_Cm_ds = 0.

        
        if self.lookup_type == 'NN':
            
            # hifi_C
            Cx = NN['Cx'](inp)
            Cz = NN['Cz'](inp)
            Cm = NN['Cm'](inp)
            Cy = NN['Cy'](inp)
            Cn = NN['Cn'](inp)
            Cl = NN['Cl'](inp)
       
            # hifi_damping
            Cxq = NN['Cxq'](inp[0:1])
            Cyr = NN['Cyr'](inp[0:1])
            Cyp = NN['Cyp'](inp[0:1])
            Czq = NN['Czq'](inp[0:1])
            Clr = NN['Clr'](inp[0:1])
            Clp = NN['Clp'](inp[0:1])
            Cmq = NN['Cmq'](inp[0:1])
            Cnr = NN['Cnr'](inp[0:1])
            Cnp = NN['Cnp'](inp[0:1])

            # hifi_C_lef
            delta_Cx_lef = NN['delta_Cx_lef'](inp[0:2])
            delta_Cz_lef = NN['delta_Cz_lef'](inp[0:2])
            delta_Cm_lef = NN['delta_Cm_lef'](inp[0:2])
            delta_Cy_lef = NN['delta_Cy_lef'](inp[0:2])
            delta_Cn_lef = NN['delta_Cn_lef'](inp[0:2])
            delta_Cl_lef = NN['delta_Cl_lef'](inp[0:2])

            # hifi_damping_lef
            delta_Cxq_lef = NN['delta_Cxq_lef'](inp[0:1])
            delta_Cyr_lef = NN['delta_Cyr_lef'](inp[0:1])
            delta_Cyp_lef = NN['delta_Cyp_lef'](inp[0:1])
            delta_Czq_lef = NN['delta_Czq_lef'](inp[0:1])
            delta_Clr_lef = NN['delta_Clr_lef'](inp[0:1])
            delta_Clp_lef = NN['delta_Clp_lef'](inp[0:1])
            delta_Cmq_lef = NN['delta_Cmq_lef'](inp[0:1])
            delta_Cnr_lef = NN['delta_Cnr_lef'](inp[0:1])
            delta_Cnp_lef = NN['delta_Cnp_lef'](inp[0:1])

            # hifi_rudder
            delta_Cy_r30 = NN['delta_Cy_r30'](inp[0:2])
            delta_Cn_r30 = NN['delta_Cn_r30'](inp[0:2])
            delta_Cl_r30 = NN['delta_Cl_r30'](inp[0:2])

            # hifi_ailerons
            delta_Cy_a20     = NN['delta_Cy_a20'](inp[0:2])
            delta_Cy_a20_lef = NN['delta_Cy_a20_lef'](inp[0:2])
            delta_Cn_a20     = NN['delta_Cn_a20'](inp[0:2])
            delta_Cn_a20_lef = NN['delta_Cn_a20_lef'](inp[0:2])
            delta_Cl_a20     = NN['delta_Cl_a20'](inp[0:2])
            delta_Cl_a20_lef = NN['delta_Cl_a20_lef'](inp[0:2])

            # hifi_other_coeffs
            delta_Cnbeta = NN['delta_Cnbeta'](inp[::2])
            delta_Clbeta = NN['delta_Clbeta'](inp[::2])
            delta_Cm = NN['delta_Cm'](inp[::2])
            eta_el = NN['eta_el'](inp[::2])
            delta_Cm_ds = 0.


        elif self.lookup_type == 'C':
            
            
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
        
              
        # In[compute Cx_tot, Cz_tot, Cm_tot, Cy_tot, Cn_tot, and Cl_tot]
            # (as on NASA report p37-40)

        # Cx_tot
        dXdQ = (cbar/(2*vt))*(Cxq + delta_Cxq_lef*dlef)

        Cx_tot = Cx + delta_Cx_lef*dlef + dXdQ*Q

        # Cz_tot
        dZdQ = (cbar/(2*vt))*(Czq + delta_Cz_lef*dlef)

        Cz_tot = Cz + delta_Cz_lef*dlef + dZdQ*Q

        # Cm_tot
        dMdQ = (cbar/(2*vt))*(Cmq + delta_Cmq_lef*dlef)

        Cm_tot = Cm*eta_el + Cz_tot*(xcgr-xcg) + delta_Cm_lef*dlef + dMdQ*Q + \
            delta_Cm + delta_Cm_ds

        # Cy_tot
        dYdail = delta_Cy_a20 + delta_Cy_a20_lef*dlef

        dYdR = (B/(2*vt))*(Cyr + delta_Cyr_lef*dlef)

        dYdP = (B/(2*vt))*(Cyp + delta_Cyp_lef*dlef)

        Cy_tot = Cy + delta_Cy_lef*dlef + dYdail*dail + delta_Cy_r30*drud + \
            dYdR*R + dYdP*P

        # Cn_tot
        dNdail = delta_Cn_a20 + delta_Cn_a20_lef*dlef

        dNdR = (B/(2*vt))*(Cnr + delta_Cnr_lef*dlef)

        dNdP = (B/(2*vt))*(Cnp + delta_Cnp_lef*dlef)

        Cn_tot = Cn + delta_Cn_lef*dlef - Cy_tot*(xcgr-xcg)*(cbar/B) + \
            dNdail*dail + delta_Cn_r30*drud + dNdR*R + dNdP*P + \
                delta_Cnbeta*beta

        # Cl_tot
        dLdail = delta_Cl_a20 + delta_Cl_a20_lef*dlef

        dLdR = (B/(2*vt))*(Clr + delta_Clr_lef*dlef)

        dLdP = (B/(2*vt))*(Clp + delta_Clp_lef*dlef)

        Cl_tot = Cl + delta_Cl_lef*dlef + dLdail*dail + delta_Cl_r30*drud + \
            dLdR*R + dLdP*P + delta_Clbeta*beta

        # In[compute Udot,Vdot, Wdot,(as on NASA report p36)]
        
        Udot = R*V - Q*W - g*st + qbar*S*Cx_tot/m + T/m

        Vdot = P*W - R*U + g*ct*sphi + qbar*S*Cy_tot/m
        
        Wdot = Q*U - P*V + g*ct*cphi + qbar*S*Cz_tot/m
        
        # In[vt_dot equation (from S&L, p82)]
        
        xdot[6] = (U*Udot + V*Vdot + W*Wdot)/vt
        
        # In[alpha_dot equation]
        
        xdot[7] = (U*Wdot - W*Udot)/(U*U + W*W)
        
        # In[beta_dot equation]
        
        xdot[8] = (Vdot*vt - V*xdot[6])/(vt*vt*cb)
        
        # In[compute Pdot, Qdot, and Rdot (as in Stevens and Lewis p32)]
        
        L_tot = Cl_tot*qbar*S*B         # get moments from coefficients
        M_tot = Cm_tot*qbar*S*cbar 
        N_tot = Cn_tot*qbar*S*B
        
        denom = Jx*Jz - Jxz*Jxz
        
        # In[Pdot]
        
        xdot[9] =  (Jz*L_tot + Jxz*N_tot - (Jz*(Jz-Jy)+Jxz*Jxz)*Q*R + \
                    Jxz*(Jx-Jy+Jz)*P*Q + Jxz*Q*Heng)/denom
            
        # In[Qdot]
        
        xdot[10] = (M_tot + (Jz-Jx)*P*R - Jxz*(P*P-R*R) - R*Heng)/Jy

        # In[Rdot]
        
        xdot[11] = (Jx*N_tot + Jxz*L_tot + (Jx*(Jx-Jy)+Jxz*Jxz)*P*Q - \
                    Jxz*(Jx-Jy+Jz)*Q*R +  Jx*Q*Heng)/denom
            
        # In[Create accelerations anx_cg, any_cg, anz_cg as outputs]
        
        def accels(state, xdot):
            
            grav = 32.174
            
            sina = sin(state[7])
            cosa = cos(state[7])
            sinb = sin(state[8])
            cosb = cos(state[8])
            vel_u = state[6]*cosb*cosa
            vel_v = state[6]*sinb
            vel_w = state[6]*cosb*sina
            u_dot =          cosb*cosa*xdot[6] \
                  - state[6]*sinb*cosa*xdot[8] \
                  - state[6]*cosb*sina*xdot[7]
            v_dot =          sinb*xdot[6] \
                  + state[6]*cosb*xdot[8]
            w_dot =          cosb*sina*xdot[6] \
                  - state[6]*sinb*sina*xdot[8] \
                  + state[6]*cosb*cosa*xdot[7]
            nx_cg = 1.0/grav*(u_dot + state[10]*vel_w - state[11]*vel_v) \
                  + sin(state[4])
            ny_cg = 1.0/grav*(v_dot + state[11]*vel_u - state[9]*vel_w) \
                  - cos(state[4])*sin(state[3])
            nz_cg = -1.0/grav*(w_dot + state[9]*vel_v - state[10]*vel_u) \
                  + cos(state[4])*cos(state[3])
                  
            return nx_cg, ny_cg, nz_cg
        
        # anx_cg, any_cg, anz_cg = accels(xu, xdot)

        accelerations = accels(xu, xdot)
        # not the primary xdot values 
        #xdot[12] = anx_cg
        #xdot[13] = any_cg
        #xdot[14] = anz_cg
        #xdot[15] = mach
        #xdot[16] = qbar
        #xdot[17] = ps

        atmospherics = torch.tensor([mach, qbar, ps]) 
        
        return xdot, accelerations, atmospherics

