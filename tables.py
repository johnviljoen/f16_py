"""
This script wraps the C shared library .so file, and implements a python lookup class

NOTE: This script is sensitive to the directory it is run from, the hifi_F16_AeroData.c
file has been compiled to be called from the aerodata directory below this file.

If this script is run from a different directory a segmentation fault WILL occur.

NOTE: This script could be further accelerated by retaining prior pointers
to prior xdots from prior lookups, but this has not yet been implemented.
"""

import torch
import ctypes
import os
import numpy as np
import sys
from parameters import c2f, f2c, aerodata_path

dtype = torch.double
tables = ctypes.CDLL(aerodata_path + "/hifi_F16_AeroData.so")

class C_lookup():

    def __init__(self):
        pass

    def hifi_C(self, inp):
        

        retVal = np.zeros(6)
        retVal_pointer = ctypes.c_void_p(retVal.ctypes.data)
        
        alpha_compat = ctypes.c_double(float(inp[0].detach().cpu().numpy()))
        beta_compat = ctypes.c_double(float(inp[1].detach().cpu().numpy()))
        el_compat = ctypes.c_double(float(inp[2].detach().cpu().numpy()))


        tables.hifi_C(alpha_compat, beta_compat, el_compat, retVal_pointer)
        
        return torch.tensor(retVal) # Cx, Cz, Cm, Cy, Cn, Cl

    def hifi_damping(self, inp):
        # this is the one that contains Clr at index 4 
        retVal = np.zeros(9)
        retVal_pointer = ctypes.c_void_p(retVal.ctypes.data)
        
        alpha_compat = ctypes.c_double(float(inp[0].detach().cpu().numpy()))
        
        tables.hifi_damping(alpha_compat, retVal_pointer)


        return torch.tensor(retVal, dtype=dtype)

    def hifi_C_lef(self, inp):
        
        ''' This table only accepts alpha up to 45 '''
        inp[0] = torch.clip(inp[0], min=-20., max=45.)
        
        retVal = np.zeros(6)
        retVal_pointer = ctypes.c_void_p(retVal.ctypes.data)
        
        alpha_compat = ctypes.c_double(float(inp[0].detach().cpu().numpy()))
        beta_compat = ctypes.c_double(float(inp[1].detach().cpu().numpy()))
        
        tables.hifi_C_lef(alpha_compat, beta_compat, retVal_pointer)
        
        return torch.tensor(retVal, dtype=dtype)
        
    def hifi_damping_lef(self, inp):
        ''' 
        This table only accepts alpha up to 45
            delta_Cxq_lef
            delta_Cyr_lef
            delta_Cyp_lef
            delta_Czq_lef
            delta_Clr_lef
            delta_Clp_lef
            delta_Cmq_lef
            delta_Cnr_lef
            delta_Cnp_lef
        '''
        inp[0] = torch.clip(inp[0], min=-20., max=45.)
       
        
        retVal = np.zeros(9)
        retVal_pointer = ctypes.c_void_p(retVal.ctypes.data)
        
        alpha_compat = ctypes.c_double(float(inp[0].detach().cpu().numpy()))
        
        tables.hifi_damping_lef(alpha_compat, retVal_pointer)
        
        return torch.tensor(retVal, dtype=dtype)

    def hifi_rudder(self, inp):
        
        retVal = np.zeros(3)
        retVal_pointer = ctypes.c_void_p(retVal.ctypes.data)
        
        alpha_compat = ctypes.c_double(float(inp[0].detach().cpu().numpy()))
        beta_compat = ctypes.c_double(float(inp[1].detach().cpu().numpy()))
        
        tables.hifi_rudder(alpha_compat, beta_compat, retVal_pointer)
        
        return torch.tensor(retVal, dtype=dtype)

    def hifi_ailerons(self, inp):
        
        ''' This table only accepts alpha up to 45 '''
        inp[0] = torch.clip(inp[0], min=-20., max=45.)
        
        retVal = np.zeros(6)
        retVal_pointer = ctypes.c_void_p(retVal.ctypes.data)
        
        alpha_compat = ctypes.c_double(float(inp[0].detach().cpu().numpy()))
        beta_compat = ctypes.c_double(float(inp[1].detach().cpu().numpy()))
        
        tables.hifi_ailerons(alpha_compat, beta_compat, retVal_pointer)
        
        return torch.tensor(retVal, dtype=dtype)

    def hifi_other_coeffs(self, inp):
        
        '''expects an input of alpha, el'''
        
        retVal = np.zeros(5)
        retVal_pointer = ctypes.c_void_p(retVal.ctypes.data)
        
        alpha_compat = ctypes.c_double(float(inp[0].detach().cpu().numpy()))
        el_compat = ctypes.c_double(float(inp[1].detach().cpu().numpy()))
        
        tables.hifi_other_coeffs(alpha_compat, el_compat, retVal_pointer)
        
        retVal[4] = 0 # ignore deep-stall regime, delta_Cm_ds = 0
        
        return torch.tensor(retVal, dtype=dtype)

c_lookup = C_lookup()

class Py_parse():

    """
    This class parses the .dat files into a dictionary of tensors of the correct dimensions,
    which can be accessed by the key of the filename from which the lookup values were read.
    """

    def __init__(self):
        # indices lookup
        self.axes = {}
        self.axes['ALPHA1'] = torch.tensor(self.read_file("aerodata/ALPHA1.dat"))
        self.axes['ALPHA2'] = torch.tensor(self.read_file("aerodata/ALPHA2.dat"))
        self.axes['BETA1'] = torch.tensor(self.read_file("aerodata/BETA1.dat"))
        self.axes['DH1'] = torch.tensor(self.read_file("aerodata/DH1.dat"))
        self.axes['DH2'] = torch.tensor(self.read_file("aerodata/DH2.dat"))
        
        # tables store the actual data, points are the alpha, beta, dh axes 
        self.tables = {}
        self.points = {}
        self.ndinfo = {}
        for file in os.listdir("aerodata"):
            if file == 'CM1020_ALPHA1_103.dat':
                continue
            alpha_len = None
            beta_len = None
            dh_len = None
            alpha_fi = None
            beta_fi = None
            dh_fi = None
            if "_ALPHA1" in file:
                alpha_len = len(self.axes['ALPHA1'])
                alpha_fi = 'ALPHA1'
            if "_ALPHA2" in file:
                alpha_len = len(self.axes['ALPHA2'])
                alpha_fi = 'ALPHA2'
            if "_BETA1" in file:
                beta_len = len(self.axes['BETA1'])
                beta_fi = 'BETA1'
            if "_DH1" in file:
                dh_len = len(self.axes['DH1'])
                dh_fi = 'DH1'
            if "_DH2" in file:
                dh_len = len(self.axes['DH2'])
                dh_fi = 'DH2'

            temp = [alpha_len, beta_len, dh_len]
            dims = [i for i in temp if i is not None]
            
            self.ndinfo[file] = {
                'alpha_fi': alpha_fi,
                'beta_fi': beta_fi,
                'dh_fi': dh_fi
            }

            # 1D tables
            if len(dims) == 1:
                self.tables[file] = torch.tensor(self.read_file(f"aerodata/{file}"))
                if file == "ETA_DH1_brett.dat":
                    self.points[file] = (self.axes[self.ndinfo[file]['dh_fi']]) 
                else:
                    self.points[file] = (self.axes[self.ndinfo[file]['alpha_fi']]) 
                    
         
            # 2D tables
            elif len(dims) == 2:
                self.tables[file] = torch.tensor(np.array(self.read_file(f"aerodata/{file}")).reshape([dims[0],dims[1]],order='F'))
                self.points[file] = (
                    self.axes[self.ndinfo[file]['alpha_fi']], 
                    self.axes[self.ndinfo[file]['beta_fi']])

                 
            # 3D tables
            elif len(dims) == 3:
                self.tables[file] = torch.tensor(np.array(self.read_file(f"aerodata/{file}")).reshape([dims[0],dims[1],dims[2]],order='F'))
                self.points[file] = (
                    self.axes[self.ndinfo[file]['alpha_fi']], 
                    self.axes[self.ndinfo[file]['beta_fi']],
                    self.axes[self.ndinfo[file]['dh_fi']]) 


    def read_file(self, path):
        """
        Utility for reading in the .dat files that comprise all of the aerodata
        """
        
        # get the indices of the various tables first
        with open(path) as f:
            lines = f.readlines()
        temp = lines[0][:-1].split()
        line = [float(i) for i in temp]
        return line


py_parse = Py_parse()

class Py_lookup():

    """
    This class takes the parsed data and does some calculations on it to form the final LUT values,
    which are also then interpolated in this class.
    """

    def __init__(self):
        self.parse = Py_parse()

        ## find lookup table for requested coefficient
        #if self.coeff in ['Cx', 'Cz', 'Cm', 'Cy', 'Cn', 'Cl']:
        #    self.table = C.hifi_C
        #    self.table_outputs = ['Cx', 'Cz', 'Cm', 'Cy', 'Cn', 'Cl']
        #elif self.coeff in ['Cxq', 'Cyr', 'Cyp', 'Czq', 'Clr', 'Clp', 'Cmq', 'Cnr', 'Cnp']:
        #    self.table = C.hifi_damping
        #    self.table_outputs = ['Cxq', 'Cyr', 'Cyp', 'Czq', 'Clr', 'Clp', 'Cmq', 'Cnr', 'Cnp']
        #elif self.coeff in ['delta_Cx_lef', 'delta_Cz_lef', 'delta_Cm_lef', 'delta_Cy_lef', 'delta_Cn_lef', 'delta_Cl_lef']:
        #    self.table = C.hifi_C_lef
        #    self.table_outputs = ['delta_Cx_lef', 'delta_Cz_lef', 'delta_Cm_lef', 'delta_Cy_lef', 'delta_Cn_lef', 'delta_Cl_lef']
        #elif self.coeff in ['delta_Cxq_lef', 'delta_Cyr_lef', 'delta_Cyp_lef', 'delta_Czq_lef', 'delta_Clr_lef', 'delta_Clp_lef', 'delta_Cmq_lef', 'delta_Cnr_lef','delta_Cnp_lef']:
        #    self.table = C.hifi_damping_lef
        #    self.table_outputs = ['delta_Cxq_lef', 'delta_Cyr_lef', 'delta_Cyp_lef', 'delta_Czq_lef', 'delta_Clr_lef', 'delta_Clp_lef', 'delta_Cmq_lef', 'delta_Cnr_lef','delta_Cnp_lef']
        #elif self.coeff in ['delta_Cy_r30', 'delta_Cn_r30', 'delta_Cl_r30']:
        #    self.table = C.hifi_rudder
        #    self.table_outputs = ['delta_Cy_r30', 'delta_Cn_r30', 'delta_Cl_r30']
        #elif self.coeff in ['delta_Cy_a20', 'delta_Cy_a20_lef', 'delta_Cn_a20', 'delta_Cn_a20_lef', 'delta_Cl_a20', 'delta_Cl_a20_lef']:
        #    self.table = C.hifi_ailerons
        #    self.table_outputs = ['delta_Cy_a20', 'delta_Cy_a20_lef', 'delta_Cn_a20', 'delta_Cn_a20_lef', 'delta_Cl_a20', 'delta_Cl_a20_lef']
        #elif self.coeff in ['delta_Cnbeta', 'delta_Clbeta', 'delta_Cm', 'eta_el', 'delta_Cm_ds']:
        #    self.table = C.hifi_other_coeffs
        #    self.table_outputs = ['delta_Cnbeta', 'delta_Clbeta', 'delta_Cm', 'eta_el', 'delta_Cm_ds']

        self.hifi_C_coeffs = ['Cx', 'Cz', 'Cm', 'Cy', 'Cn', 'Cl']
        self.hifi_damping_coeffs = ['Cxq', 'Cyr', 'Cyp', 'Czq', 'Clr', 'Clp', 'Cmq', 'Cnr', 'Cnp']
        self.hifi_C_lef_coeffs = ['delta_Cx_lef', 'delta_Cz_lef', 'delta_Cm_lef', 'delta_Cy_lef', 'delta_Cn_lef', 'delta_Cl_lef']
        self.hifi_damping_lef_coeffs = ['delta_Cxq_lef', 'delta_Cyr_lef', 'delta_Cyp_lef', 'delta_Czq_lef', 'delta_Clr_lef', 'delta_Clp_lef', 'delta_Cmq_lef', 'delta_Cnr_lef','delta_Cnp_lef']
        self.hifi_rudder_coeffs = ['delta_Cy_r30', 'delta_Cn_r30', 'delta_Cl_r30']
        self.hifi_ailerons_coeffs = ['delta_Cy_a20', 'delta_Cy_a20_lef', 'delta_Cn_a20', 'delta_Cn_a20_lef', 'delta_Cl_a20', 'delta_Cl_a20_lef']
        self.hifi_other_coeffs_coeffs = ['delta_Cnbeta', 'delta_Clbeta', 'delta_Cm', 'eta_el', 'delta_Cm_ds']

        

    # hifi_C(alpha, beta, el)
    def hifi_C(self):
        return [self.parse.tables[c2f[i]] for i in self.hifi_C_coeffs]

    def hifi_damping(self):
        return [self.parse.tables[c2f[i]] for i in self.hifi_damping_coeffs]

    #def hifi_C_lef(self):
    #    return [self.parse.tables[c2f[i]] for i in self.hifi_C_lef_coeffs]

    #def hifi_

    def get_bounds_1d(self, inp, coeff):
        # can be either alpha or el

        ndinfo = self.parse.ndinfo[c2f[coeff]]
        try:
            assert ndinfo['alpha_fi'] is not None
            alpha_ax = self.parse.axes[ndinfo['alpha_fi']]
            out0 = len([i for i in alpha_ax-inp[0] if i < 0]) - 1
            out1 = out0 + 1

        except:
            assert ndinfo['dh_fi'] is not None
            dh_ax = self.parse.axes[ndinfo['dh_fi']]
            out0 = len([i for i in dh_ax-inp[-1] if i < 0]) - 1
            out1 = out0 + 1
        #except ValueError:
        #    print('1D table bounds called for neither alpha or dh, no other type of table exists')
            



        return out0, out1



    def get_bounds_2d(self, inp, coeff):
        # inp expected in form [alpha, beta]

        ndinfo = self.parse.ndinfo[c2f[coeff]]

        alpha_ax = self.parse.axes[ndinfo['alpha_fi']]
        beta_ax = self.parse.axes[ndinfo['beta_fi']]

        alpha0 = len([i for i in alpha_ax-inp[0] if i < 0]) - 1
        alpha1 = alpha0 + 1

        beta0 = len([i for i in beta_ax-inp[1] if i < 0]) - 1
        beta1 = beta0 + 1


        return alpha0, alpha1, beta0, beta1


    def get_bounds_3d(self, inp, coeff):
        
        # this is for getting the indices of the alpha, beta, el values around the inp
        # table = py_parse.tables[c2f[table_name]]

        # inp expected in form [alpha, beta, el]

        ndinfo = self.parse.ndinfo[c2f[coeff]]

        alpha_ax = self.parse.axes[ndinfo['alpha_fi']]
        beta_ax = self.parse.axes[ndinfo['beta_fi']]
        dh_ax = self.parse.axes[ndinfo['dh_fi']]

        alpha0 = len([i for i in alpha_ax-inp[0] if i < 0]) - 1
        alpha1 = alpha0 + 1

        beta0 = len([i for i in beta_ax-inp[1] if i < 0]) - 1
        beta1 = beta0 + 1
        
        dh0 = len([i for i in dh_ax-inp[2] if i < 0]) - 1
        dh1 = dh0 + 1

        return alpha0, alpha1, beta0, beta1, dh0, dh1
        
    
    def interp_1d(self, inp, coeff):
        # here x0, x1 refer to alpha0, alpha1 usually (one exception where it is dh)
        # and y0, y1 refer to the coefficent values at these values of alpha
        x0_idx, x1_idx = self.get_bounds_1d(inp, coeff)
        if self.parse.ndinfo[c2f[coeff]]['dh_fi'] is None:
            x = inp[0]
            x0 = self.parse.axes[self.parse.ndinfo[c2f[coeff]]['alpha_fi']][x0_idx]
            x1 = self.parse.axes[self.parse.ndinfo[c2f[coeff]]['alpha_fi']][x1_idx]
        else:
            x = inp[-1]
            x0 = self.parse.axes[self.parse.ndinfo[c2f[coeff]]['dh_fi']][x0_idx]
            x1 = self.parse.axes[self.parse.ndinfo[c2f[coeff]]['dh_fi']][x1_idx]

        #x0 = self.parse.axes[c2f[coeff]]
        y0 = self.parse.tables[c2f[coeff]][x0_idx]
        y1 = self.parse.tables[c2f[coeff]][x1_idx]

        return y0*(x1 - x)/(x1 - x0) + y1*(x - x0)/(x1 - x0)

    def interp_2d(self, inp, coeff):
        x0_idx, x1_idx, y0_idx, y1_idx = self.get_bounds_2d(inp, coeff)

        x = inp[0]
        y = inp[1]

        alpha_table = self.parse.axes[self.parse.ndinfo[c2f[coeff]]['alpha_fi']]
        beta_table = self.parse.axes[self.parse.ndinfo[c2f[coeff]]['beta_fi']]
        x0 = alpha_table[x0_idx]
        x1 = alpha_table[x1_idx]
        y0 = beta_table[y0_idx]
        y1 = beta_table[y1_idx]

        Q00 = self.parse.tables[c2f[coeff]][x0_idx, y0_idx].to(torch.float32) 
        Q01 = self.parse.tables[c2f[coeff]][x0_idx, y1_idx].to(torch.float32)
        Q10 = self.parse.tables[c2f[coeff]][x1_idx, y0_idx].to(torch.float32)
        Q11 = self.parse.tables[c2f[coeff]][x1_idx, y1_idx].to(torch.float32)

        ### WARNING, this Q01, Q10 ordering is a GUESS its one of two possibilities
        C = 1/((x1 - x0)*(y1 - y0)) * torch.tensor([[x1 - x, x - x0]]) @ torch.tensor([[Q00, Q01],[Q10, Q11]]) @ torch.tensor([[y1 - y],[y - y0]])

        return C

    def interp_3d(self, inp, coeff):
        x0_idx, x1_idx, y0_idx, y1_idx, z0_idx, z1_idx = self.get_bounds_3d(inp, coeff)

        x = inp[0]
        y = inp[1]
        z = inp[2]

        alpha_table = self.parse.axes[self.parse.ndinfo[c2f[coeff]]['alpha_fi']]
        beta_table = self.parse.axes[self.parse.ndinfo[c2f[coeff]]['beta_fi']]
        dh_table = self.parse.axes[self.parse.ndinfo[c2f[coeff]]['dh_fi']]
        x0 = alpha_table[x0_idx]
        x1 = alpha_table[x1_idx]
        y0 = beta_table[y0_idx]
        y1 = beta_table[y1_idx]
        z0 = dh_table[z0_idx]
        z1 = dh_table[z1_idx]

        # at the base dh level
        Q000 = self.parse.tables[c2f[coeff]][x0_idx, y0_idx, z0_idx].to(torch.float32)
        Q010 = self.parse.tables[c2f[coeff]][x0_idx, y1_idx, z0_idx].to(torch.float32)
        Q100 = self.parse.tables[c2f[coeff]][x1_idx, y0_idx, z0_idx].to(torch.float32)
        Q110 = self.parse.tables[c2f[coeff]][x1_idx, y1_idx, z0_idx].to(torch.float32)
       
        # at the top dh level
        Q001 = self.parse.tables[c2f[coeff]][x0_idx, y0_idx, z1_idx].to(torch.float32)
        Q011 = self.parse.tables[c2f[coeff]][x0_idx, y1_idx, z1_idx].to(torch.float32)
        Q101 = self.parse.tables[c2f[coeff]][x1_idx, y0_idx, z1_idx].to(torch.float32)
        Q111 = self.parse.tables[c2f[coeff]][x1_idx, y1_idx, z1_idx].to(torch.float32)

        
        
        

py_lookup = Py_lookup()
py_lookup.get_bounds_3d(torch.tensor([0.,0.1,0.1]), 'Cx')

py_lookup.get_bounds_2d(torch.tensor([0.,0.1]), 'Cx_lef')
py_lookup.get_bounds_1d(torch.tensor([0.]), 'CXq')

CXq = py_lookup.interp_1d(torch.tensor([0.]), 'CXq')
Cx_lef = py_lookup.interp_2d(torch.tensor([0.,0.]), 'Cx_lef')

# Cx = py_lookup.interp_3d(torch.tensor([0.,0.,0.]), 'Cx')

