import itertools
import torch
from tables import py_lookup
from parameters import c2f, f2c
import matplotlib.pyplot as plt
from tqdm import tqdm

# alpha axis array
ALPHA1 = py_lookup.parse.axes['ALPHA1']
ALPHA2 = py_lookup.parse.axes['ALPHA2']

# beta axis array
BETA1 = py_lookup.parse.axes['BETA1']

# el axis array
DH1 = py_lookup.parse.axes['DH1']
DH2 = py_lookup.parse.axes['DH2']

def polyfit1d(x_axis, filename):
    coeff = torch.zeros([x_axis.shape[0]])
 
    # fill the coefficient tensor with values
    for i, x in enumerate(x_axis):
        coeff[i] = py_lookup.interp_1d(torch.tensor([x]),f2c[filename])

    # define a lookup function for this coefficient for convenience
    def f(x):
        return py_lookup.interp_1d(torch.tensor([x]), f2c[filename]) 

    # define a meshgrid for the known points, flatten and turn into normalised column vector
    X = x_axis
    norm_val = x_axis.abs().max()
    XX = (X/norm_val).unsqueeze(1)

    """ 
    form the nth order matrices for polyfit by finding every possible
    combination of powers of x,y using itertools.product
    """
    x_order = X.shape[0]
    x_list = []
    for i in range(x_order):
        x_list.append(XX**i)


    # comb is now a list of tuples of 3 elements. We wish to multiply
    # each of these with themselves
    temp = []
    for elem in x_list:
        temp.append(elem[0])


    # concatenate the output list together to form the power coeff matrix A
    A = torch.cat(temp, axis=0).unsqueeze(0)
    B = coeff.flatten().unsqueeze(0)

    # use the lstsq function rather than linalg inv as it handles ill conditioned problems well
    C,_,_,_ = torch.linalg.lstsq(A,B)

    # evaluate the polynomial
    output = (A @ C).reshape(X.shape)

    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    #ax.plot_surface(X, Y, output, rstride=1, cstride=1, alpha=0.2)
    #ax.scatter(X, Y, coeff, c='r', s=10)
    #plt.xlabel('alpha')
    #plt.ylabel('beta')
    #ax.set_zlabel('Z')
    ##ax.axis('equal')
    ##ax.axis('tight')
    #plt.show()

    return C, output

def polyfit2d(x_axis, y_axis, filename):
    coeff = torch.zeros([
        x_axis.shape[0],
        y_axis.shape[0]])
 
    # fill the coefficient tensor with values
    for i, x in enumerate(x_axis):
        for j, y in enumerate(y_axis):
            coeff[i,j] = py_lookup.interp_2d(torch.tensor([x,y]),f2c[filename])

    # define a lookup function for this coefficient for convenience
    #def f(x,y):
    #    return py_lookup.interp_2d(torch.tensor([x,y]), f2c[filename]) 

    # define a meshgrid for the known points, flatten and turn into normalised column vector
    X, Y = torch.meshgrid(x_axis, y_axis)
    XX = X.flatten().unsqueeze(1)
    YY = Y.flatten().unsqueeze(1)
    norm_val = max(
        x_axis.abs().max(),
        y_axis.abs().max())
    XX = XX/norm_val
    YY = YY/norm_val

    """ 
    form the nth order matrices for polyfit by finding every possible
    combination of powers of x,y using itertools.product
    """
    x_order = 20
    y_order = 19
    x_list = []
    y_list = []
    for i in range(x_order):
        x_list.append(XX**i)
    for i in range(y_order):
        y_list.append(YY**i)

    # find every combination of powers using itertools
    comb = list(itertools.product(
        x_list,
        y_list
    ))

    # comb is now a list of tuples of 3 elements. We wish to multiply
    # each of these with themselves
    temp = []
    for elem in comb:
        temp.append(elem[0]*elem[1])
    
    # concatenate the output list together to form the power coeff matrix A
    A = torch.cat(temp, axis=1)
    B = coeff.flatten()

    # use the lstsq function rather than linalg inv as it handles ill conditioned problems well
    C,_,_,_ = torch.linalg.lstsq(A,B)

    # evaluate the polynomial
    output = (A @ C).reshape(X.shape)

    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    #ax.plot_surface(X, Y, output, rstride=1, cstride=1, alpha=0.2)
    #ax.scatter(X, Y, coeff, c='r', s=10)
    #plt.xlabel('alpha')
    #plt.ylabel('beta')
    #ax.set_zlabel('Z')
    ##ax.axis('equal')
    ##ax.axis('tight')
    #plt.show()

    return C, output

def polyfit3d(x_axis, y_axis, z_axis, filename):
    coeff = torch.zeros([
        x_axis.shape[0],
        y_axis.shape[0],
        z_axis.shape[0]])
 
    # fill the coefficient tensor with values
    for i, x in enumerate(x_axis):
        for j, y in enumerate(y_axis):
            for k, z in enumerate(z_axis):
                coeff[i,j,k] = py_lookup.interp_3d(torch.tensor([x,y,z]),f2c[filename])

    # define a lookup function for this coefficient for convenience
    def f(x,y,z):
        return py_lookup.interp_3d(torch.tensor([x,y,z]), f2c[filename]) 

    # define a meshgrid for the known points, flatten and turn into normalised column vector
    X, Y, Z = torch.meshgrid(x_axis, y_axis, z_axis)
    XX = X.flatten().unsqueeze(1)
    YY = Y.flatten().unsqueeze(1)
    ZZ = Z.flatten().unsqueeze(1)
    norm_val = max(
        x_axis.abs().max(),
        y_axis.abs().max(),
        z_axis.abs().max())
    XX = XX/norm_val
    YY = YY/norm_val
    ZZ = ZZ/norm_val

    """ 
    form the nth order matrices for polyfit by finding every possible
    combination of powers of x,y,z using itertools.product
    """
    x_order = 20
    y_order = 19
    z_order = 2
    x_list = []
    y_list = []
    z_list = []
    for i in range(x_order):
        x_list.append(XX**i)
    for i in range(y_order):
        y_list.append(YY**i)
    for i in range(z_order):
        z_list.append(ZZ**i)

    # find every combination of powers using itertools
    comb = list(itertools.product(
        x_list,
        y_list,
        z_list
    ))

    # comb is now a list of tuples of 3 elements. We wish to multiply
    # each of these with themselves
    temp = []
    for elem in comb:
        temp.append(elem[0]*elem[1]*elem[2])
    
    # concatenate the output list together to form the power coeff matrix A
    A = torch.cat(temp, axis=1)
    B = coeff.flatten()

    # use the lstsq function rather than linalg inv as it handles ill conditioned problems well
    C,_,_,_ = torch.linalg.lstsq(A,B)

    # evaluate the polynomial
    output = (A @ C).reshape(X.shape)

    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    #slice = 1
    #ax.plot_surface(X[:,:,slice], Y[:,:,slice], output[:,:,slice], rstride=1, cstride=1, alpha=0.2)
    #ax.scatter(X[:,:,slice], Y[:,:,slice], coeff[:,:,slice], c='r', s=10)
    #plt.xlabel('alpha')
    #plt.ylabel('beta')
    #ax.set_zlabel('Z')
    ##ax.axis('equal')
    ##ax.axis('tight')
    #plt.show()

    return C, output

for filename in tqdm(f2c):
    #print(filename)
    #filename = c2f['Cy']

    # we need the ndinfo
    #print('gathering ndinfo')
    try:
        ndinfo = py_lookup.parse.ndinfo[filename]
    except:
        print('Skipping delta_Cm_ds')
        continue

    """ 
    there are four options here
        1D ALPHA1
        1D DH1 (eta_el)
        2D ALPHA1 BETA1
        3D ALPHA1 BETA1 DH1
    """

    if ndinfo['alpha_fi'] is not None:
        if ndinfo['beta_fi'] is not None:
            if ndinfo['dh_fi'] is not None:
                # 3D tables of alpha1, beta1, dh1
                print(f"3D table of ALPHA1, BETA1, DH1 detected, {f2c[filename]}")
                x_axis = py_lookup.parse.axes[ndinfo['alpha_fi']]
                y_axis = py_lookup.parse.axes[ndinfo['beta_fi']]
                z_axis = py_lookup.parse.axes[ndinfo['dh_fi']]
                C, out = polyfit3d(x_axis,y_axis,z_axis, filename)
                torch.save((C,out), f"aerodata_pt/{f2c[filename]}.pt")
            else:
                # 2D tables of alpha1, beta1
                print(f"2D table of ALPHA1, BETA1 detected, {f2c[filename]}")
                x_axis = py_lookup.parse.axes[ndinfo['alpha_fi']]
                y_axis = py_lookup.parse.axes[ndinfo['beta_fi']]
                C, out = polyfit2d(x_axis,y_axis, filename)
                torch.save((C,out), f"aerodata_pt/{f2c[filename]}.pt")
        else:
            # 1D tables of alpha1
            print(f"1D table of ALPHA1 detected, {f2c[filename]}")
            x_axis = py_lookup.parse.axes[ndinfo['alpha_fi']]
            C, out = polyfit1d(x_axis, filename)
            torch.save((C,out), f"aerodata_pt/{f2c[filename]}.pt")
    elif ndinfo['dh_fi'] is not None:
        # 1D table of dh1 (eta_el)
        print(f"1D table of DH1 detected, {f2c[filename]}")
        x_axis = py_lookup.parse.axes[ndinfo['dh_fi']]
        C, out = polyfit1d(x_axis, filename)
        torch.save((C,out), f"aerodata_pt/{f2c[filename]}.pt")

    # initialise the coefficient tensor based on ndinfo size
    #x_axis = py_lookup.parse.axes[ndinfo['alpha_fi']]
    #y_axis = py_lookup.parse.axes[ndinfo['beta_fi']]
    #z_axis = py_lookup.parse.axes[ndinfo['dh_fi']]

    # this is how we call each of the different types
    #C, out = polyfit1d(x_axis, c2f['CXq'])
    #C, out = polyfit1d(z_axis, c2f['eta_el'])
    #C, out = polyfit2d(x_axis,y_axis,c2f['Cy'])
    #C, out = polyfit3d(x_axis,y_axis,z_axis,c2f['Cx'])

    #import ipdb
    #ipdb.set_trace()

