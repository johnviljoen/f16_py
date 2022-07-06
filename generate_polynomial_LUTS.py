import numpy as np
from tables import py_lookup
import matplotlib.pyplot as plt


def main():
    print('this is the main function')
    
    # lets do a 1D first w/ hifi_damping
    key = 'CX1120_ALPHA1_204.dat'
    table_1D = py_lookup.tables[key]

    # fidelity
    ndinfo = py_lookup.ndinfo[key]
    for i, alpha in enumerate(py_lookup.axes[ndinfo['alpha_fi']]):
        print(alpha)
        print(i)
        print(table_1D[i])
    C = np.polyfit(py_lookup.axes[ndinfo['alpha_fi']], table_1D, 20)
   
    #x[0]**20 * C[0]

    plt.plot(py_lookup.axes[ndinfo['alpha_fi']], table_1D)
    plt.show()

    import pdb
    pdb.set_trace()
     

if __name__ == '__main__':
    main()

