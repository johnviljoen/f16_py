import itertools
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Generate Data...
    numdata = 100
    x = np.random.random(numdata)
    y = np.random.random(numdata)
    z = x**2 + y**2 + 3*x**3 + y + np.random.random(numdata)

    # Fit a 3rd order, 2d polynomial
    m = polyfit2d(x,y,z)

    # Evaluate it on a grid...
    nx, ny = 20, 20
    xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), nx), 
                         np.linspace(y.min(), y.max(), ny))
    zz = polyval2d(xx, yy, m)

    # Plot
    #plt.imshow(zz, extent=(x.min(), y.max(), x.max(), y.min()))
    #plt.scatter(x, y, c=z)
    #plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, alpha=0.2)
    ax.scatter(x, y, z, c='r', s=10)
    plt.xlabel('alpha')
    plt.ylabel('beta')
    ax.set_zlabel('Z')
    #ax.axis('equal')
    #ax.axis('tight')
    plt.show()

    import ipdb
    ipdb.set_trace()

def polyfit2d(x, y, z, order=3): # 9 for perfect
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(G, z)

    import ipdb
    ipdb.set_trace()
    return m

def polyval2d(x, y, m):
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z

main()

