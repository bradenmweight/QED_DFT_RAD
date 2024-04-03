import numpy as np

"""
This will store all methods having to do with plane wave basis
"""


def get_G_GRID( PROPERTIES ):

    E_CUT = PROPERTIES["E_CUT"]
    a     = PROPERTIES["a"]

    # Determine the maximum k-vector magnitude from the cutoff energy
    G_MAX = np.sqrt(2 * E_CUT )
    # Reciprocal Lattice Vectors
    b = 2*np.pi/a

    # Generate a grid of k-vectors within the cutoff energy
    G_GRID = []
    n_GRID = []
    nmax   = 20 # Check up to this many reciprical points. There is a better way...
    G_GRID.append(0.)
    n_GRID.append([0,0,0])
    for n0 in range( -nmax,nmax ): # This is hard-coded. Change later.
        for n1 in range( -nmax,nmax ): # This is hard-coded. Change later.
            for n2 in range( -nmax,nmax ): # This is hard-coded. Change later.
                G = np.array([ n0 * b[0], n1 * b[1], n2 * b[2] ]) # Integer multiples of reciprocal lattice vectors
                if ( n0 == 0 and n1 == 0 and n2 == 0 ):
                    continue
                if ( np.linalg.norm(G) >= G_MAX ):
                    #print(n0,n1,n2, np.linalg.norm([n0,n1,n2]), G_MAX)
                    continue
                G_GRID.append( np.linalg.norm(G) )
                n_GRID.append( [n0,n1,n2] )

    G_GRID = np.array( G_GRID )
    n_GRID = np.array( n_GRID )
    NG = len(G_GRID)
    print( "# Basis Functions =", NG )
    print( "# Points (x,y,z) =", np.max(n_GRID[:,0]), np.max(n_GRID[:,1]), np.max(n_GRID[:,2])  )
    PROPERTIES["NBASIS"] = NG
    PROPERTIES["G_GRID"] = G_GRID
    PROPERTIES["n_GRID"] = n_GRID


    ####### For testing ########
    #from matplotlib import pyplot as plt
    
    # x = np.linspace(0,int(a[0]),1000)
    # for Gi,G in enumerate(G_GRID):
    #     plt.plot( x, (np.exp(1j * G[0] * x)).real )
    # plt.xlabel("Simulation Box, x (a.u.)",fontsize=15)
    # plt.ylabel("Real Part of Plane Wave Basis Functions",fontsize=15)
    # plt.savefig("G_GRID.jpg",dpi=300)
    # plt.clf()

    # G_MAGS = np.array([ [G_GRID[j,0],G_GRID[j,1]] for j in range(NG) ])
    # plt.scatter( G_MAGS[:,0], G_MAGS[:,1] )
    # plt.xlabel("Gx",fontsize=15)
    # plt.ylabel("Gy",fontsize=15)
    # plt.savefig("G_MAGS.jpg",dpi=300)
    # plt.clf()

    return PROPERTIES
