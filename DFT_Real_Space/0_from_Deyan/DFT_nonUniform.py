import numpy as np
from scipy.sparse.linalg import cg as conjugate_gradient
from scipy.special import erf
from matplotlib import pyplot as plt
from numba import njit
from time import time
import subprocess as sp

from findiff import FinDiff

def get_globals():

    global N_SCF_ITER, dE_THRESH, do_COMPENSATION, do_TIMING
    N_SCF_ITER = 20
    dE_THRESH  = 1e-5
    do_COMPENSATION = True
    do_TIMING = False

    # Real space grid x
    global Nx, xGRID, dx
    Nx    = 10
    XMIN  = -3
    XMAX  =  3
    # xGRID = np.linspace(-3, 3, Nx)
    # dx    = xGRID[1] - xGRID[0]
    x_nu = np.r_[np.linspace(XMIN, 0.5, 3, endpoint=False), np.linspace(0.5, 1.2, 7, endpoint=False),
                 np.linspace(1.2, 1.9, 2, endpoint=False), np.linspace(1.9, 2.9, 5, endpoint=False),
                 np.linspace(2.9, 10, 3)]
    
    print( "Matrix Dimension: (%1.0f,%1.0f)" % (Nx**3,Nx**3) )
    print( "Grid Spacing: %1.3f a.u." % dx )

    global I
    I = np.identity( Nx )

    # Define system parameters for Helium
    global Ne, NOCC
    Ne    = 2       # Number of electrons
    NOCC  = Ne // 2 # Number of occupied orbitals

    global Zion, GEOM
    # He atom
    Zion = np.array([2]) # atomic numbers
    GEOM = np.zeros( (1,3) )
    GEOM[0] = np.array( [0.0, 0.0, 0.0] )
    """
    # H4 system
    Zion = np.array([1,1,1,1]) # atomic numbers
    GEOM = np.zeros( (4,3) )
    GEOM[0] = np.array( [-1.0, -1.0, 0.0] )
    GEOM[1] = np.array( [-1.0, 1.0, 0.0] )
    GEOM[2] = np.array( [1.0, -1.0, 0.0] )
    GEOM[3] = np.array( [1.0, 1.0, 0.0] )
    """
    """
    # H2 molecule
    Zion  = np.array([1,1]) # atomic numbers
    GEOM    = np.zeros( (2,3) )
    GEOM[0] = np.array( [-2.0, 0.0, 0.0] )
    GEOM[1] = np.array( [ 2.0, 0.0, 0.0] )
    """

    V_EXT = get_External_Potential()
    V_EXT = V_EXT.reshape( (Nx,Nx,Nx) )
    plt.imshow( V_EXT[:,:,Nx//2], origin='lower', cmap='gray', extent=(xGRID[0],xGRID[-1],xGRID[0],xGRID[-1]) )
    plt.colorbar(pad=0.01)
    plt.savefig( "V_EXT.jpg", dpi=300 )
    plt.clf()


def timer_func(func): 
    def wrap_func(*args, **kwargs): 
        t1 = time() 
        result = func(*args, **kwargs)
        t2 = time() 
        if ( do_TIMING == True ):
            print('\t%s executed in %1.3f s' % (func.__name__, t2-t1)) 
        return result 
    return wrap_func 





@timer_func
def get_T():

    def get_T_DVR():
        T = np.zeros( (Nx, Nx) )
        for j in range(Nx):
            for k in range(Nx):
              if ( j == k ):
                T[j,j] = np.pi**2 / 3
              else:
                T[j,k] = 2 * (-1)**(j-k) / (j-k)**2
        return T / 2 / dx**2
  
    def get_T_FD(): # First-order central difference
        T = np.diag(  -np.ones(Nx-1), k=-1 ) + \
            np.diag( 2*np.ones(Nx),   k=0 )  + \
            np.diag(  -np.ones(Nx-1), k=1 )
        return T / 2 / dx**2

    # Compute the total kinetic energy operator in three dimensions
    T = get_T_DVR()
    #T = get_T_FD()
    T = np.kron( np.kron(T, I), I ) + \
       np.kron( np.kron(I, T), I ) + \
       np.kron( np.kron(I, I), T )

    return T

@timer_func
def get_External_Potential():
    V_EXT = np.zeros( (Nx,Nx,Nx) )
    for xi,x in enumerate( xGRID ):
        for yi,y in enumerate( xGRID ):
            for zi,z in enumerate( xGRID ):
                r   = np.array([x,y,z])
                for at in range( len(GEOM) ):
                    dr  = np.linalg.norm( r - GEOM[at] )
                    V_EXT[xi,yi,zi] += -Zion[at] / dr
    return V_EXT.flatten()



@timer_func
@njit
def get_Exchange_Correlation( n ):

        # Define parameters for exchange-correlation potential
        a,b,c,d,gamma,beta1,beta2 =  0.0311,-0.048,0.0020,-0.0116,-0.1423,1.0529,0.3334

        # Compute exchange-correlation potential (Local Density Approximation)
        V_EXCH = -(3/np.pi)**(1/3) * n ** (1/3)
        rs     = ( (3/(4*np.pi))**(1/3) ) * (1/n)**(1/3)
        f_EXCH = -(9*np.pi/4)**(1/3) * (3/4/np.pi) * rs

        f_CORR = np.zeros( (Nx**3) )
        V_CORR = np.zeros( (Nx**3) )
        for j in range( Nx**3 ):
            if (rs[j] < 1):
                f_CORR[j] = a*np.log(rs[j])+b+c*rs[j]*np.log(rs[j])+d*rs[j]
                V_CORR[j] = a*np.log(rs[j])+(b-(a/3)) + \
                              (2/3)*c*rs[j]*np.log(rs[j]) + \
                              (1/3)*(2*d-c)*rs[j]
            else:
                f_CORR[j] = gamma/(1+beta1*np.sqrt(rs[j])+beta2*rs[j])
                V_CORR[j] = f_CORR[j]*(1 + (7/6)*beta1*np.sqrt(rs[j]) + \
                              (4/3)*beta2*rs[j])/(1+beta1*np.sqrt(rs[j]) + \
                              beta2*rs[j])

        f_XC = f_CORR + f_EXCH
        V_XC = V_EXCH + V_CORR
        return V_XC.flatten(), f_XC

@timer_func
def get_Compenstation_Charge():

    if ( do_COMPENSATION == False ):
        return 0.0, 0.0
        #return np.zeros( (Nx**3) ), np.zeros( (Nx**3) )

    # Define compensation charge and potential (for real-space grids only)
    SIG     = 1.0 # Choose width to be Bohr radius
    n_COMP  = np.zeros( (Nx,Nx,Nx) )
    V_COMP  = np.zeros( (Nx,Nx,Nx) )
    for xi,x in enumerate( xGRID ):
        for yi,y in enumerate( xGRID ):
            for zi,z in enumerate( xGRID ):
                r   = np.array([x,y,z])
                for at in range( len(GEOM) ):
                    dr  = np.sqrt( np.linalg.norm( r - GEOM[at] ) )
                    n_COMP[xi,yi,zi]  += -Zion[at] * np.exp(-dr**2/2/SIG**2) / np.sqrt(2 * np.pi)**3 / SIG**3 # Gaussian compensation charge
                    V_COMP[xi,yi,zi]   += -Zion[at] / dr * erf(dr/np.sqrt(2))                # Integral of Gaussian compensation charge
                    #V_COMP[xi,yi,zi]   += -0.5 * Zion[at] * erf(dr/np.sqrt(2)/SIG)                # Integral of Gaussian compensation charge
    
    NORM       = np.sum(n_COMP) * dx**3
    n_COMP    *= -np.sum(Zion)/NORM # Normalize the Gaussian charge
    V_COMP    *= -np.sum(Zion)/NORM # Normalize the Gaussian charge
    return n_COMP.flatten(), V_COMP.flatten()

@timer_func
def get_Hartree_Potential( n, T, n_COMP, V_COMP ):
    # Solve Poisson equation to get Hartree potential
    V_HART, info = conjugate_gradient( T, -4*np.pi*(n+n_COMP) )
    V_HART -= V_COMP
    return V_HART # (Nx**3) vector

@timer_func
@njit
def get_Kinetic_Energy( phi, T ):
    E_KIN = 0
    for i in range( NOCC ):
        E_KIN += 2 * phi[:,i] @ T @ phi[:,i]
    return E_KIN

#@njit
def solve_KS( H ):
    # Solve the Kohn-Sham equations
    eps, phi = np.linalg.eigh( H )
    return eps, phi

def get_Total_Energy( n, phi, T, f_XC, V_HART, V_EXT ):
    E_KIN   = get_Kinetic_Energy( phi, T )
    E_EXT   = np.sum( n * V_EXT) * dx**3
    E_HART  = 0.5 * np.sum( n * V_HART ) * dx**3
    E_XC    = np.sum( f_XC * n ) * dx**3
    return E_KIN + E_EXT + E_HART + E_XC

def get_Density( phi ):
    n = np.zeros( (Nx**3) )
    for i in range( NOCC ):
        n += 2 * phi[:,i]**2
    return n

@timer_func
def do_SCF_Iterations( T, V_EXT ):
    n_COMP, V_COMP = get_Compenstation_Charge()
    V_TOT = np.diag( V_EXT ) # Initial guess for total potential

    for scf_iter in range( N_SCF_ITER ):
        print( "\n\nIteration: %1.0f" % (scf_iter) )
        T0 = time()

        # Solve KS equations
        TEIG = time()
        eps, phi = solve_KS( T + V_TOT )
        print("\tEigensolver Time: %1.3f" % (time() - TEIG) )
        eps = np.real( eps[:2*NOCC] )
        phi = np.real( phi[:,:2*NOCC] )

        # Construct electron density from occupied orbitals
        n = get_Density( phi )

        V_XC, f_XC = get_Exchange_Correlation( n )
        V_HART     = get_Hartree_Potential( n, T, n_COMP, V_COMP )
        V_TOT      = np.diag( V_XC + V_HART + V_EXT )

        E_TOT      = get_Total_Energy( n, phi, T, f_XC, V_HART, V_EXT )

        print( "Energy: %1.6f a.u." % (E_TOT) )
        if ( scf_iter > 1 and abs(E_TOT - E_OLD) < dE_THRESH ):
            break
        E_OLD = E_TOT * 1

        print( "\tSCF Time: %1.3f s" % (time() - T0 ) )

    return E_TOT, eps, phi

def plot_Orbitals( EIGS, PSI ):
    sp.call("rm Orbital*.jpg", shell=True)
    # Plot the occupied orbitals
    PSI = PSI.reshape( (Nx,Nx,Nx,2*NOCC) )
    for i in range( 2*NOCC ):
        plt.imshow( PSI[:,:,Nx//2,i], cmap='afmhot_r', extent=(xGRID[0],xGRID[-1],xGRID[0],xGRID[-1]) )
        plt.colorbar(pad=0.01)
        plt.title("Orbital Energy: %1.3f a.u." % EIGS[i],fontsize=15)
        plt.savefig( "Orbital_%1.0f.jpg" % i, dpi=300 )
        plt.clf()

def main():
    get_globals()
    T                = get_T()                     # (Nx**3,Nx**3) matrix
    V_EXT            = get_External_Potential()    # (Nx**3) vector
    E_TOT, EIGS, PSI = do_SCF_Iterations( T, V_EXT )
    print("Final Energy: %1.6f a.u." % E_TOT)
    plot_Orbitals( EIGS, PSI )

if ( __name__ == "__main__" ):
    main()
