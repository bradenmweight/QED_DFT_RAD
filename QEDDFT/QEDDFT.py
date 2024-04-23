import numpy as np
import matplotlib.pyplot as plt
from numba import jit

def get_Globals():
    # System parameters
    global dim, L, width, Nx, xGRID, dx, OCC
    dim   = 1
    width = 0.5  # Width of the external potential
    L     = 30.0  # Length of the spatial dimension
    Nx    = 501  # Number of grid points
    xGRID = np.linspace(-L/2, L/2, Nx)  # Spatial grid
    dx    = xGRID[1] - xGRID[0]  # Grid spacing
    OCC   = np.array([2,2,2]) # Occupied Orbital Occupations

    # Cavity parameters
    global omega, lmcpl
    omega = 1.0
    lmcpl = 1 #1e-1 #1 #10

    # Construct the discretized second derivative operator
    global T
    T = np.zeros( (Nx,Nx) )
    for xi in range( Nx ):
        for xj in range( Nx ):
            if (xi == xj):
                T[xi,xi] = np.pi**2/3
            else:
                T[xi,xj] = 2*(-1)**(xi-xj)/(xi-xj)**2
    T /= dx**2 * 2

    # Construct the external/local potential
    V_EXT = np.diag( get_Potential(xGRID) )

    # Construct the bare Hamiltonian
    global H0
    H0 = T + V_EXT

    # Define the iteration parameters
    global Nit, threshold
    Nit       = 500
    threshold = 1e-6

# Define the external potential
@jit(nopython=True)
def get_Potential(x):
    return -1./np.sqrt(x**2+width**2)

# Define the pxLDA potential
@jit(nopython=True)
def get_v_ep_x_LDA(n,omega,lmcpl,d=1):
    if ( d != 1 ):
        print( "LDA ep exchange is only defined for d = 1")
        return np.diag( 0 * n )
    v_pxlda = -(np.pi**2/8.0)*(lmcpl**2/(omega**2+lmcpl**2))*n**2
    return np.diag( v_pxlda )

@jit(nopython=True)
def get_V_ee_x_LDA( n, d=1 ):
    V_ee_x_LDA = -((3/np.pi)**(1/3)) * (d/(d+2)) * n ** (1/d)
    return np.diag( V_ee_x_LDA )

@jit(nopython=True)
def get_E_ee_x_LDA( n, d=1 ):
    E_ee_x_LDA = np.sum( -((3/4)*(3/np.pi)**(1/3)) * (d/(d+2)) * n**(2/d) ) * dx**d
    return E_ee_x_LDA

@jit(nopython=True)
def get_V_ee_c_LDA(n, d=3):
    if ( d != 3 ):
        print( "LDA correlation is only defined for d = 3")
        return np.diag( 0 * n )
    A, B, C, D = 0.0311, -0.048, 0.002, -0.0116
    rs = (3/4/np.pi)**(1/3) / n**(1/3)
    Q = np.sqrt(4*C - D**2)
    x = np.sqrt(rs)
    f1 = 2*B/(Q-D)
    f2 = 2*B/(Q+D)
    f3 = np.log(x**2/(1 + 2*A*x + B*x**2))
    f4 = 2*(A + 2*B*x)*(1/np.sqrt(1 + 4*A*x + 4*B*x**2))
    V_ee_c_LDA = A*(f1*f3 - f2*(f3 + f4))
    return np.diag( V_ee_c_LDA )

@jit(nopython=True)
def get_E_ee_c_LDA(n, d=3):
    if ( d != 3 ):
        print( "LDA correlation is only defined for d = 3")
        return 0
    A, B, C, D = 0.0311, -0.048, 0.002, -0.0116
    rs = (3/4/np.pi)**(1/3) / n**(1/3)
    Q = np.sqrt(4*C - D**2)
    x = np.sqrt(rs)
    f1 = 2*B/(Q-D)
    f2 = 2*B/(Q+D)
    f3 = np.log(x**2/(1 + 2*A*x + B*x**2))
    E_ee_c_LDA = np.sum( A*(f1*f3 - f2*(f3 + x*(1/np.sqrt(1 + 4*A*x + 4*B*x**2)))) ) * dx**d
    return E_ee_c_LDA

def get_V_HART( n ):

    # Define the momentum grid
    kGRID = np.fft.fftfreq(Nx) * 2 * np.pi / dx

    # Fourier transform of the density
    n_k   = np.fft.fft(n, norm='ortho') / np.sqrt( Nx )
    n_k[0] = 0.0 # Exclude the k = 0 component

    # Fourier transform of the Coulomb interaction
    v_k = 4 * np.pi / (kGRID ** 2 + 1e-10)

    # Convolution in Fourier space
    vH_k = n_k * v_k

    # Inverse Fourier transform to get Hartree potential in real space
    V_HART = np.fft.ifft(vH_k).real / np.sqrt( Nx )
    V_HART = np.roll( V_HART, Nx//2)

    # plt.plot( xGRID, V_HART )
    # plt.savefig("V_HART.jpg", dpi=300)
    # plt.clf()

    return np.diag( V_HART )

@jit(nopython=True)
def get_new_rho(Hamiltonian):
    Ei, Ui = np.linalg.eigh(Hamiltonian)
    rho_gs = np.zeros( (Nx) )
    for occ in range( len(OCC) ):
        rho_gs += OCC[occ] * np.abs(Ui[:,occ])**2 / dx
    return Ei, Ui, rho_gs

# Define rho difference
@jit(nopython=True)
def get_rho_diff(rho_old, rho_new, dx):
    sum_abs_drho = np.sum(np.abs(rho_new - rho_old))*dx
    return sum_abs_drho

@jit(nopython=True)
def get_Kinetic_Energy( Ui ):
    Ek = 0.0
    for occ in range( len(OCC) ):
        Ek += 0.5 * OCC[occ] * Ui[:,occ] @ T @ Ui[:,occ] * dx**3
    return Ek

#@jit(nopython=True)
def get_Total_Energy( n, Ui, V_XC, V_HART, V_ep ):
    E_KIN   = get_Kinetic_Energy( Ui )
    E_EXT   = 0.5 * np.sum( n * get_Potential(xGRID) ) * dx**3
    E_HART  = 0.5 * np.sum( n * V_HART ) * dx**3
    E_XC    = 0.5 * np.sum( n * V_XC   ) * dx**3
    E_ep    = 0.5 * np.sum( n * V_ep   ) * dx**3
    print( "Ek = %1.4f Eext = %1.4f Ehart = %1.4f E_ee_xc = %1.4f E_ep_xc = %1.4f" % (E_KIN, E_EXT, E_HART, E_XC, E_ep) )
    return E_KIN + E_EXT + E_HART + E_XC + E_ep

def do_SCF( cavity=True ):

    Ei, Ui, rho_old = get_new_rho( H0 )

    # Self consistent loop
    for it in range(Nit):

        V_HART     = get_V_HART( rho_old )
        V_px_LDA   = get_v_ep_x_LDA(rho_old,omega,lmcpl, d=dim) # Coded only for d=1
        V_ee_x_LDA = get_V_ee_x_LDA(rho_old, d=dim) # For any d
        V_ee_c_LDA = get_V_ee_c_LDA(rho_old, d=dim) # Only written for d=3

        V_XC = V_ee_x_LDA + V_ee_c_LDA + V_px_LDA * (cavity==True)

        H               = H0 + V_HART + V_XC
        Ei, Ui, rho_new = get_new_rho( H )
        drho            = get_rho_diff(rho_old, rho_new, dx)

        rho_old = rho_new

        EGS = get_Total_Energy( rho_new, Ui, np.diagonal(V_XC), np.diagonal(V_HART), np.diagonal(V_px_LDA) ) #, f_XC, V_HART, V_EXT )

        print(f"interation: {it};    AbsDenDiff={drho:1.6f}")
        if ( np.abs(drho) < threshold ):
            break
        if ( it == Nit-1 ):
            print( "Warning: Not converged." )

    return rho_new, EGS

def plot_density( rho_outside, EGS_outside, rho_cavity, EGS_cavity ):


    plt.plot( xGRID, get_Potential(xGRID), '-', color='black', lw=6, alpha=0.5, label='$V_\mathrm{EXT}$')

    plt.plot(xGRID, xGRID*0 + EGS_outside, color='black', linestyle='--', lw=1)
    plt.plot(xGRID, xGRID*0 + EGS_cavity, color='red', linestyle='--', lw=1)
    plt.plot(xGRID, rho_outside + EGS_outside, color='black', linestyle='-', lw=4, label='Without Cavity')
    plt.plot(xGRID, rho_cavity  + EGS_cavity, color='red', linestyle='--', lw=4, label='Inside Cavity')

    plt.xlabel('Position [Bohr]', size=15)
    plt.ylabel(r'Density [Bohr$^{-1}$]', size=15)
    plt.legend()

    plt.tight_layout()
    plt.savefig("Density.jpg", dpi=300)

def main():
    get_Globals()

    rho_outside, EGS_outside = do_SCF( cavity=False )
    rho_cavity, EGS_cavity = do_SCF( cavity=True )
    plot_density( rho_outside, EGS_outside, rho_cavity, EGS_cavity )

if ( __name__ == "__main__" ):
    main()
