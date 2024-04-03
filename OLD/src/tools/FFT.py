import numpy as np
from numpy.fft import fft, fftn, fftfreq

def get_KGRID_1D( x ):
    dx = x[1]  - x[0]
    Lx = x[-1] - x[0]
    Nx = len(x)

    # Define the k-grid (reciprocal grid)
    #dk   = 2 * np.pi / Lx
    kmax = 1 / 2 / dx # This is not angular frequency
    #kmax = 2 * np.pi / dx # This is angular frequency
    k    = np.linspace( -kmax, kmax, Nx )
    return k

def my_DFT_1D( x, f_x ):

    dx = x[1]  - x[0]
    Nx = len(x)

    # Define the Fourier matrix, W
    n = np.arange(Nx).reshape( (-1,1) )
    m = np.arange(Nx).reshape( (1,-1) )

    ### Centered transform
    ### Here, we shift the indices to the center
    a = Nx//2
    W = np.exp( -2*np.pi*1j * (m-a) * (n-a) / Nx )
    
    # Operate W on the real-space function
    #f_k = np.einsum( "kx,x", W[:,:], f_x[:] )
    f_k = W @ f_x
    
    # Add normalization and integral infinitesimal
    f_k *= dx / np.sqrt( 2 * np.pi )

    return f_k


def numpy_DFT_1D( x, f_x ):
    """
    x 1D grid
    f_x = one-dimensional function
    """
    dx   = x[1] - x[0]
    Nx = len(x)
    f_k  = fft( f_x, n=Nx, norm="ortho" )
    f_k  = np.roll( f_k, (Nx-1)//2 )
    f_k *= np.exp( 1j * 2 * np.pi * np.arange( Nx )/2 )
    return f_k

def get_KGRID_3D( x, N ):

    dx = x[:,1] - x[:,0]

    k = x * 1.0
    for d in range( 3 ):
        k[d] = fftfreq( N[d], d=dx[d] )

    return k

def numpy_DFT_3D( f_x, N=None ):
    """
    N = (Nx, Ny, Nz) grid points
    f_x = three-dimensional function
    """
    if ( N is None ):
        N = ( len(f_x), len(f_x), len(f_x) )
    
    f_k = fftn( f_x, s=N, norm="ortho" )

    return f_k

def my_DFT_3D( x, f_x ):
    """
    NOT WORKING YET.
    """

    dx = x[:,1] - x[:,0]
    N = f_x.shape

    # Create a 3D grid of indices
    n1, n2, n3 = np.meshgrid(np.arange(N[0]), np.arange(N[1]), np.arange(N[2]), indexing='ij')

    # Calculate the phase factors
    #a = N//2 # Centered transform: Shift the indices to the center # TODO
    theta = 2 * np.pi * (n1 * np.arange(N[0])[:, None, None] / N[0] +
                        n2 * np.arange(N[1])[None, :, None] / N[1] +
                        n3 * np.arange(N[2])[None, None, :] / N[2])

    # Operate W on the real-space function
    W = np.exp( -1j * theta )
    f_k = np.sum(W[:,:,:] * f_x[:,:,:,None], axis=(0, 1, 2))
    
    # Add normalization and integral infinitesimal
    f_k *= np.prod(dx) / np.sqrt( 2 * np.pi ) ** 3

    return f_k

if ( __name__ == "__main__" ):
    from matplotlib import pyplot as plt
    """
    Once finalized, place into unittests
    """
    # CHECK 1D EXAMPLE
    XGRID = np.linspace( -10*np.pi,10*np.pi,1001 )
    k0    = 2.0
    f_x   = np.exp( -XGRID**2 / 2 ) * np.exp( -1j * 2 * np.pi * k0 * XGRID )
    #f_k   = my_DFT_1D( XGRID, f_x )
    f_k   = numpy_DFT_1D( XGRID, f_x )
    KGRID = get_KGRID_1D( XGRID )

    plt.plot( KGRID, np.exp(-(2*np.pi*(KGRID+k0))**2 / 2), lw=5, label="EXACT" )
    plt.plot( XGRID, f_x.real, "-", c="black", label="Re[f(x)]" )
    plt.plot( XGRID, f_x.imag, "--", c="black",label="IM[f(x)]" )
    plt.plot( KGRID, f_k.real, "-", c="red",label="Re[f(k)]" )
    plt.plot( KGRID, f_k.imag, "--", c="red",label="Im[f(k)]" )
    plt.xlabel("x, k (a.u.)",fontsize=15)
    plt.ylabel("f(x), f(k) (a.u.)",fontsize=15)
    plt.legend()
    plt.xlim(-3,3)
    plt.savefig("FFT_TEST_1D.jpg",dpi=300)
    plt.clf()

    # CHECK 3D EXAMPLE
    XGRID = np.linspace( -2*np.pi,2*np.pi,51 )
    Nx    = len(XGRID)
    print("Dimesion:", Nx*Nx*Nx)
    kx    = 2.0
    ky    = -2.0
    kz    = 0.0
    f_x   = np.zeros( (Nx,Nx,Nx), dtype=complex )
    EXACT = np.zeros( (Nx,Nx,Nx), dtype=complex )
    for xi in range( Nx ):
        for yi in range( Nx ):
            f_x[xi,yi,:] = np.exp( -XGRID[xi]**2 / 2 ) * np.exp( -1j * 2 * np.pi * kx * XGRID[xi] ) \
                         * np.exp( -XGRID[yi]**2 / 2 ) * np.exp( -1j * 2 * np.pi * k0 * XGRID[yi] ) \
                         * np.exp( -XGRID[:] **2 / 2 ) * np.exp( -1j * 2 * np.pi * k0 * XGRID[:]  )
    
    f_x = f_x / np.linalg.norm(f_x)
    print( "Norm x", np.linalg.norm(f_x) )
    #f_k   = numpy_DFT_3D( f_x, N=(Nx,Nx,Nx) )
    f_k   = my_DFT_3D( np.array([XGRID,XGRID,XGRID]), f_x )
    KGRID = get_KGRID_3D( np.array([XGRID, XGRID, XGRID]), N=(Nx,Nx,Nx) )
    print( "Norm k", np.linalg.norm(f_k) )

    X,Y = np.meshgrid(XGRID,XGRID)
    plt.imshow( f_x[:,:,Nx//2].real, origin='lower' )
    plt.colorbar(pad=0.01)
    plt.savefig("FFT_TEST_3D_fx.jpg",dpi=300)
    plt.clf()

    plt.imshow( f_k[:,:,Nx//2].real, origin='lower' )
    plt.colorbar(pad=0.01)
    plt.savefig("FFT_TEST_3D_fk_RE.jpg",dpi=300)
    plt.clf()

    plt.imshow( f_k[:,:,Nx//2].imag, origin='lower' )
    plt.colorbar(pad=0.01)
    plt.savefig("FFT_TEST_3D_fk_IM.jpg",dpi=300)
    plt.clf()


