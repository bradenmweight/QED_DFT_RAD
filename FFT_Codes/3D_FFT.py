import numpy as np
from matplotlib import pyplot as plt

def get_Globals():
    global Lx, Nx, x, dx
    XMIN = -10
    XMAX =  10
    Lx   = XMAX - XMIN
    Nx   = 256
    x    = np.linspace( 0, Lx, Nx )
    dx   = x[1] - x[0]

def FFT_1D( f, dx, Nx ):
    fk = np.fft.fft( f, norm='ortho' ) / np.sqrt( Nx )
    k  = np.fft.fftfreq( Nx ) * 2 * np.pi / dx
    fk = np.fft.fftshift( fk )
    k  = np.fft.fftshift( k )
    return k, fk

def FFT_2D( f, dx, Nx ):
    fk   = np.fft.fft2( f, norm='ortho' ) / np.sqrt( Nx )**2
    kx   = np.fft.fftfreq( Nx ) * 2 * np.pi / dx
    ky   = np.fft.fftfreq( Nx ) * 2 * np.pi / dx
    kx,ky = np.meshgrid( kx, ky )
    fk   = np.fft.fftshift( fk )
    kx   = np.fft.fftshift( kx )
    ky   = np.fft.fftshift( ky )
    return kx, ky, fk

def FFT_3D( f, dx, Nx ):
    fk   = np.fft.fftn( f, norm='ortho' ) / np.sqrt( Nx )**3
    kx   = np.fft.fftfreq( Nx ) * 2 * np.pi / dx
    ky   = np.fft.fftfreq( Nx ) * 2 * np.pi / dx
    kz   = np.fft.fftfreq( Nx ) * 2 * np.pi / dx
    kx,ky,kz = np.meshgrid( kx, ky, kz )
    fk   = np.fft.fftshift( fk )
    kx   = np.fft.fftshift( kx )
    ky   = np.fft.fftshift( ky )
    kz   = np.fft.fftshift( kz )
    return kx, ky, kz, fk

def TEST_1D():

    f    = np.sin( 2 * np.pi * x / Lx ) * np.exp( -x**2 / 2 )
    k,fk = FFT_1D( f, dx, Nx )

    plt.plot( k, np.abs(fk) )
    plt.savefig("FFT_1D.jpg", dpi=300)
    plt.clf()


def TEST_2D():
    fx   = np.sin( 2 * np.pi * x / Lx ) * np.exp( -x**2 / 2 / 0.5**2 )
    fy   = np.sin( 2 * np.pi * x / Lx ) * np.exp( -x**2 / 2 / 2**2 )
    f    = np.outer( fx, fy )
    kx,ky,fk = FFT_2D( f, dx, Nx )

    plt.imshow( np.abs(fk), extent=(kx.min(),kx.max(),ky.min(),ky.max()), origin='lower' )
    plt.colorbar(pad=0.01)
    plt.savefig("FFT_2D.jpg", dpi=300)
    plt.clf()


def TEST_3D():
    fx   = np.sin( 2 * np.pi * x / Lx ) * np.exp( -x**2 / 2 / 0.5**2 )
    fy   = np.sin( 2 * np.pi * x / Lx ) * np.exp( -x**2 / 2 / 2**2 )
    fz   = np.sin( 2 * np.pi * x / Lx ) * np.exp( -x**2 / 2 )
    f    = np.outer( fx, np.outer( fy, fz ) )
    f    = f.reshape( (Nx,Nx,Nx) )
    kx,ky,kz,fk = FFT_3D( f, dx, Nx )

    plt.imshow( np.abs(fk[:,:,Nx//2]), extent=(kx.min(),kx.max(),ky.min(),ky.max()), origin='lower' )
    #plt.imshow( np.abs(fk[:,:,-1]), extent=(kx.min(),kx.max(),ky.min(),ky.max()), origin='lower' )
    plt.colorbar(pad=0.01)
    plt.savefig("FFT_3D.jpg", dpi=300)
    plt.clf()

def main():
    get_Globals()
    TEST_1D()
    TEST_2D()
    TEST_3D()

if ( __name__ == "__main__" ):
    main()