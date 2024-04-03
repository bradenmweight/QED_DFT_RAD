import numpy as np
from plane_wave_basis import get_G_GRID
from energy import get_Energy

"""
This is the main driver code.

Read input from user: input.py
Build system and basis: build.py/plane_wave_basis.py

Construct Hamiltonian: KS.py
Solve KS equations: SCF.py
"""

# Local Density Approximation
# E_xc   = \int d3r eps_xc[n(r)] * n(r)
# eps_xc = n(r) ** (1/3) * (-3/4) * (np.pi/3)**(1/3)

# Constants
ATOMS  = [["He",2,2,[5.0,5.0,5.0]]] # Symbol, Z, z, position (a.u.)
E_CUT  = 50.0  # Maximum included plane wave energy in a.u.
a      = np.array([10.0,10.0,10.0]) # Lattice Constants (Real Space, a.u.)
VOLUME = np.prod(a)

PROPERTIES = {}
PROPERTIES["ATOMS"]  = ATOMS
PROPERTIES["E_CUT"]  = E_CUT
PROPERTIES["a"]      = a
PROPERTIES["VOLUME"] = VOLUME

PROPERTIES = get_G_GRID( PROPERTIES )

"""
Kohn-Sham Energy in Plane Wave Basis
E_KS  = E_KIN + E_EXT + E_HAR + E_XC + E_RR
E_KIN = \sum_{k} \sum_{n} \sum_{G} |G + k|^2 |c_{mk}(G)|^2
E_EXT = \sum_{G!=0} V_ext(G) n(G)
E_HAR = \sum_{G!=0} |n(G)|^2 / |G|^2
E_XC  = \int dr n(r) eps_xc( n(r) )
E_RR  = \sum_{I!=I'} ZZ'/|R-R'|

c_{mk}(G) = Plane wave coefficient
G = Plane wave basis vector
k = K-point
n = band label
"""

G_GRID = PROPERTIES["G_GRID"]
NBASIS = PROPERTIES["NBASIS"]

c_G  = np.random.random( size=PROPERTIES["NBASIS"] ) + 0j # Start with random coefficients
c_G += 1j * np.random.random( size=PROPERTIES["NBASIS"] ) # Start with random coefficients
c_G  = c_G / np.linalg.norm(c_G)
PROPERTIES["c_G"] = c_G
print( "NORM:", np.linalg.norm(c_G) )


#def get_Hamiltonian( PROPERTIES ):


E0 = get_energy( PROPERTIES )

print ( E0 )

#for step in range( 100 ):
#    E1 = 
