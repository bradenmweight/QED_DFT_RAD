import numpy as np

def lda_x(n):
    f = -3 / 4 * (3 / (2 * np.pi))**(2 / 3)
    rs = (3 / (4 * np.pi * n))**(1 / 3)

    ex = f / rs
    vx = 4 / 3 * ex
    return ex, vx


def coulomb(atoms, op):

    with np.errstate(divide='ignore', invalid='ignore'):
        Vcoul = -4 * np.pi * atoms.Z[0] / atoms.G2
    Vcoul[0] = 0
    return op.J(Vcoul * atoms.Sf)