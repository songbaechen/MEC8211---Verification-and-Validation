import numpy as np


def create_mesh(R, N): 
    '''
    Docstring for create_mesh
    
    Il faut créer une grille radfiale uniforme en 1D [0, R] 

    :param R: Rayon total
    :param N: Nombre de noeuds
    '''
    if N  < 3: 
        raise ValueError("Il faut avoir au moins 3 noeuds pour résoudre le problème")
    
    r = np.linspace(0.0, R, N)
    dr = r[1] - r[0]

    return r, dr

def solve_scheme_1(R, D_eff, S, C_e, N) : 

    r, dr = create_mesh(R, N)

    A = np.zeros((N, N))
    b = np.zeros(N)

    A[0, 0] = 1.0
    A[0, 1] = -1.0
    b[0] = 0.0

    for i in range(1, N-1): 
        ri = r[i]

        A[i, i-1] = 1.0 / dr**2
        A[i, i]   = -2.0 / dr**2 - 1.0 / (ri * dr)
        A[i, i+1] = 1.0 / dr**2 + 1.0 / (ri * dr)

        b[i] = S / D_eff

    A[N - 1, :] = 0.0
    A[N - 1, N - 1] = 1.0
    b[N - 1] = C_e

    C = np.linalg.solve(A, b)
    return r, C

def solve_scheme_2(R, D_eff, S, C_e, N): 

    r, dr = create_mesh(R, N)

    A = np.zeros((N, N))
    b = np.zeros(N)

    A[0, 0] = 1.0
    A[0, 1] = -1.0
    b[0] = 0.0

    for i in range(1, N-1): 
        ri = r[i]

        A[i, i-1] = (1.0 / dr**2) - (1.0 / (2.0 * dr * ri))
        A[i, i]     = -2.0 / dr**2
        A[i, i + 1] = (1.0 / dr**2) + (1.0 / (2.0 * dr * ri))

        b[i] = S / D_eff


    A[N - 1, :] = 0.0
    A[N - 1, N - 1] = 1.0
    b[N - 1] = C_e

    C = np.linalg.solve(A, b)
    return r, C

