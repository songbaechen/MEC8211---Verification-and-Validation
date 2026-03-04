"""
Schémas numériques d'ordre 2 pour la diffusion radiale.
"""
import numpy as np

from mesh_and_parameters import ProblemParameters, create_mesh

def solve_unsteady_scheme(param: ProblemParameters, n_profile: int):
    """
    Fonction calculant la concentration du profil avec un schéma d'ordre 2.
    Retourne le maillage et un array contenant les concentrations aux nœuds résolu numériquement.

    :param param: Data class contenant les paramètres de la simulation [ProblemParameters]
    :param n_profile: Nombre de nœuds [int]

    :return: r_mesh, c_array [tuple[np.ndarray, np.ndarray]]
    """
    r_mesh, dr = create_mesh(param.r, n_profile)

    D = param.d_eff
    k = param.k
    dt = param.dt
    t_final = param.t_final

    Nt = int(np.ceil(t_final / dt)) + 1
    t = np.linspace(0.0, dt*(Nt-1), Nt)

    # C(r,0) = 0
    Cn = np.zeros(n_profile, dtype = np.float64)

    C_hist = np.zeros((Nt, n_profile), dtype=np.float64)
    C_hist[0, :] = Cn.copy()

    for n in range(Nt - 1): 

        a = np.zeros((n_profile, n_profile), dtype=np.float64)  
        b = np.zeros(n_profile, dtype=np.float64)                    

        # Boundary conditions au centre
        a[0, 0] = -3.0
        a[0, 1] =  4.0
        a[0, 2] = -1.0
        b[0] = 0.0

        # Noeuds internes
        for i in range(1, n_profile -1): 
            
            ri = r_mesh[i]

            L_i_moins_1 = (1.0 / dr**2) - (1.0 / (2.0 * dr * ri))
            L_i = -2.0 / dr**2
            L_i_plus_1 = (1.0 / dr**2) + (1.0 / (2.0 * dr * ri))

            a[i, i-1] = -D * L_i_moins_1
            a[i, i] = (1.0 / dt + k) - D * L_i
            a[i, i + 1] = -D * L_i_plus_1

            b[i] = (1.0 / dt) * Cn[i]

        # condition de Dirichlet au bord
        a[n_profile - 1, :]             = 0.0
        a[n_profile - 1, n_profile - 1] = 1.0
        b[n_profile - 1]                = param.c_e

        Cn_plus_1 = np.linalg(a, b)

        C_hist[n+1, :] = Cn_plus_1
        Cn = Cn_plus_1

    return r_mesh, t , C_hist
