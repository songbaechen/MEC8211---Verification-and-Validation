"""
Schémas numériques d'ordre 2 pour la diffusion radiale.
"""
import numpy as np

from mesh_and_parameters import ProblemParameters, create_mesh
from mms_solution import MMSParams, mms_function, dirichlet_bord_mms, source_term_MMS

def solve_unsteady_scheme(param: ProblemParameters, n_profile: int, mms: MMSParams | None = None):
    """
    Fonction calculant la concentration du profil avec un schéma d'ordre 2.
    Retourne le maillage et un array contenant les concentrations aux nœuds résolu numériquement.

    :param param: Data class contenant les paramètres de la simulation [ProblemParameters]
    :param n_profile: Nombre de nœuds [int]

    :return: r_mesh, c_array [tuple[np.ndarray, np.ndarray]]
    """
    r_mesh, dr = create_mesh(param.r, n_profile)

    D = float(param.d_eff)
    k = float(param.k)
    dt = float(param.dt)
    t_final = float(param.t_final)

    Nt = int(np.ceil(t_final / dt)) + 1
    t = np.linspace(0.0, dt*(Nt-1), Nt)

    # C(r,0) = 0
    if mms is None:
        Cn = np.zeros(n_profile, dtype=np.float64)
    else:
        Cn = np.array(
        [mms_function(float(r_i), 0.0, float(param.r), mms) for r_i in r_mesh],
        dtype=np.float64
    )

    C_hist = np.zeros((Nt, n_profile), dtype=np.float64)
    C_hist[0, :] = Cn.copy()

    for n in range(Nt - 1): 

        t_np1 = float(t[n + 1])
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
            a[i, i] = (1.0/dt + k) - D*L_i
            a[i, i + 1] = -D * L_i_plus_1

            b[i] = (1.0 / dt) * Cn[i]

            # Source MMS
            if mms is not None:
                b[i] += source_term_MMS(
                    ri,
                    t_np1,
                    float(param.r),
                    D,
                    k,
                    mms,
                )

        # condition de Dirichlet au bord
        a[n_profile - 1, :]             = 0.0
        a[n_profile - 1, n_profile - 1] = 1.0
        if mms is None:
            b[n_profile - 1] = float(param.c_e)
        else:
            b[n_profile - 1] = dirichlet_bord_mms(t_np1, float(param.r), mms)

        Cn_plus_1 = np.linalg.solve(a, b)

        C_hist[n+1, :] = Cn_plus_1
        Cn = Cn_plus_1

    return r_mesh, t , C_hist
