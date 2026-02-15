"""
Schémas numériques d'ordre 1 et 2 pour la diffusion radiale.
"""
import numpy as np

from mesh_and_parameters import ProblemParameters, create_mesh


def solve_scheme_1(param: ProblemParameters, n_profile: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Fonction calculant la concentration du profil avec un schéma d'ordre 1.
    Retourne le maillage et un array contenant les concentrations aux nœuds résolu numériquement.

    :param param: Data class contenant les paramètres de la simulation [ProblemParameters]
    :param n_profile: Nombre de nœuds [int]

    :return: r_mesh, c_array [tuple[np.ndarray, np.ndarray]]
    """
    r_mesh, dr = create_mesh(param.r, n_profile)

    a = np.zeros((n_profile, n_profile), dtype=np.float64)  # LHS/influence matrix
    b = np.zeros(n_profile, dtype=np.float64)                     # RHS

    # symétrie, pas de flux à r=0
    a[0, 0] = 1.0
    a[0, 1] = -1.0
    b[0]    = 0.0

    for i in range(1, n_profile-1):
        ri = r_mesh[i]

        a[i, i-1] = 1.0 / dr**2
        a[i, i]   = -2.0 / dr**2 - 1.0 / (ri * dr)
        a[i, i+1] = 1.0 / dr**2 + 1.0 / (ri * dr)

        b[i] = param.s / param.d_eff

    # condition de Dirichlet
    a[n_profile - 1, :]             = 0.0
    a[n_profile - 1, n_profile - 1] = 1.0
    b[n_profile - 1]                = param.c_e

    c_array = np.linalg.solve(a.astype(np.float64), b.astype(np.float64))
    return r_mesh, c_array


def solve_scheme_2(param: ProblemParameters, n_profile: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Fonction calculant la concentration du profil avec un schéma d'ordre 2.
    Retourne le maillage et un array contenant les concentrations aux nœuds résolu numériquement.

    :param param: Data class contenant les paramètres de la simulation [ProblemParameters]
    :param n_profile: Nombre de nœuds [int]

    :return: r_mesh, c_array [tuple[np.ndarray, np.ndarray]]
    """
    r_mesh, dr = create_mesh(param.r, n_profile)

    a = np.zeros((n_profile, n_profile), dtype=np.float64)  # LHS/influence matrix
    b = np.zeros(n_profile, dtype=np.float64)                     # RHS

    # symétrie, pas de flux à r=0
    a[0, 0] = 1.0
    a[0, 1] = -1.0
    b[0]    = 0.0

    for i in range(1, n_profile-1):
        ri = r_mesh[i]

        a[i, i-1]   = (1.0 / dr**2) - (1.0 / (2.0 * dr * ri))
        a[i, i]     = -2.0 / dr**2
        a[i, i + 1] = (1.0 / dr**2) + (1.0 / (2.0 * dr * ri))

        b[i] = param.s / param.d_eff

    # condition de Dirichlet
    a[n_profile - 1, :]             = 0.0
    a[n_profile - 1, n_profile - 1] = 1.0
    b[n_profile - 1]                = param.c_e

    c_array = np.linalg.solve(a.astype(np.float64), b.astype(np.float64))
    return r_mesh, c_array
