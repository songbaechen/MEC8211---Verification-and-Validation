"""
Calcul de la solution analytique du profil de concentration pour le problème de diffusion radiale.
"""
import numpy as np
from mesh_and_parameters import ProblemParameters


def analytic_solution(param: ProblemParameters, r_array: np.ndarray) -> np.ndarray:
    """
    Fonction qui retourne la solution analytique.

    :param param: Data class contenant les paramètres de la simulation [ProblemParameters]
    :param r_array: Array contenant le maillage [np.ndarray]

    :return: solution [np.ndarray]
    """
    r_array = np.asarray(r_array, dtype=np.float64)
    sol = param.c_e + (param.s / (4 * param.d_eff) ) * (r_array**2 - param.r**2)

    return sol
