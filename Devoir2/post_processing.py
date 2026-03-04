"""
Fonctions de post-traitement : normes d’erreur et tracés des profils de concentration.
"""
import numpy as np
import matplotlib.pyplot as plt

from mesh_and_parameters import ProblemParameters
from finite_differences_schemes import solve_unsteady_scheme
from mms_solution import*


def error_norms(c_num_hist: np.ndarray, 
    r_mesh: np.ndarray, 
    time_array: np.ndarray,
    problem: ProblemParameters,
    mms: MMSParams,) -> tuple[float, float, float]:
    """
    Retourne les erreurs L1, L2 et L_inf (espace x temps)
    """
    num_time_steps, num_nodes = c_num_hist.shape
    dr = float(r_mesh[1] - r_mesh[0])
    dt = float(time_array[1] - time_array[0])

    error = np.zeros_like(c_num_hist, dtype=float)

    for n_idx in range(num_time_steps):
        t_n = float(time_array[n_idx])
        for i_idx in range(num_nodes):
            r_i = float(r_mesh[i_idx])
            c_exact = mms_function(r_i, t_n, float(problem.r), mms)
            error[n_idx, i_idx] = c_num_hist[n_idx, i_idx] - c_exact

    l1_norm = float(np.sum(np.abs(error)) * dr * dt)
    l2_norm = float(np.sqrt(np.sum(error**2) * dr * dt))
    linf_norm = float(np.max(np.abs(error)))

    return l1_norm, l2_norm, linf_norm
