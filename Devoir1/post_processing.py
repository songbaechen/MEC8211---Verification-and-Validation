import numpy as np
import matplotlib.pyplot as plt

from analytical_solution import analytic_solution
from finite_differences_schemes import solve_scheme_1, solve_scheme_2  


def norms_L1_L2_Linf(C_num, C_ref, dr):

    e = C_num - C_ref
    L1 = np.sum(np.abs(e)) * dr
    L2 = np.sqrt(np.sum(e**2) * dr)
    Linf = np.max(np.abs(e))
    return L1, L2, Linf


