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

def plot_profiles(R, D_eff, S, C_e, N_profile):
    """
    Trace C(r) à l'état stationnaire (schéma 1, schéma 2) vs analytique.
    """
    r1, C1 = solve_scheme_1(R, D_eff, S, C_e, N_profile)
    r2, C2 = solve_scheme_2(R, D_eff, S, C_e, N_profile)

    C_exact = analytic_solution(r1, R, D_eff, S, C_e)

    plt.figure(dpi=250)
    plt.plot(r1, C_exact, linewidth=2, label="Analytique ")
    plt.plot(r1, C1, "--", linewidth=2, label=f"Schéma 1 (N={N_profile})")
    plt.plot(r2, C2, "-.", linewidth=2, label=f"Schéma 2 (N={N_profile})")
    plt.xlabel("r [m]")
    plt.ylabel("C [mol/m³]")
    plt.title("Profil de concentration à l’état stationnaire")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

