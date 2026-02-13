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
    '''
    Docstring for plot_profiles

    Trace C(r) les profil de concentration 
    
    :param R: Rayon
    :param D_eff: Coeff de diffusion
    :param S: quantite de sel
    :param C_e: concentration a la surface
    :param N_profile: Nombre de noeuds
    '''

    r1, C1 = solve_scheme_1(R, D_eff, S, C_e, N_profile)
    r2, C2 = solve_scheme_2(R, D_eff, S, C_e, N_profile)

    C_exact = analytic_solution(r1, R, D_eff, S, C_e)

    # Superpose les 3 courbes
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

def plot_error_norms(R, D_eff, S, C_e, N_list):
    """
    Calcule et trace L1, L2, Linf sur un même graphique (pour chaque schéma)
    en fonction de h = dr (échelle log-log).
    """
    h = []
    # Schéma 1
    L1_1, L2_1, Linf_1 = [], [], []
    # Schéma 2
    L1_2, L2_2, Linf_2 = [], [], []

    for N in N_list:
        # Schéma 1
        r1, C1 = solve_scheme_1(R, D_eff, S, C_e, N)
        dr1 = r1[1] - r1[0]
        Cex1 = analytic_solution(r1, R, D_eff, S, C_e)
        a, b, c = norms_L1_L2_Linf(C1, Cex1, dr1)
        L1_1.append(a); L2_1.append(b); Linf_1.append(c)

        # Schéma 2
        r2, C2 = solve_scheme_2(R, D_eff, S, C_e, N)
        dr2 = r2[1] - r2[0]
        Cex2 = analytic_solution(r2, R, D_eff, S, C_e)
        a, b, c = norms_L1_L2_Linf(C2, Cex2, dr2)
        L1_2.append(a); L2_2.append(b); Linf_2.append(c)

        h.append(dr1)

        # Print des erreurs
        print(f"N={N:4d}  h={dr1:.3e} | "
              f"S1: L1={L1_1[-1]:.3e}, L2={L2_1[-1]:.3e}, Linf={Linf_1[-1]:.3e} | "
              f"S2: L1={L1_2[-1]:.3e}, L2={L2_2[-1]:.3e}, Linf={Linf_2[-1]:.3e}")


    h = np.array(h)

    # Plot log-log : erreurs vs h
    plt.figure(dpi=170)
    plt.loglog(h, L1_1, "o--", linewidth=2, label="Schéma 1 - L1")
    plt.loglog(h, L2_1, "s--", linewidth=2, label="Schéma 1 - L2")
    plt.loglog(h, Linf_1, "^-", linewidth=2, label="Schéma 1 - Linf")

    plt.loglog(h, L1_2, "o-.", linewidth=2, label="Schéma 2 - L1")
    plt.loglog(h, L2_2, "s-.", linewidth=2, label="Schéma 2 - L2")
    plt.loglog(h, Linf_2, "^-.", linewidth=2, label="Schéma 2 - Linf")

    # plt.gca().invert_xaxis()  
    plt.xlabel("Pas de maille h = dr [m]")
    plt.ylabel("Norme de l’erreur")
    plt.title("Vérification: erreurs L1, L2 et L_infini vs dr")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()

    Nvals = np.array(N_list, dtype=float)

    plt.figure(dpi=170)
    plt.loglog(Nvals, L1_1, "o--", linewidth=2, label="Schéma 1 - L1")
    plt.loglog(Nvals, L2_1, "s--", linewidth=2, label="Schéma 1 - L2")
    plt.loglog(Nvals, Linf_1, "^-", linewidth=2, label="Schéma 1 - Linf")

    plt.loglog(Nvals, L1_2, "o-.", linewidth=2, label="Schéma 2 - L1")
    plt.loglog(Nvals, L2_2, "s-.", linewidth=2, label="Schéma 2 - L2")
    plt.loglog(Nvals, Linf_2, "^-.", linewidth=2, label="Schéma 2 - Linf")

    plt.xlabel("Nombre de noeuds N")
    plt.ylabel("Norme de l’erreur")
    plt.title("Vérification: erreurs L1, L2 et L∞ vs N (log-log)")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()