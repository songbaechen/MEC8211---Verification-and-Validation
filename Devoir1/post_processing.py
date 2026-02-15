"""
Fonctions de post-traitement : normes d’erreur et tracés des profils de concentration.
"""
import numpy as np
import matplotlib.pyplot as plt

from mesh_and_parameters import ProblemParameters
from analytical_solution import analytic_solution
from finite_differences_schemes import solve_scheme_1, solve_scheme_2


def error_norms(c_num: np.ndarray, c_ref: np.ndarray, dr: np.ndarray)\
        -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Retourne les erreurs L1, L2 et L_inf.

    :param c_num : Concentration numérique [np.ndarray]
    :param c_ref : Concentration analytique [np.ndarray]
    :param dr : delta r, Espacement nœuds [np.ndarray]]

    :return: L1, L2, L_inf [tuple[np.ndarray, np.ndarray, np.ndarray]]
    """
    c_num = np.asarray(c_num, dtype=np.float64)
    c_ref = np.asarray(c_ref, dtype=np.float64)
    dr = np.asarray(dr, dtype=np.float64)

    e = c_num - c_ref

    l1 = np.sum(np.abs(e)) * dr
    l2 = np.sqrt(np.sum(e**2) * dr)
    l_inf = np.max(np.abs(e))
    return l1, l2, l_inf


def plot_profiles(param: ProblemParameters, n_profile, plot_1, plot_2):
    """
    Plot les profils de concentration en fonction de la position radiale C(r) [mol/m^3 ; m].

    Parameters
    :param param: Data class contenant les paramètres de la simulation [ProblemParameters]
    :param n_profile : Nombre de nœuds [int]
    :param plot_1: Option de plot le profil du schéma 1 [bool]
    :param plot_2: Option de plot le profil du schéma 2 [bool]

    :return: None
    """
    r1, c1 = solve_scheme_1(param, n_profile)
    c_exact = analytic_solution(param, r1)

    plt.figure(dpi=250)

    # analytique
    plt.plot(r1, c_exact, linewidth=2, label="Analytique ")

    # schéma 1
    if plot_1:
        plt.plot(r1, c1, "--", linewidth=2, label=f"Schéma 1 (N={n_profile})")

    # schéma 2
    if plot_2:
        r2, c2 = solve_scheme_2(param, n_profile)
        plt.plot(r2, c2, "-.", linewidth=2, label=f"Schéma 2 (N={n_profile})")

    # titres
    plt.xlabel("r [m]")
    plt.ylabel("C [mol/m³]")
    plt.title("Profil de concentration à l’état stationnaire")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def ordre_de_convergence(h: np.ndarray, erreur_l1: np.ndarray, erreur_l2: np.ndarray, erreur_l_inf: np.ndarray) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fonction calculant l'ordre de convergence avec les erreurs L1, L2 et L_inf.
        p_i = log(E_coarse / E_fine) / log(h_coarse / h_fine)

    :param h: Array contenant les delta r, distance entre les nœuds [np.ndarray[float]]
    :param erreur_l1: Array contenant les erreurs L1 [np.ndarray[float]]
    :param erreur_l2: Array contenant les erreurs L2 [np.ndarray[float]]
    :param erreur_l_inf: Array contenant les erreurs L_inf [np.ndarray[float]]

    :return: conv_l1, conv_l2, conv_l_inf [tuple[np.ndarray, np.ndarray, np.ndarray]]
    """
    h = np.asarray(h, dtype=float).copy()
    e1 = np.asarray(erreur_l1, dtype=float).copy()
    e2 = np.asarray(erreur_l2, dtype=float).copy()
    e_inf = np.asarray(erreur_l_inf, dtype=float).copy()

    if not (len(h) == len(e1) == len(e2) == len(e_inf)):
        raise ValueError("h, erreur_l1, erreur_l2, erreur_l_inf doivent avoir la même longueur.")
    if len(h) < 2:
        raise ValueError("Il faut au moins 2 maillages pour calculer un ordre de convergence.")

    # Trie par h décroissant (coarse -> fine) pour que (h[i-1] > h[i]) en général
    order = np.argsort(h)[::-1]
    h = h[order]
    e1, e2, e_inf = e1[order], e2[order], e_inf[order]

    conv_l1 = np.full(len(h), np.nan)
    conv_l2 = np.full(len(h), np.nan)
    conv_l_inf = np.full(len(h), np.nan)

    print("\nOrdres de convergence (entre i-1 -> i):")
    print(" i |   h_{i-1}     h_i   |   p(L1)    p(L2)   p(Linf)")
    print("-" * 62)

    for i in range(1, len(h)):
        denom = np.log(h[i-1] / h[i])
        p1 = np.log(e1[i-1] / e1[i]) / denom
        p2 = np.log(e2[i-1] / e2[i]) / denom
        pinf = np.log(e_inf[i-1] / e_inf[i]) / denom

        conv_l1[i] = p1
        conv_l2[i] = p2
        conv_l_inf[i] = pinf

        print(f"{i:2d} | {h[i-1]:.3e}  {h[i]:.3e} | {p1:8.3f}  {p2:8.3f}  {pinf:8.3f}")

    return conv_l1, conv_l2, conv_l_inf


def plot_error_norms(param: ProblemParameters, n_profil_list: list[int],
                     plot_1: bool, plot_2: bool):
    """
    Calcule et trace L1, L2, L_inf en fonction de h = dr (échelle log-log).

    :param param: Data class contenant les paramètres de la simulation [ProblemParameters]
    :param n_profil_list: list contenant les différents nombres de nœuds (discrétisation)
                          [list[int]]
    :param plot_1: Option de plot les erreurs du schéma 1 [bool]
    :param plot_2: Option de plot les erreurs du schéma 2 [bool]

    :return: None
    """
    if not (plot_1 or plot_2):
        print("Demande de plot schéma 1 = False et demande de plot schéma 2 = False")

    h = []
    l1_1, l2_1, l_inf_1 = [], [], []    # schéma 1
    l1_2, l2_2, l_inf_2 = [], [], []    # schéma 2

    # Résolution du problème pour les différents niveaux de discrétisation
    for n_profil_tmp in n_profil_list:
        # Schéma 1
        r1, c1 = solve_scheme_1(param, n_profil_tmp)
        dr1 = r1[1] - r1[0]
        c_analytique_1 = analytic_solution(param, r1)
        a, b, c = error_norms(c1, c_analytique_1, dr1)
        l1_1.append(a)
        l2_1.append(b)
        l_inf_1.append(c)

        # Schéma 2
        r2, c2 = solve_scheme_2(param, n_profil_tmp)
        dr2 = r2[1] - r2[0]
        c_analytique_2 = analytic_solution(param, r2)
        a, b, c = error_norms(c2, c_analytique_2, dr2)
        l1_2.append(a)
        l2_2.append(b)
        l_inf_2.append(c)

        h.append(dr1)
        # Print des erreurs
        # print(f"N={n_profil_tmp:4d}  h={dr1:.3e} | "
        #       f"S1: L1={l1_1[-1]:.3e}, L2={l2_1[-1]:.3e}, Linf={l_inf_1[-1]:.3e} | "
        #       f"S2: L1={l1_2[-1]:.3e}, L2={l2_2[-1]:.3e}, Linf={l_inf_2[-1]:.3e}")

    h = np.array(h)
    ordre_de_convergence(h=h, erreur_l1=np.array(l1_1), erreur_l2=np.array(l2_1), erreur_l_inf=np.array(l_inf_1))
    ordre_de_convergence(h=h, erreur_l1=np.array(l1_2), erreur_l2=np.array(l2_2), erreur_l_inf=np.array(l_inf_2))

    # Plot log-log : erreurs vs h
    plt.figure(dpi=170)
    if plot_1:
        plt.loglog(h, l1_1, "o--", linewidth=2, label="Schéma 1 - L1")
        plt.loglog(h, l2_1, "s--", linewidth=2, label="Schéma 1 - L2")
        plt.loglog(h, l_inf_1, "^-", linewidth=2, label="Schéma 1 - Linf")
    if plot_2:
        plt.loglog(h, l1_2, "o-.", linewidth=2, label="Schéma 2 - L1")
        plt.loglog(h, l2_2, "s-.", linewidth=2, label="Schéma 2 - L2")
        plt.loglog(h, l_inf_2, "^-.", linewidth=2, label="Schéma 2 - Linf")

    # plt.gca().invert_xaxis()
    plt.xlabel("Pas de maille h = dr [m]")
    plt.ylabel("Norme de l’erreur")
    plt.title("Vérification: erreurs L1, L2 et L_infini vs dr")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()

    nvals = np.array(n_profil_list, dtype=float)

    plt.figure(dpi=170)
    if plot_1:
        plt.loglog(nvals, l1_1, "o--", linewidth=2, label="Schéma 1 - L1")
        plt.loglog(nvals, l2_1, "s--", linewidth=2, label="Schéma 1 - L2")
        plt.loglog(nvals, l_inf_1, "^-", linewidth=2, label="Schéma 1 - Linf")

    if plot_2:
        plt.loglog(nvals, l1_2, "o-.", linewidth=2, label="Schéma 2 - L1")
        plt.loglog(nvals, l2_2, "s-.", linewidth=2, label="Schéma 2 - L2")
        plt.loglog(nvals, l_inf_2, "^-.", linewidth=2, label="Schéma 2 - Linf")

    plt.xlabel("Nombre de noeuds N")
    plt.ylabel("Norme de l’erreur")
    plt.title("Vérification: erreurs L1, L2 et L∞ vs N (log-log)")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()
