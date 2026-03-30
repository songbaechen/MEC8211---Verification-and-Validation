import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from problem_definition.lbm_devoir3 import Generate_sample, LBM
from problem_definition.problem_definition import SampleParameters, ProblemParameters


def richardson_convergence(
        srq_list: list[float],
        dx_list: list[float],
        p_est: float = 2.0,
        max_n_iter: int = 100,
        tol: float = 5e-3
    ) -> float:
    """
    Estimation itérative de l'ordre de convergence observé.
    Convention: listes ordonnées du plus grossier au plus fin.
    -1 = finest, -2 = medium, -3 = coarsest
    """
    r12 = dx_list[-2] / dx_list[-1]   # medium / finest  (> 1)
    r23 = dx_list[-3] / dx_list[-2]   # coarsest / medium (> 1)

    f1 = srq_list[-1]   # finest
    f2 = srq_list[-2]   # medium
    f3 = srq_list[-3]   # coarsest

    p_k = p_est

    for i in range(max_n_iter):
        up_term = (r12 ** p_k - 1) * (f3 - f2) / (f2 - f1) + r12 ** p_k
        if up_term <= 0:
            # Fallback: utiliser la formule directe (ratio constant)
            if abs(f2 - f1) > 1e-30:
                p_kp1 = np.log(abs((f3 - f2) / (f2 - f1))) / np.log(r12)
            else:
                p_kp1 = p_k
        else:
            p_kp1 = np.log(up_term) / np.log(r12 * r23)

        if abs(p_kp1 - p_k) < tol:
            return p_kp1

        p_k = p_kp1

    print(f"WARNING: Richardson did not converge, returning last p = {p_k:.4f}")
    return p_k


def gci_factor(
        f_fine: float,
        f_coarse: float,
        r: float,
        p_hat: float,
        p_formal: float = 2.0
    ) -> float:
    """
    Calcul du GCI (Grid Convergence Index) entre deux maillages.
    f_fine:   SRQ sur le maillage fin
    f_coarse: SRQ sur le maillage grossier
    r:        ratio de raffinement (dx_coarse / dx_fine) > 1
    p_hat:    ordre de convergence observé
    p_formal: ordre formel de la méthode
    """
    p_ratio = abs((p_hat - p_formal) / p_formal)

    if p_ratio > 0.1:
        f_s = 3.0
        p = min(max(0.5, p_hat), p_formal)
    else:
        f_s = 1.25
        p = p_formal

    return f_s * abs(f_coarse - f_fine) / (r ** p - 1)


def convergence_analysis(
        initial_dx: float,
        initial_nx: int,
        ratio: float,
        n_steps: int,
        sample_parameters: SampleParameters,
        problem_parameters: ProblemParameters
    ) -> tuple[list[float], list[float], list[int]]:
    """
    Analyse de convergence spatiale.
    Garde Nx * dx = constant en utilisant un ratio de raffinement.

    Retourne: (dx_list, k_list, nx_list)  ordonnés du plus grossier au plus fin
    """
    dx_list = []
    k_list = []
    nx_list = []

    for i in range(n_steps):
        factor = ratio ** i
        nx = int(initial_nx * factor)
        dx = initial_dx / factor       # nx * dx = initial_nx * initial_dx = constant

        print(f"  Step {i+1}/{n_steps}: nx = {nx}, dx = {dx:.2e} m, "
              f"Nx*dx = {nx*dx:.2e} m")

        d_equivalent = Generate_sample(
            seed=sample_parameters.seed,
            filename=sample_parameters.filename,
            mean_d=sample_parameters.mean_d,
            std_d=sample_parameters.std_d,
            poro=sample_parameters.poro,
            nx=nx,
            dx=dx
        )

        k = LBM(
            filename=problem_parameters.filename,
            NX=nx,
            deltaP=problem_parameters.deltaP,
            dx=dx,
            d_equivalent=d_equivalent
        )

        dx_list.append(dx)
        k_list.append(k)
        nx_list.append(nx)
        print(f"    -> k = {k:.6e}")

    return dx_list, k_list, nx_list


def plot_convergence_loglog(dx_list, k_list, p_hat, p_formal=2.0, save_path="convergence_loglog.png"):
    """
    Graphique log-log: k (perméabilité) vs dx
    Avec droite de pente théorique (ordre formel) et observée.
    """
    dx_arr = np.array(dx_list)
    k_arr = np.array(k_list)

    # Richardson extrapolated value (from the 3 finest grids)
    f1 = k_arr[-1]   # finest
    f2 = k_arr[-2]
    r = dx_arr[-2] / dx_arr[-1]
    k_exact = f1 + (f1 - f2) / (r ** p_hat - 1)

    # Erreur relative par rapport à la valeur extrapolée
    err = np.abs(k_arr - k_exact)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Subplot 1: k vs dx ---
    ax1 = axes[0]
    ax1.loglog(dx_arr, k_arr, 'o-', color='#2563eb', markersize=8, linewidth=1.5,
               label='$k$ (LBM)')
    ax1.axhline(y=k_exact, color='#dc2626', linestyle='--', linewidth=1,
                label=f'$k_{{exact}}$ Richardson = {k_exact:.4e}')
    ax1.set_xlabel(r'$\Delta x$ (m)', fontsize=12)
    ax1.set_ylabel(r'Perméabilité $k$', fontsize=12)
    ax1.set_title('Convergence spatiale — SRQ vs $\\Delta x$', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, which='both', alpha=0.3)

    # --- Subplot 2: erreur vs dx (log-log) avec pentes ---
    ax2 = axes[1]
    # Ne tracer que les points où l'erreur > 0
    mask = err > 0
    ax2.loglog(dx_arr[mask], err[mask], 's-', color='#2563eb', markersize=8,
               linewidth=1.5, label=f'Erreur observée')

    # Droite de pente p_hat (observée)
    if np.sum(mask) >= 2:
        dx_ref = dx_arr[mask]
        C_obs = err[mask][-1] / dx_ref[-1] ** p_hat
        err_fit_obs = C_obs * dx_ref ** p_hat
        ax2.loglog(dx_ref, err_fit_obs, '--', color='#16a34a', linewidth=1.5,
                   label=f'Pente observée $\\hat{{p}}$ = {p_hat:.2f}')

    # Droite de pente formelle
    if np.sum(mask) >= 2:
        C_for = err[mask][-1] / dx_ref[-1] ** p_formal
        err_fit_for = C_for * dx_ref ** p_formal
        ax2.loglog(dx_ref, err_fit_for, ':', color='#9333ea', linewidth=1.5,
                   label=f'Pente formelle $p$ = {p_formal:.1f}')

    ax2.set_xlabel(r'$\Delta x$ (m)', fontsize=12)
    ax2.set_ylabel(r'$|k - k_{exact}|$', fontsize=12)
    ax2.set_title('Ordre de convergence (log-log)', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"  [Saved] {save_path}")


def plot_gci(dx_list, k_list, gci_list, k_exact, save_path="gci_barres_erreur.png"):
    """
    Graphique GCI: barres d'erreur sur chaque maillage.
    gci_list[i] = GCI associé au maillage i (0 pour le plus grossier si non calculable).
    """
    dx_arr = np.array(dx_list)
    k_arr = np.array(k_list)
    gci_arr = np.array(gci_list)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.errorbar(dx_arr, k_arr, yerr=gci_arr, fmt='o', color='#2563eb',
                markersize=8, capsize=6, capthick=1.5, elinewidth=1.5,
                label='$k$ ± GCI')

    ax.axhline(y=k_exact, color='#dc2626', linestyle='--', linewidth=1.2,
               label=f'$k_{{exact}}$ Richardson = {k_exact:.4e}')

    ax.set_xscale('log')
    ax.set_xlabel(r'$\Delta x$ (m)', fontsize=12)
    ax.set_ylabel(r'Perméabilité $k$', fontsize=12)
    ax.set_title('GCI — Barres d\'incertitude numérique', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    # Annoter chaque point
    for i, (dx, k, gci) in enumerate(zip(dx_arr, k_arr, gci_arr)):
        label_txt = f"GCI={gci:.2e}" if gci > 0 else ""
        ax.annotate(label_txt, (dx, k + gci), textcoords="offset points",
                    xytext=(5, 8), fontsize=8, color='#374151')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"  [Saved] {save_path}")


def print_summary(dx_list, k_list, p_hat, gci_list, k_exact):
    """Affiche un tableau récapitulatif dans la console."""
    print("\n" + "=" * 75)
    print("  RÉSUMÉ — Analyse de convergence spatiale")
    print("=" * 75)
    print(f"  {'Maillage':<10} {'dx (m)':<14} {'Nx':<8} {'k':<16} {'GCI':<14}")
    print("-" * 75)
    for i, (dx, k, gci) in enumerate(zip(dx_list, k_list, gci_list)):
        nx = int(round(dx_list[0] * (dx_list[0] / dx_list[-1]) / dx))  # approximation
        print(f"  {i+1:<10} {dx:<14.2e} {'-':<8} {k:<16.6e} {gci:<14.2e}")
    print("-" * 75)
    print(f"  Ordre observé (Richardson) : p_hat = {p_hat:.4f}")
    print(f"  Valeur extrapolée k_exact  : {k_exact:.6e}")
    print(f"  Incertitude numérique u_num (GCI fin) : {gci_list[-1]:.6e}")
    print("=" * 75)


# ============================================================================
#  MAIN
# ============================================================================
if __name__ == "__main__":
    filename = "convergence_space.tiff"

    # --- Paramètres de base ---
    nx_base = 100
    dx_base = 2e-6          # m
    ratio = 2               # facteur de raffinement
    n_steps = 4             # nombre de niveaux de maillage (min 3 pour Richardson)
    seed = 42               # seed fixe pour figer la géométrie

    print(f"Domain size constant: Nx*dx = {nx_base * dx_base:.2e} m")
    print(f"Ratio de raffinement: {ratio}")
    print(f"Nombre de niveaux: {n_steps}\n")

    sample_parameters = SampleParameters(
        seed=seed,
        filename=filename,
        mean_d=12.5,
        std_d=2.85,
        poro=0.9,
        nx=nx_base,
        dx=dx_base
    )
    problem_parameters = ProblemParameters(
        filename=filename,
        NX=nx_base,
        deltaP=0.1,
        dx=dx_base,
        d_equivalent=0.0        # temporaire, modifié dans la fonction
    )

    # --- 1. Lancer l'analyse de convergence ---
    print("=" * 50)
    print("  CONVERGENCE SPATIALE")
    print("=" * 50)
    dx_list, k_list, nx_list = convergence_analysis(
        initial_dx=dx_base,
        initial_nx=nx_base,
        ratio=ratio,
        n_steps=n_steps,
        sample_parameters=sample_parameters,
        problem_parameters=problem_parameters
    )

    # --- 2. Richardson: ordre observé (besoin de 3 points minimum) ---
    p_formal = 2.0
    p_hat = richardson_convergence(k_list, dx_list, p_est=p_formal)
    print(f"\nOrdre observé (Richardson itératif): p_hat = {p_hat:.4f}")
    print(f"Ordre formel: p_formal = {p_formal:.1f}")

    # Valeur extrapolée de Richardson
    f1 = k_list[-1]    # finest
    f2 = k_list[-2]    # medium
    r = dx_list[-2] / dx_list[-1]
    k_exact = f1 + (f1 - f2) / (r ** p_hat - 1)
    print(f"Valeur extrapolée (Richardson): k_exact = {k_exact:.6e}")

    # --- 3. GCI pour chaque paire de maillages ---
    gci_list = [0.0]    # pas de GCI pour le maillage le plus grossier
    for i in range(1, len(dx_list)):
        r_i = dx_list[i - 1] / dx_list[i]      # > 1
        gci_i = gci_factor(
            f_fine=k_list[i],
            f_coarse=k_list[i - 1],
            r=r_i,
            p_hat=p_hat,
            p_formal=p_formal
        )
        gci_list.append(gci_i)

    u_num = gci_list[-1]    # incertitude numérique = GCI du maillage le plus fin
    print(f"u_num (GCI finest mesh) = {u_num:.6e}")

    # --- 4. Vérification asymptotique du GCI ---
    if len(gci_list) >= 3:
        r_ref = dx_list[-3] / dx_list[-2]
        gci_ratio = gci_list[-2] / (r_ref ** p_hat * gci_list[-1]) if gci_list[-1] > 0 else float('inf')
        print(f"Vérification asymptotique GCI_coarse / (r^p * GCI_fine) = {gci_ratio:.4f}  (≈ 1.0 si convergé)")

    # --- 5. Tableau récapitulatif ---
    print_summary(dx_list, k_list, p_hat, gci_list, k_exact)

    # --- 6. Graphiques ---
    plot_convergence_loglog(dx_list, k_list, p_hat, p_formal,
                           save_path="convergence_loglog.png")

    plot_gci(dx_list, k_list, gci_list, k_exact,
             save_path="gci_barres_erreur.png")