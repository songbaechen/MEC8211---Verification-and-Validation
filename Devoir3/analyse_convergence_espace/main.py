"""
Question A — Détermination de l'erreur numérique de la perméabilité calculée par LBM
                en fonction du maillage.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import linregress

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
from problem_definition.lbm_devoir3 import Generate_sample, LBM


# ==============================================================================
# PARAMÈTRES
# ==============================================================================

DELTA_P      = 0.1
PORO         = 0.9
MEAN_FIBER_D = 12.5
ECART_TYPE_D = 2.85

NX_LIST = [50, 75, 100, 150, 200]
DX_LIST = [4e-6, 200 / 75 * 1e-6, 2e-6, 200 / 150 * 1e-6, 1e-6]

SEEDS = [101, 102, 103, 104, 105, 106, 107]

PF = 2   # ordre de convergence théorique LBM
R  = 2   # facteur de raffinement entre les deux maillages les plus fins

SEED_VISU = 101
NX_VISU   = [50, 100, 200]
DX_VISU   = [4e-6, 2e-6, 1e-6]


# ==============================================================================
# FONCTIONS
# ==============================================================================

def run_convergence_study():
    """Exécute les simulations LBM sur tous les maillages × seeds."""
    k_means, k_stds = [], []
    for nx, dx in zip(NX_LIST, DX_LIST):
        k_seeds = []
        for seed in SEEDS:
            filename = f"fiber_mat_NX{nx}_seed{seed}.tiff"
            d_eq = Generate_sample(seed, filename, MEAN_FIBER_D, ECART_TYPE_D, PORO, nx, dx, plotting=False)
            k = LBM(filename, nx, DELTA_P, dx, d_eq, plotting=False)
            k_seeds.append(k)
        k_means.append(np.mean(k_seeds))
        k_stds.append(np.std(k_seeds, ddof=1))
    return np.array(k_means), np.array(k_stds), np.array(DX_LIST)


def compute_apparent_order(k_means, dx_array):
    """Régression log-log de l'erreur relative → ordre apparent p."""
    k_fin     = k_means[-1]
    err_rel   = np.abs(k_fin - k_means[:-1]) / k_fin
    dx_coarse = dx_array[:-1]

    slope, intercept, r_value, _, _ = linregress(np.log(dx_coarse), np.log(err_rel))
    return slope, intercept, r_value, err_rel, dx_coarse


def compute_numerical_uncertainty(k_means, p_apparent):
    """Calcul de k_num et u_num selon le rapport |p - pf| / pf."""
    rapport = abs((p_apparent - PF) / PF)

    if rapport < 0.01:
        k_num = k_means[-1] + (k_means[-1] - k_means[-2]) / (R**PF - 1)
        u_num = 0.0
        print(f"  Branche <1% : extrapolation de Richardson")
        print(f"  k_num = k_extrap = {k_num:.4f} µm²  (pas d'incertitude)")
    elif rapport <= 0.10:
        GCI   = (1.25 / (R**PF - 1)) * abs(k_means[-2] - k_means[-1])
        u_num = GCI / 2
        k_num = k_means[-1]
        print(f"  Branche ≤10% : GCI (Fs=1.25, p=pf={PF})")
        print(f"  k_num = {k_num:.4f} µm²,  u_num = {u_num:.4f} µm²")
    else:
        p_exp = min(max(0.5, p_apparent), PF)
        GCI   = (3 / (R**p_exp - 1)) * abs(k_means[-2] - k_means[-1])
        u_num = GCI / 2
        k_num = k_means[-1]
        print(f"  Branche >10% : GCI (Fs=3, p_exp={p_exp:.2f})")
        print(f"  k_num = {k_num:.4f} µm²,  u_num = {u_num:.4f} µm²")

    return k_num, u_num, rapport


def plot_convergence(ax, dx_coarse, err_rel, intercept, p_apparent):
    """Erreur relative log-log + droite de régression."""
    dx_fit  = np.linspace(dx_coarse.min() * 0.8, dx_coarse.max() * 1.2, 100)
    err_fit = np.exp(intercept) * dx_fit**p_apparent

    ax.loglog(dx_coarse * 1e6, err_rel * 100, "bo", ms=8, lw=2, label="Erreur relative")
    ax.loglog(dx_fit * 1e6, err_fit * 100, "r--", lw=2,
              label=f"Régression : pente = {p_apparent:.2f}")
    ax.set(xlabel="dx (µm)", ylabel="Erreur relative (%)",
           title="Convergence en maillage (log-log)")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)


def plot_permeability(ax, dx_array, k_means, k_stds, u_num):
    """k moyen ± écart-type en fonction de dx."""
    k_fin = k_means[-1]
    ax.errorbar(dx_array, k_means, yerr=k_stds, fmt="bo-", ms=8, lw=2, capsize=5,
                label="k moyen ± écart-type (seeds)")
    ax.axhline(k_fin, color="r", ls="--", lw=1.5, label=f"k_fin = {k_fin:.2f} µm²")
    ax.fill_between(dx_array, k_fin - u_num, k_fin + u_num, color="red", alpha=0.15,
                    label=f"±u_num = ±{u_num:.4f} µm²")
    ax.set(xlabel="dx (m)", ylabel="k (µm²)",
           title="Perméabilité moyenne par niveau de maillage")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_mesh_refinement():
    """Visualisation du raffinement géométrique (seed fixe)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        f"Raffinement de maillage — seed fixe = {SEED_VISU}\n"
        f"(NX×dx = 200 µm constant → même géométrie physique)", fontsize=13,
    )
    for j, (nx, dx) in enumerate(zip(NX_VISU, DX_VISU)):
        filename = f"fiber_mat_visu_NX{nx}.tiff"
        Generate_sample(SEED_VISU, filename, MEAN_FIBER_D, ECART_TYPE_D, PORO, nx, dx, plot=False)
        img = np.array(Image.open(filename))
        axes[j].imshow(img, cmap="gray", origin="lower")
        axes[j].set_title(f"NX={nx}, dx={dx * 1e6:.1f} µm", fontsize=12)
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig("raffinement_geometrie.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Graphique sauvegardé : raffinement_geometrie.png")


def print_summary(k_means, k_stds, p_apparent, rapport, k_num, u_num):
    """Affiche le résumé final."""
    print("\n--- RÉSUMÉ ---")
    for i, nx in enumerate(NX_LIST):
        print(f"  NX={nx:3d} : k_moy = {k_means[i]:.4f} ± {k_stds[i]:.4f} µm²")
    print(f"  Ordre apparent p    = {p_apparent:.4f}")
    print(f"  |p - pf|/pf         = {rapport * 100:.2f} %")
    print(f"  k_num               = {k_num:.4f} µm²")
    print(f"  u_num               = {u_num:.4f} µm²")


# ==============================================================================
# EXÉCUTION
# ==============================================================================

if __name__ == "__main__":

    # 1. Simulations
    k_means, k_stds, dx_array = run_convergence_study()

    # 2. Analyse
    p_apparent, intercept, _, err_rel, dx_coarse = compute_apparent_order(k_means, dx_array)
    k_num, u_num, rapport = compute_numerical_uncertainty(k_means, p_apparent)

    # 3. Graphiques
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    plot_convergence(ax1, dx_coarse, err_rel, intercept, p_apparent)
    plot_permeability(ax2, dx_array, k_means, k_stds, u_num)
    plt.tight_layout()
    plt.savefig("convergence_maillage.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Graphique sauvegardé : convergence_maillage.png")

    plot_mesh_refinement()

    # 4. Résumé
    print_summary(k_means, k_stds, p_apparent, rapport, k_num, u_num)