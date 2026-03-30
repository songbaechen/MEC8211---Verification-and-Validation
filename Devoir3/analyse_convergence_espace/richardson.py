"""
Question A — Estimation de l'incertitude numérique u_num sur la perméabilité
par étude de convergence en maillage
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
# PARAMÈTRES DU PROBLÈME
# ==============================================================================

PRESSURE_DROP = 0.1
TARGET_POROSITY = 0.9
FIBER_DIAMETER_MEAN = 12.5
FIBER_DIAMETER_STD = 2.85

# Domaine physique conservé constant : NX * dx = 200 µm
GRID_SIZES = [50, 75, 100, 150, 200]
CELL_SIZES = [4e-6, (200 / 75) * 1e-6, 2e-6, (200 / 150) * 1e-6, 1e-6]

FIXED_SEED = 42

THEORETICAL_ORDER = 2


# ==============================================================================
# FONCTIONS UTILITAIRES
# ==============================================================================

def run_mesh_case(nx: int, dx: float, seed: int) -> float:
    """
    Génère une géométrie fibreuse et calcule la perméabilité LBM
    pour un niveau de maillage donné.
    """
    image_name = f"fiber_mesh_NX{nx}_seed{seed}.tiff"

    equivalent_diameter = Generate_sample(
        seed,
        image_name,
        FIBER_DIAMETER_MEAN,
        FIBER_DIAMETER_STD,
        TARGET_POROSITY,
        nx,
        dx,
        plotting=False
    )

    permeability = LBM(
        image_name,
        nx,
        PRESSURE_DROP,
        dx,
        equivalent_diameter,
        plotting=False
    )

    return permeability


def compute_mesh_convergence(mesh_sizes, spacings, seed):
    """
    Calcule la perméabilité pour tous les niveaux de maillage.
    """
    permeability_values = []

    for nx, dx in zip(mesh_sizes, spacings):
        k_value = run_mesh_case(nx, dx, seed)
        permeability_values.append(k_value)

    return np.asarray(permeability_values, dtype=float)


def estimate_observed_order(dx_values, relative_errors):
    """
    Régression log-log pour obtenir l'ordre apparent de convergence.
    """
    log_dx = np.log(dx_values)
    log_err = np.log(relative_errors)

    slope, intercept, r_value, _, _ = linregress(log_dx, log_err)

    return slope, intercept, r_value


def estimate_numerical_uncertainty(k_values, dx_values, p_estimated, p_theoretical):
    """
    Détermine k_num et u_num selon les critères du cours.
    Le ratio de raffinement est calculé à partir des dx réels.
    """
    fine_value = k_values[-1]
    medium_value = k_values[-2]
    ratio = dx_values[-2] / dx_values[-1]

    discrepancy_ratio = abs((p_estimated - p_theoretical) / p_theoretical)

    if discrepancy_ratio < 0.01:
        k_num = fine_value + (fine_value - medium_value) / (ratio**p_theoretical - 1.0)
        u_num = 0.0
        method_label = "Extrapolation de Richardson"

    elif discrepancy_ratio <= 0.10:
        gci = (1.25 / (ratio**p_theoretical - 1.0)) * abs(medium_value - fine_value)
        u_num = gci / 2.0
        k_num = fine_value
        method_label = "GCI avec Fs = 1.25"

    else:
        p_bounded = min(max(0.5, p_estimated), p_theoretical)
        gci = (3.0 / (ratio**p_bounded - 1.0)) * abs(medium_value - fine_value)
        u_num = gci / 2.0
        k_num = fine_value
        method_label = f"GCI avec Fs = 3 et p_exp = {p_bounded:.2f}"

    return k_num, u_num, discrepancy_ratio, method_label


def plot_convergence_results(dx_values, k_values, coarse_dx, coarse_errors, fit_slope, fit_intercept, u_num):
    """
    Produit deux graphiques de convergence.
    """
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))

    # ------------------------------------------------------------------
    # Graphique 1 : erreur relative log-log
    # ------------------------------------------------------------------
    fitted_dx = np.linspace(coarse_dx.min() * 0.85, coarse_dx.max() * 1.15, 200)
    fitted_error = np.exp(fit_intercept) * fitted_dx**fit_slope

    axes[0].loglog(
        coarse_dx * 1e6,
        coarse_errors * 100,
        marker="o",
        linewidth=1.8,
        markersize=6,
        label="Erreur relative"
    )
    axes[0].loglog(
        fitted_dx * 1e6,
        fitted_error * 100,
        linestyle="--",
        linewidth=2.0,
        label=f"Régression log-log (p = {fit_slope:.2f})"
    )
    axes[0].set_xlabel("Taille de maille Δx (µm)")
    axes[0].set_ylabel("Erreur relative (%)")
    axes[0].set_title("Erreur relative par rapport au maillage le plus fin")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    # ------------------------------------------------------------------
    # Graphique 2 : k en fonction de Δx avec barres d'erreur ±u_num
    # ------------------------------------------------------------------
    axes[1].errorbar(
        dx_values * 1e6,
        k_values,
        yerr=2 * u_num,
        fmt="s-",
        markersize=6,
        linewidth=1.8,
        capsize=6,
        capthick=1.5,
        label=f"k ± GCI (±{2 * u_num:.4f} µm²)"
    )
    axes[1].axhline(
        k_values[-1],
        linestyle="--",
        linewidth=1.5,
        label=f"Maillage fin : {k_values[-1]:.2f} µm²"
    )
    axes[1].invert_xaxis()
    axes[1].set_xlabel("Taille de maille Δx (µm)")
    axes[1].set_ylabel("Perméabilité k (µm²)")
    axes[1].set_title("Évolution de la perméabilité avec le raffinement")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("convergence_maillage.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("Graphique sauvegardé : convergence_maillage.png")


def plot_geometry_refinement(seed):
    """
    Visualisation de la même géométrie physique pour différents maillages.
    """
    selected_nx = [50, 100, 200]
    selected_dx = [4e-6, 2e-6, 1e-6]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle(
        f"Raffinement géométrique à seed fixé = {seed}\n"
        f"(NX × Δx constant = 200 µm)",
        fontsize=13
    )

    for ax, nx, dx in zip(axes, selected_nx, selected_dx):
        img_name = f"geometry_seed{seed}_NX{nx}.tiff"
        Generate_sample(
            seed,
            img_name,
            FIBER_DIAMETER_MEAN,
            FIBER_DIAMETER_STD,
            TARGET_POROSITY,
            nx,
            dx,
            plotting=False
        )

        image_data = np.array(Image.open(img_name))
        ax.imshow(image_data, cmap="gray", origin="lower")
        ax.set_title(f"NX = {nx}, Δx = {dx * 1e6:.2f} µm", fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("raffinement_geometrie.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("Graphique sauvegardé : raffinement_geometrie.png")


def print_summary(mesh_sizes, dx_values, k_values, p_estimated, discrepancy_ratio, k_num, u_num, method_label):
    """
    Affiche le résumé final.
    """
    print("\n" + "=" * 60)
    print("Résumé de l'étude de convergence en maillage")
    print("=" * 60)

    for nx, dx, k_val in zip(mesh_sizes, dx_values, k_values):
        print(f"NX = {nx:3d} | Δx = {dx * 1e6:6.3f} µm | k = {k_val:8.4f} µm²")

    print("-" * 60)
    print(f"Ordre apparent p               : {p_estimated:.4f}")
    print(f"|p - p_th| / p_th             : {100 * discrepancy_ratio:.2f} %")
    print(f"Méthode retenue               : {method_label}")
    print(f"k_num                         : {k_num:.4f} µm²")
    print(f"u_num                         : {u_num:.4f} µm²")
    print("=" * 60 + "\n")


# ==============================================================================
# SCRIPT PRINCIPAL
# ==============================================================================

if __name__ == "__main__":
    permeability_by_mesh = compute_mesh_convergence(GRID_SIZES, CELL_SIZES, FIXED_SEED)

    finest_mesh_value = permeability_by_mesh[-1]
    relative_error_vs_finest = np.abs(finest_mesh_value - permeability_by_mesh[:-1]) / finest_mesh_value
    coarse_spacings = np.asarray(CELL_SIZES[:-1], dtype=float)
    full_spacings = np.asarray(CELL_SIZES, dtype=float)

    apparent_order, log_fit_intercept, r_fit = estimate_observed_order(
        coarse_spacings,
        relative_error_vs_finest
    )

    k_num, u_num, relative_order_gap, selected_method = estimate_numerical_uncertainty(
        permeability_by_mesh,
        full_spacings,
        apparent_order,
        THEORETICAL_ORDER
    )

    plot_convergence_results(
        full_spacings,
        permeability_by_mesh,
        coarse_spacings,
        relative_error_vs_finest,
        apparent_order,
        log_fit_intercept,
        u_num
    )

    # plot_geometry_refinement(FIXED_SEED)

    print_summary(
        GRID_SIZES,
        full_spacings,
        permeability_by_mesh,
        apparent_order,
        relative_order_gap,
        k_num,
        u_num,
        selected_method
    )