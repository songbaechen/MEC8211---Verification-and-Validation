"""
Question E — Estimation de l'erreur du modèle
"""

import numpy as np

from monte_carlo import monte_carlo_analysis, MonteCarloInputs
from simulation_error import compute_simulation_error
from experimental_uncertainty import compute_experimental_uncertainty
from numerical_uncertainty import run_numerical_uncertainty_analysis


def compute_validation_uncertainty(u_num, u_input_minus, u_input_plus, u_d):
    """
    Calcule l'incertitude de validation asymétrique.
    """
    u_val_minus = np.sqrt(u_num**2 + u_input_minus**2 + u_d**2)
    u_val_plus = np.sqrt(u_num**2 + u_input_plus**2 + u_d**2)
    return u_val_minus, u_val_plus


def compute_model_error_interval(simulation_error, u_val_minus, u_val_plus, coverage_factor=2.0):
    """
    Calcule l'intervalle de confiance sur l'erreur du modèle.
    """
    lower_bound = simulation_error - coverage_factor * u_val_plus
    upper_bound = simulation_error + coverage_factor * u_val_minus
    return lower_bound, upper_bound


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Question A
    # ------------------------------------------------------------------
    a_results = run_numerical_uncertainty_analysis()
    u_num = a_results["u_num"]

    # ------------------------------------------------------------------
    # Question B
    # ------------------------------------------------------------------
    mc_inputs = MonteCarloInputs()
    b_results = monte_carlo_analysis(mc_inputs)

    s_median = b_results["median_k"]
    u_input_minus = b_results["u_minus"]
    u_input_plus = b_results["u_plus"]

    # ------------------------------------------------------------------
    # Question C
    # ------------------------------------------------------------------
    u_d = compute_experimental_uncertainty()

    # ------------------------------------------------------------------
    # Question D
    # ------------------------------------------------------------------
    e_sim = compute_simulation_error(b_results)

    # ------------------------------------------------------------------
    # Question E
    # ------------------------------------------------------------------
    coverage_factor = 2.0

    u_val_minus, u_val_plus = compute_validation_uncertainty(
        u_num,
        u_input_minus,
        u_input_plus,
        u_d
    )

    delta_lower, delta_upper = compute_model_error_interval(
        e_sim,
        u_val_minus,
        u_val_plus,
        coverage_factor
    )

    print("\n" + "=" * 58)
    print("Résumé de l'erreur du modèle")
    print("=" * 58)
    print(f"u_num                         : {u_num:.4f} µm²")
    print(f"u_input⁻                      : {u_input_minus:.4f} µm²")
    print(f"u_input⁺                      : {u_input_plus:.4f} µm²")
    print(f"u_D                           : {u_d:.4f} µm²")
    print(f"E = S - D                     : {e_sim:.4f} µm²")
    print(f"u_val⁻                        : {u_val_minus:.4f} µm²")
    print(f"u_val⁺                        : {u_val_plus:.4f} µm²")
    print(f"k                             : {coverage_factor:.1f}")
    print("-" * 58)
    print("Intervalle sur δ_model :")
    print(f"{delta_lower:.4f} ≤ δ_model ≤ {delta_upper:.4f}  µm²")
    print("=" * 58 + "\n")