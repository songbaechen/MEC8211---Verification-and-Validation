"""
Question D — Erreur de simulation E = S - D
"""

from monte_carlo import monte_carlo_analysis, MonteCarloInputs


def compute_simulation_error(mc_results, experimental_median=80.6):
    simulated_median = mc_results["median_k"]
    return simulated_median - experimental_median


if __name__ == "__main__":
    inputs = MonteCarloInputs()
    results = monte_carlo_analysis(inputs)

    E = compute_simulation_error(results)

    print("\n" + "=" * 50)
    print("Résumé de l'erreur de simulation")
    print("=" * 50)
    print(f"Médiane simulée S             : {results['median_k']:.2f} µm²")
    print(f"Médiane expérimentale D       : {80.6:.2f} µm²")
    print(f"Erreur E = S - D              : {E:.2f} µm²")
    print("=" * 50)