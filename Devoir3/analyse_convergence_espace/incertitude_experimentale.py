"""
Question C — Estimation de l'incertitude expérimentale u_D
"""

import numpy as np

def compute_experimental_uncertainty(
    reproducibility_std=14.7,
    instrument_uncertainty=10.0
):
    u_D = np.sqrt(reproducibility_std**2 + instrument_uncertainty**2)
    return u_D


if __name__ == "__main__":
    u_D = compute_experimental_uncertainty()

    print("\n" + "=" * 50)
    print("Résumé de l'incertitude expérimentale")
    print("=" * 50)
    print(f"Incertitude totale u_D          : {u_D:.2f} µm²")
    print("=" * 50)