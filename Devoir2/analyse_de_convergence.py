"""
Lit un fichier CSV de résultats et trace l'analyse de convergence en espace
ou en temps.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_orders(x_vals: np.ndarray, err_vals: np.ndarray) -> np.ndarray:
    """
    Calcule les ordres de convergence entre niveaux successifs.

    Parameters
    ----------
    x_vals : np.ndarray
        Valeurs de h ou de dt, triées du plus grand au plus petit.
    err_vals : np.ndarray
        Valeurs d'erreur associées.

    Returns
    -------
    np.ndarray
        Tableau des ordres, avec NaN à l'indice 0.
    """
    orders = np.full(len(x_vals), np.nan, dtype=float)

    for idx in range(1, len(x_vals)):
        ratio_x = x_vals[idx - 1] / x_vals[idx]
        ratio_e = err_vals[idx - 1] / err_vals[idx]
        orders[idx] = np.log(ratio_e) / np.log(ratio_x)

    return orders


def main() -> None:
    """
    Lit le CSV, filtre selon le mode demandé, trace les courbes de convergence
    et sauvegarde/affiche la figure selon les options choisies.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Fichier CSV d'entrée.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["space", "time"],
        help="Type d'analyse à effectuer."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Nom du fichier image à sauvegarder (ex. convergence_space.png)."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Affiche la figure à l'écran."
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = df[df["mode"] == args.mode].copy()

    if df.empty:
        raise ValueError("Aucune donnée trouvée dans le CSV pour le mode demandé.")

    if args.mode == "space":
        x_col = "h"
        x_label = "h = Δr [m]"
        title = "Convergence en espace"
        ordre_theorique = 2.0
    else:
        x_col = "dt"
        x_label = "Δt [s]"
        title = "Convergence en temps"
        ordre_theorique = 1.0

    df = df.sort_values(by=x_col, ascending=False)

    x_vals = df[x_col].to_numpy(dtype=float)
    l1_vals = df["L1"].to_numpy(dtype=float)
    l2_vals = df["L2"].to_numpy(dtype=float)
    linf_vals = df["Linf"].to_numpy(dtype=float)

    p_l1 = compute_orders(x_vals, l1_vals)
    p_l2 = compute_orders(x_vals, l2_vals)
    p_linf = compute_orders(x_vals, linf_vals)

    print("\nOrdres de convergence :")
    print(" i |     x_{i-1}       x_i    |   p(L1)    p(L2)   p(Linf)")
    print("-" * 66)
    for idx in range(1, len(x_vals)):
        print(
            f"{idx:2d} | {x_vals[idx - 1]:.3e}  {x_vals[idx]:.3e} |"
            f" {p_l1[idx]:8.3f}  {p_l2[idx]:8.3f}  {p_linf[idx]:8.3f}"
        )

    c_ref = l2_vals[0] / (x_vals[0] ** ordre_theorique)
    ref_curve = c_ref * (x_vals ** ordre_theorique)

    plt.figure(dpi=100)
    plt.loglog(x_vals, l1_vals, "o-.", linewidth=2, label="L1")
    plt.loglog(x_vals, l2_vals, "s-.", linewidth=2, label="L2")
    plt.loglog(x_vals, linf_vals, "^-.", linewidth=2, label="L∞")
    plt.loglog(
        x_vals,
        ref_curve,
        color="black",
        linestyle="-",
        linewidth=2,
        label=f"O(x^{ordre_theorique:.0f})"
    )

    plt.xlabel(x_label)
    plt.ylabel("Norme de l'erreur")
    plt.title(title)
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()

    if args.output is not None:
        plt.savefig(args.output, dpi=300, bbox_inches="tight")
        print(f"Figure sauvegardée : {args.output}")

    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()