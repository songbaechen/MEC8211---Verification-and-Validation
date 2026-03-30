"""
Question B — Incertitude u_input par Monte-Carlo
"""

from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
from scipy.stats import lognorm

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from problem_definition.lbm_devoir3 import Generate_sample, LBM
from problem_definition.problem_definition import SampleParameters, ProblemParameters


@dataclass
class MonteCarloInputs:
    n_runs: int = 200
    porosity_mean: float = 0.900
    porosity_std: float = 7.50e-3
    mean_fiber_diameter: float = 12.5
    fiber_diameter_std: float = 2.85
    nx: int = 100
    dx: float = 2e-6
    delta_p: float = 0.1
    filename: str = "fiber_mat_MC.tiff"
    rng_seed: int = 42
    sample_seed: int = 0


def draw_porosity(rng, mean, std):
    porosity = rng.normal(mean, std)
    while not (0.0 < porosity < 1.0):
        porosity = rng.normal(mean, std)
    return porosity


def run_single_case(inputs: MonteCarloInputs, porosity: float):
    sample_params = SampleParameters(
        seed=inputs.sample_seed,
        filename=inputs.filename,
        mean_d=inputs.mean_fiber_diameter,
        std_d=inputs.fiber_diameter_std,
        poro=porosity,
        nx=inputs.nx,
        dx=inputs.dx,
    )

    d_equivalent = Generate_sample(
        sample_params.seed,
        sample_params.filename,
        sample_params.mean_d,
        sample_params.std_d,
        sample_params.poro,
        sample_params.nx,
        sample_params.dx,
        False
    )

    problem_params = ProblemParameters(
        filename=inputs.filename,
        NX=inputs.nx,
        deltaP=inputs.delta_p,
        dx=inputs.dx,
        d_equivalent=d_equivalent,
    )

    permeability = LBM(
        problem_params.filename,
        problem_params.NX,
        problem_params.deltaP,
        problem_params.dx,
        problem_params.d_equivalent,
        False
    )

    return permeability


def compute_statistics(permeabilities):
    permeabilities = np.asarray(permeabilities, dtype=float)
    log_k = np.log(permeabilities)

    mu_log = np.mean(log_k)
    sigma_log = np.std(log_k, ddof=1)

    median_k = np.exp(mu_log)
    fvg = np.exp(sigma_log)

    u_minus = median_k - median_k / fvg
    u_plus = median_k * fvg - median_k

    shape, loc, scale = lognorm.fit(permeabilities, floc=0)

    return {
        "permeabilities": permeabilities,
        "median_k": median_k,
        "fvg": fvg,
        "u_minus": u_minus,
        "u_plus": u_plus,
        "shape": shape,
        "loc": loc,
        "scale": scale,
    }


def plot_results(stats_results, n_runs):
    k = stats_results["permeabilities"]
    median_k = stats_results["median_k"]
    fvg = stats_results["fvg"]

    x_range = np.linspace(k.min() * 0.7, k.max() * 1.3, 300)
    pdf_fit = lognorm.pdf(
        x_range,
        s=stats_results["shape"],
        loc=stats_results["loc"],
        scale=stats_results["scale"]
    )
    cdf_fit = lognorm.cdf(
        x_range,
        s=stats_results["shape"],
        loc=stats_results["loc"],
        scale=stats_results["scale"]
    )

    # =========================
    # Figure 1 : PDF
    # =========================
    plt.figure(figsize=(7, 5))
    plt.hist(
        k,
        bins=10,
        density=True,
        alpha=0.65,
        edgecolor="black",
        label="Échantillons Monte-Carlo"
    )
    plt.plot(
        x_range,
        pdf_fit,
        linewidth=2.2,
        label="Ajustement log-normal"
    )
    plt.axvline(
        median_k,
        linestyle="--",
        linewidth=1.6,
        label=f"Médiane = {median_k:.2f} µm²"
    )
    plt.axvline(
        median_k / fvg,
        linestyle=":",
        linewidth=1.4,
        label=f"Médiane / FVG = {median_k / fvg:.2f}"
    )
    plt.axvline(
        median_k * fvg,
        linestyle=":",
        linewidth=1.4,
        label=f"Médiane × FVG = {median_k * fvg:.2f}"
    )
    plt.title("Distribution de la perméabilité")
    plt.xlabel("Perméabilité k (µm²)")
    plt.ylabel("Densité de probabilité")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig("MC_PDF.png", dpi=150, bbox_inches="tight")
    plt.show()

    # =========================
    # Figure 2 : CDF
    # =========================
    k_sorted = np.sort(k)
    cdf_emp = np.arange(1, len(k_sorted) + 1) / len(k_sorted)

    plt.figure(figsize=(7, 5))
    plt.step(
        k_sorted,
        cdf_emp,
        where="post",
        linewidth=2,
        label="CDF empirique"
    )
    plt.plot(
        x_range,
        cdf_fit,
        linewidth=2.2,
        label="CDF log-normale ajustée"
    )
    plt.axvline(
        median_k,
        linestyle="--",
        linewidth=1.6,
        label=f"Médiane = {median_k:.2f} µm²"
    )
    plt.axhline(
        0.5,
        linestyle=":",
        linewidth=1.2,
        alpha=0.7,
        label="CDF = 0.5"
    )
    plt.title("Répartition cumulée de la perméabilité")
    plt.xlabel("Perméabilité k (µm²)")
    plt.ylabel("Probabilité cumulée")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig("MC_CDF.png", dpi=150, bbox_inches="tight")
    plt.show()

    # =========================
    # Figure 3 : Convergence
    # =========================
    cumulative_medians = [
        np.exp(np.mean(np.log(k[:n])))
        for n in range(1, len(k) + 1)
    ]

    plt.figure(figsize=(7, 4))
    plt.plot(
        range(1, n_runs + 1),
        cumulative_medians,
        linewidth=1.8,
        marker="o",
        markersize=5,
        label="Médiane cumulée"
    )
    plt.axhline(
        median_k,
        linestyle="--",
        linewidth=1.5,
        label=f"Médiane finale = {median_k:.2f} µm²"
    )
    plt.xlabel("Nombre de simulations N")
    plt.ylabel("Médiane cumulée de k (µm²)")
    plt.title("Convergence de la médiane (Monte-Carlo)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("MC_convergence.png", dpi=150, bbox_inches="tight")
    plt.show()


def monte_carlo_analysis(inputs: MonteCarloInputs):
    rng = np.random.default_rng(inputs.rng_seed)

    permeabilities = []

    for _ in range(inputs.n_runs):
        porosity = draw_porosity(
            rng,
            inputs.porosity_mean,
            inputs.porosity_std
        )

        permeability = run_single_case(inputs, porosity)
        permeabilities.append(permeability)

    stats_results = compute_statistics(permeabilities)

    print("\n" + "=" * 52)
    print("Résumé Monte-Carlo")
    print("=" * 52)
    print(f"Nombre d'échantillons         : {len(stats_results['permeabilities'])}")
    print(f"Médiane de perméabilité       : {stats_results['median_k']:.2f} µm²")
    print(f"Facteur de variation géom.    : {stats_results['fvg']:.3f}")
    print(f"u_input⁻                      : {stats_results['u_minus']:.2f} µm²")
    print(f"u_input⁺                      : {stats_results['u_plus']:.2f} µm²")
    print("=" * 52 + "\n")

    plot_results(stats_results, inputs.n_runs)

    return stats_results


if __name__ == "__main__":
    inputs = MonteCarloInputs()
    results = monte_carlo_analysis(inputs)