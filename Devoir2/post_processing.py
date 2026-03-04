"""
Fonctions de post-traitement : normes d’erreur et tracés des profils de concentration.
"""
import numpy as np
import matplotlib.pyplot as plt

from mesh_and_parameters import ProblemParameters
from finite_differences_schemes import solve_unsteady_scheme
from mms_solution import MMSParams, mms_function, source_term_MMS


def error_norms(c_num_hist: np.ndarray, 
    r_mesh: np.ndarray, 
    time_array: np.ndarray,
    problem: ProblemParameters,
    mms: MMSParams,) -> tuple[float, float, float]:
    """
    Retourne les erreurs L1, L2 et L_inf (espace x temps)
    """
    num_time_steps, num_nodes = c_num_hist.shape
    dr = float(r_mesh[1] - r_mesh[0])
    dt = float(time_array[1] - time_array[0])

    error = np.zeros_like(c_num_hist, dtype=float)

    for n_idx in range(num_time_steps):
        t_n = float(time_array[n_idx])
        for i_idx in range(num_nodes):
            r_i = float(r_mesh[i_idx])
            c_exact = mms_function(r_i, t_n, float(problem.r), mms)
            error[n_idx, i_idx] = c_num_hist[n_idx, i_idx] - c_exact

    l1_norm = float(np.sum(np.abs(error)) * dr * dt)
    l2_norm = float(np.sqrt(np.sum(error**2) * dr * dt))
    linf_norm = float(np.max(np.abs(error)))

    return l1_norm, l2_norm, linf_norm

def compute_convergence_orders(
    step_sizes: list[float],
    l1_errors: list[float],
    l2_errors: list[float],
    linf_errors: list[float],) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcule les ordres de convergence entre niveaux successifs.
    """
    h_vals = np.asarray(step_sizes, dtype=float)
    e_l1 = np.asarray(l1_errors, dtype=float)
    e_l2 = np.asarray(l2_errors, dtype=float)
    e_linf = np.asarray(linf_errors, dtype=float)

    # Tri coarse -> fine (h décroissant)
    sort_idx = np.argsort(h_vals)[::-1]
    h_sorted = h_vals[sort_idx]
    e_l1_sorted = e_l1[sort_idx]
    e_l2_sorted = e_l2[sort_idx]
    e_linf_sorted = e_linf[sort_idx]

    p_l1 = np.full(h_sorted.size, np.nan, dtype=float)
    p_l2 = np.full(h_sorted.size, np.nan, dtype=float)
    p_linf = np.full(h_sorted.size, np.nan, dtype=float)

    print("\nOrdres de convergence (entre i-1 -> i):")
    print(" i |   h_{i-1}     h_i   |   p(L1)    p(L2)   p(Linf)")
    print("-" * 62)

    for idx in range(1, h_sorted.size):
        ratio_h = h_sorted[idx - 1] / h_sorted[idx]
        ratio_l1 = e_l1_sorted[idx - 1] / e_l1_sorted[idx]
        ratio_l2 = e_l2_sorted[idx - 1] / e_l2_sorted[idx]
        ratio_linf = e_linf_sorted[idx - 1] / e_linf_sorted[idx]

        denom = np.log(ratio_h)
        if abs(denom) < 1e-15:
            raise ValueError("Deux pas successifs sont identiques (log(h_{i-1}/h_i)=0).")

        p_l1[idx] = np.log(ratio_l1) / denom
        p_l2[idx] = np.log(ratio_l2) / denom
        p_linf[idx] = np.log(ratio_linf) / denom

        print(
            f"{idx:2d} | {h_sorted[idx-1]:.3e}  {h_sorted[idx]:.3e} |"
            f" {p_l1[idx]:8.3f}  {p_l2[idx]:8.3f}  {p_linf[idx]:8.3f}"
        )

    return h_sorted, p_l1, p_l2, p_linf

def convergence_space(
    problem: ProblemParameters,
    mms: MMSParams,
    num_nodes_list: list[int],
    dt_fixed: float) -> dict:
    """
    Étude de convergence en espace
    """

    dt_saved = float(problem.dt)
    problem.dt = float(dt_fixed)

    dr_list: list[float] = []
    l1_list: list[float] = []
    l2_list: list[float] = []
    linf_list: list[float] = []

    for num_nodes in num_nodes_list:
        r_mesh, time_array, c_hist = solve_unsteady_scheme(problem, int(num_nodes), mms)
        dr_val = float(r_mesh[1] - r_mesh[0])

        l1_val, l2_val, linf_val = error_norms(c_hist, r_mesh, time_array, problem, mms)

        dr_list.append(dr_val)
        l1_list.append(l1_val)
        l2_list.append(l2_val)
        linf_list.append(linf_val)

        print(
            f"[SPACE] N={int(num_nodes):5d}  dr={dr_val:.3e} |"
            f" L1={l1_val:.3e}  L2={l2_val:.3e}  Linf={linf_val:.3e}"
        )

    problem.dt = dt_saved

    h_sorted, p_l1, p_l2, p_linf = compute_convergence_orders(dr_list, l1_list, l2_list, linf_list)

    sort_idx = np.argsort(np.asarray(dr_list, dtype=float))[::-1]
    l1_sorted = np.asarray(l1_list, dtype=float)[sort_idx]
    l2_sorted = np.asarray(l2_list, dtype=float)[sort_idx]
    linf_sorted = np.asarray(linf_list, dtype=float)[sort_idx]

    return {
        "h_sorted": h_sorted,
        "L1_sorted": l1_sorted,
        "L2_sorted": l2_sorted,
        "Linf_sorted": linf_sorted,
        "p_L1": p_l1,
        "p_L2": p_l2,
        "p_Linf": p_linf,
    }

def convergence_time(
    problem: ProblemParameters,
    mms: MMSParams,
    dt_list: list[float],
    num_nodes_fixed: int) -> dict:
    """
    Étude de convergence en temps
    """

    dt_saved = float(problem.dt)

    dt_used: list[float] = []
    l1_list: list[float] = []
    l2_list: list[float] = []
    linf_list: list[float] = []

    for dt_val in dt_list:


        problem.dt = float(dt_val)
        r_mesh, time_array, c_hist = solve_unsteady_scheme(problem, int(num_nodes_fixed), mms)

        l1_val, l2_val, linf_val = error_norms(c_hist, r_mesh, time_array, problem, mms)

        dt_used.append(float(dt_val))
        l1_list.append(l1_val)
        l2_list.append(l2_val)
        linf_list.append(linf_val)

        print(
            f"[TIME]  dt={float(dt_val):.3e} s |"
            f" L1={l1_val:.3e}  L2={l2_val:.3e}  Linf={linf_val:.3e}"
        )

    problem.dt = dt_saved

    h_sorted, p_l1, p_l2, p_linf = compute_convergence_orders(dt_used, l1_list, l2_list, linf_list)

    sort_idx = np.argsort(np.asarray(dt_used, dtype=float))[::-1]
    l1_sorted = np.asarray(l1_list, dtype=float)[sort_idx]
    l2_sorted = np.asarray(l2_list, dtype=float)[sort_idx]
    linf_sorted = np.asarray(linf_list, dtype=float)[sort_idx]

    return {
        "h_sorted": h_sorted,
        "L1_sorted": l1_sorted,
        "L2_sorted": l2_sorted,
        "Linf_sorted": linf_sorted,
        "p_L1": p_l1,
        "p_L2": p_l2,
        "p_Linf": p_linf,
    }

def plot_mms_solution_profiles(
    problem: ProblemParameters,
    mms: MMSParams,
    num_nodes: int,
    times_to_plot: list[float]):
    """
    Plot des profils C_mms(r,t) pour plusieurs temps.
    """

    r_mesh = np.linspace(0.0, float(problem.r), int(num_nodes))

    plt.figure(dpi=300)

    for t_val in times_to_plot:
        c_vals = [
            mms_function(float(r_i), float(t_val), float(problem.r), mms)
            for r_i in r_mesh
        ]
        plt.plot(r_mesh, c_vals, label=f"t={t_val:.3e} s")

    plt.xlabel("r [m]")
    plt.ylabel("C_mms(r,t)")
    plt.title("Solution manufacturée C_mms(r,t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_mms_source_profiles(
    problem: ProblemParameters,
    mms: MMSParams,
    num_nodes: int,
    times_to_plot: list[float]):
    """
    Plot des profils du terme source S_mms(r,t).
    """
    r_mesh = np.linspace(0.0, float(problem.r), int(num_nodes))

    d_eff = float(problem.d_eff)
    k_reac = float(problem.k)

    plt.figure(dpi=300)

    for t_val in times_to_plot:
        s_vals = [
            source_term_MMS(
                float(r_i),
                float(t_val),
                float(problem.r),
                d_eff,
                k_reac,
                mms,
            )
            for r_i in r_mesh
        ]
        plt.plot(r_mesh, s_vals, label=f"t={t_val:.3e} s")

    plt.xlabel("r [m]")
    plt.ylabel("S_mms(r,t)")
    plt.title("Terme source manufacturé S_mms(r,t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_error_convergence_space(space_results: dict):
    """
    Plot Log-log des erreurs (L1, L2, Linf) en fonction de dr.
    """
    h_sorted = np.asarray(space_results["h_sorted"], dtype=float)
    l1_sorted = np.asarray(space_results["L1_sorted"], dtype=float)
    l2_sorted = np.asarray(space_results["L2_sorted"], dtype=float)
    linf_sorted = np.asarray(space_results["Linf_sorted"], dtype=float)

    plt.figure(dpi=220)
    plt.loglog(h_sorted, l1_sorted, "o-.", linewidth=2, label="L1")
    plt.loglog(h_sorted, l2_sorted, "s-.", linewidth=2, label="L2")
    plt.loglog(h_sorted, linf_sorted, "^-.", linewidth=2, label="Linf")

    plt.xlabel("h = Δr [m]")
    plt.ylabel("Norme de l'erreur (espace × temps)")
    plt.title("Convergence en espace (Δt fixé petit)")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_error_convergence_time(time_results: dict):
    """
    Plot Log-log des erreurs (L1, L2, Linf) en fonction de dt.
    """
    h_sorted = np.asarray(time_results["h_sorted"], dtype=float)
    l1_sorted = np.asarray(time_results["L1_sorted"], dtype=float)
    l2_sorted = np.asarray(time_results["L2_sorted"], dtype=float)
    linf_sorted = np.asarray(time_results["Linf_sorted"], dtype=float)

    plt.figure(dpi=220)
    plt.loglog(h_sorted, l1_sorted, "o-.", linewidth=2, label="L1")
    plt.loglog(h_sorted, l2_sorted, "s-.", linewidth=2, label="L2")
    plt.loglog(h_sorted, linf_sorted, "^-.", linewidth=2, label="Linf")

    plt.xlabel("h = Δt [s]")
    plt.ylabel("Norme de l'erreur (espace × temps)")
    plt.title("Convergence en temps (Δr fixé petit)")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_heatmaps_num_mms_error(
    c_num_hist: np.ndarray,
    r_mesh: np.ndarray,
    time_array: np.ndarray,
    problem: ProblemParameters,
    mms: MMSParams):
    """
    Plot Heat-Maps
    """
    num_time_steps, num_nodes = c_num_hist.shape

    c_mms_hist = np.zeros_like(c_num_hist, dtype=float)
    for n_idx in range(num_time_steps):
        t_val = float(time_array[n_idx])
        for i_idx in range(num_nodes):
            r_val = float(r_mesh[i_idx])
            c_mms_hist[n_idx, i_idx] = mms_function(r_val, t_val, float(problem.r), mms)

    error_hist = c_num_hist - c_mms_hist

    extent = (float(r_mesh[0]), float(r_mesh[-1]), float(time_array[0]), float(time_array[-1]))

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14.0, 4.2), dpi=220)

    im_num = axes[0].imshow(c_num_hist, aspect="auto", origin="lower", extent=extent)
    axes[0].set_title("Numérique")
    axes[0].set_xlabel("r [m]")
    axes[0].set_ylabel("t [s]")
    fig.colorbar(im_num, ax=axes[0])

    im_mms = axes[1].imshow(c_mms_hist, aspect="auto", origin="lower", extent=extent)
    axes[1].set_title("MMS")
    axes[1].set_xlabel("r [m]")
    fig.colorbar(im_mms, ax=axes[1])

    im_err = axes[2].imshow(error_hist, aspect="auto", origin="lower", extent=extent)
    axes[2].set_title("Erreur (num - MMS)")
    axes[2].set_xlabel("r [m]")
    fig.colorbar(im_err, ax=axes[2])

    fig.tight_layout()
    plt.show()