"""
Fonctions de post-traitement : normes d’erreur et tracés des profils de concentration.
"""
import numpy as np
import matplotlib.pyplot as plt

from mesh_and_parameters import ProblemParameters
from finite_differences_schemes import solve_unsteady_scheme
from mms_solution import MMSParams, mms_function, source_term_MMS, mms_iteration


def error_norms(
    c_num_hist: np.ndarray,
    c_mms_hist: np.ndarray,
    r_mesh: np.ndarray,
    time_array: np.ndarray
) -> tuple[float, float, float]:
    """
    Calcule les normes d'erreur L1, L2 et L_inf entre la solution numérique
    et la solution manufacturée déjà évaluée sur la grille espace-temps.

    Les deux tableaux doivent avoir le même format, soit `(nt, n_profile)`.

    Parameters
    ----------
    c_num_hist : np.ndarray
        Historique temporel de la solution numérique.
    c_mms_hist : np.ndarray
        Historique temporel de la solution manufacturée évaluée sur la même
        grille espace-temps que la solution numérique.
    r_mesh : np.ndarray
        Maillage radial.
    time_array : np.ndarray
        Vecteur temps.

    Returns
    -------
    tuple[float, float, float]
        Les normes d'erreur L1, L2 et L_inf.
    """
    if c_num_hist.shape != c_mms_hist.shape:
        raise ValueError(
            "c_num_hist et c_mms_hist doivent avoir la même dimension."
        )

    if len(r_mesh) < 2 or len(time_array) < 2:
        raise ValueError(
            "r_mesh et time_array doivent contenir au moins deux points."
        )

    dr = float(r_mesh[1] - r_mesh[0])
    dt = float(time_array[1] - time_array[0])

    error = c_num_hist - c_mms_hist

    l1_norm = float(np.sum(np.abs(error)) * dr * dt)
    l2_norm = float(np.sqrt(np.sum(error ** 2) * dr * dt))
    linf_norm = float(np.max(np.abs(error)))

    return l1_norm, l2_norm, linf_norm


def compute_convergence_orders(
    l1_errors_dict: dict[float, float],
    l2_errors_dict: dict[float, float],
    linf_errors_dict: dict[float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcule les ordres de convergence entre niveaux successifs à partir de
    dictionnaires d'erreurs indexés par la taille de pas h.

    Les dictionnaires doivent contenir les erreurs L1, L2 et L_inf associées
    aux mêmes valeurs de h. Les valeurs de h sont triées du maillage le plus
    grossier au plus fin avant le calcul des ordres.

    Parameters
    ----------
    l1_errors_dict : dict[float, float]
        Dictionnaire des erreurs L1, indexées par h.
    l2_errors_dict : dict[float, float]
        Dictionnaire des erreurs L2, indexées par h.
    linf_errors_dict : dict[float, float]
        Dictionnaire des erreurs L_inf, indexées par h.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Un tuple contenant :
        - `h_sorted` : les tailles de pas triées du plus grand au plus petit ;
        - `p_l1` : les ordres de convergence associés à la norme L1 ;
        - `p_l2` : les ordres de convergence associés à la norme L2 ;
        - `p_linf` : les ordres de convergence associés à la norme L_inf.
    """
    h_keys_l1 = set(l1_errors_dict.keys())
    h_keys_l2 = set(l2_errors_dict.keys())
    h_keys_linf = set(linf_errors_dict.keys())

    if h_keys_l1 != h_keys_l2 or h_keys_l1 != h_keys_linf:
        raise ValueError(
            "Les dictionnaires d'erreurs doivent contenir exactement les mêmes valeurs de h."
        )

    h_sorted = np.array(sorted(h_keys_l1, reverse=True), dtype=float)

    e_l1_sorted = np.array([l1_errors_dict[h] for h in h_sorted], dtype=float)
    e_l2_sorted = np.array([l2_errors_dict[h] for h in h_sorted], dtype=float)
    e_linf_sorted = np.array([linf_errors_dict[h] for h in h_sorted], dtype=float)

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
            raise ValueError(
                "Deux pas successifs sont identiques (log(h_{i-1}/h_i)=0)."
            )

        p_l1[idx] = np.log(ratio_l1) / denom
        p_l2[idx] = np.log(ratio_l2) / denom
        p_linf[idx] = np.log(ratio_linf) / denom

        print(
            f"{idx:2d} | {h_sorted[idx - 1]:.3e}  {h_sorted[idx]:.3e} |"
            f" {p_l1[idx]:8.3f}  {p_l2[idx]:8.3f}  {p_linf[idx]:8.3f}"
        )

    return h_sorted, p_l1, p_l2, p_linf


def plot_mms_solution_profiles(
    problem: ProblemParameters,
    mms: MMSParams,
    num_nodes: int,
    times_to_plot: list[float]):
    """
    Plot des profils C_mms(r,t) pour plusieurs temps.
    """

    r_mesh = np.linspace(0.0, float(problem.r), int(num_nodes))

    plt.figure(dpi=100)

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

    plt.figure(dpi=100)

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


def plot_error_convergence_space(
    l1_errors_dict: dict[float, float],
    l2_errors_dict: dict[float, float],
    linf_errors_dict: dict[float, float],
    show_reference: bool = True,
    ordre_theorique: float = 2.0
) -> None:
    """
    Trace les erreurs spatiales en échelle log-log.

    Les erreurs L1, L2 et L_inf sont fournies sous forme de dictionnaires
    indexés par le pas spatial h = Δr. Une droite de référence correspondant
    à l'ordre théorique attendu peut être affichée pour faciliter l'analyse
    de la convergence.

    Parameters
    ----------
    l1_errors_dict : dict[float, float]
        Dictionnaire des erreurs L1 indexées par h.
    l2_errors_dict : dict[float, float]
        Dictionnaire des erreurs L2 indexées par h.
    linf_errors_dict : dict[float, float]
        Dictionnaire des erreurs L_inf indexées par h.
    show_reference : bool, optional
        Active ou désactive l'affichage de la pente théorique. Par défaut True.
    ordre_theorique : float, optional
        Ordre théorique utilisé pour tracer la pente de référence. Par défaut 2.

    Returns
    -------
    None
        Affiche directement la figure.
    """

    h_keys_l1 = set(l1_errors_dict.keys())
    h_keys_l2 = set(l2_errors_dict.keys())
    h_keys_linf = set(linf_errors_dict.keys())

    if h_keys_l1 != h_keys_l2 or h_keys_l1 != h_keys_linf:
        raise ValueError(
            "Les dictionnaires d'erreurs doivent contenir exactement les mêmes valeurs de h."
        )

    h_sorted = np.array(sorted(h_keys_l1, reverse=True), dtype=float)

    l1_sorted = np.array([l1_errors_dict[h] for h in h_sorted], dtype=float)
    l2_sorted = np.array([l2_errors_dict[h] for h in h_sorted], dtype=float)
    linf_sorted = np.array([linf_errors_dict[h] for h in h_sorted], dtype=float)

    plt.figure(dpi=100)

    plt.loglog(h_sorted, l1_sorted, "o-.", linewidth=2, label="L1")
    plt.loglog(h_sorted, l2_sorted, "s-.", linewidth=2, label="L2")
    plt.loglog(h_sorted, linf_sorted, "^-.", linewidth=2, label="L∞")

    if show_reference:
        # Construction de la droite O(h^p)
        h_ref = h_sorted
        c_ref = l2_sorted[0] / (h_ref[0] ** ordre_theorique)
        ref_curve = c_ref * (h_ref ** ordre_theorique)

        plt.loglog(
            h_ref,
            ref_curve,
            color="black",
            linestyle="-",
            linewidth=2,
            label=f"O(h^{ordre_theorique:.0f})"
        )

    plt.xlabel("h = Δr [m]")
    plt.ylabel("Norme de l'erreur")
    plt.title("Convergence en espace")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_error_convergence_time(
    l1_errors_dict: dict[float, float],
    l2_errors_dict: dict[float, float],
    linf_errors_dict: dict[float, float],
    show_reference: bool = True,
    ordre_theorique: float = 1.0
) -> None:
    """
    Trace les erreurs temporelles en échelle log-log.

    Les erreurs L1, L2 et L_inf sont fournies sous forme de dictionnaires
    indexés par le pas de temps Δt.

    Une droite de référence correspondant à l'ordre théorique attendu peut
    être affichée afin de faciliter l'analyse de la convergence temporelle.

    Parameters
    ----------
    l1_errors_dict : dict[float, float]
        Dictionnaire des erreurs L1 indexées par Δt.
    l2_errors_dict : dict[float, float]
        Dictionnaire des erreurs L2 indexées par Δt.
    linf_errors_dict : dict[float, float]
        Dictionnaire des erreurs L_inf indexées par Δt.
    show_reference : bool, optional
        Active ou désactive l'affichage de la pente théorique.
        Par défaut True.
    ordre_theorique : float, optional
        Ordre théorique utilisé pour tracer la pente de référence.
        Par défaut 1 (schéma Euler implicite).

    Returns
    -------
    None
        La fonction affiche directement la figure.
    """

    dt_keys_l1 = set(l1_errors_dict.keys())
    dt_keys_l2 = set(l2_errors_dict.keys())
    dt_keys_linf = set(linf_errors_dict.keys())

    if dt_keys_l1 != dt_keys_l2 or dt_keys_l1 != dt_keys_linf:
        raise ValueError(
            "Les dictionnaires d'erreurs doivent contenir exactement les mêmes valeurs de Δt."
        )

    dt_sorted = np.array(sorted(dt_keys_l1, reverse=True), dtype=float)

    l1_sorted = np.array([l1_errors_dict[dt] for dt in dt_sorted], dtype=float)
    l2_sorted = np.array([l2_errors_dict[dt] for dt in dt_sorted], dtype=float)
    linf_sorted = np.array([linf_errors_dict[dt] for dt in dt_sorted], dtype=float)

    plt.figure(dpi=100)

    plt.loglog(dt_sorted, l1_sorted, "o-.", linewidth=2, label="L1")
    plt.loglog(dt_sorted, l2_sorted, "s-.", linewidth=2, label="L2")
    plt.loglog(dt_sorted, linf_sorted, "^-.", linewidth=2, label="L∞")

    if show_reference:
        dt_ref = dt_sorted
        c_ref = l2_sorted[0] / (dt_ref[0] ** ordre_theorique)
        ref_curve = c_ref * (dt_ref ** ordre_theorique)

        plt.loglog(
            dt_ref,
            ref_curve,
            color="black",
            linestyle="-",
            linewidth=2,
            label=f"O(Δt^{ordre_theorique:.0f})"
        )

    plt.xlabel("Δt [s]")
    plt.ylabel("Norme de l'erreur")
    plt.title("Convergence en temps")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_heatmaps_num_mms_error(
    param: ProblemParameters,
    n_profile: int,
    dt: float,
    mms: MMSParams
) -> None:
    """
    Trace les cartes de chaleur de la solution numérique, de la solution MMS
    exacte et de l'erreur associée.

    La fonction résout d'abord le problème numérique avec
    `solve_unsteady_scheme`, puis construit la solution manufacturée exacte
    avec `mms_iteration`. Les trois champs suivants sont ensuite affichés :
    - la solution numérique ;
    - la solution MMS ;
    - l'erreur définie par (numérique - MMS).

    Parameters
    ----------
    param : ProblemParameters
        Paramètres physiques du problème.
    n_profile : int
        Nombre de nœuds du maillage radial.
    dt : float
        Pas de temps.
    mms : MMSParams
        Paramètres de la solution manufacturée.

    Returns
    -------
    None
        La fonction affiche directement la figure.
    """
    r_mesh, time_array, c_num_hist = solve_unsteady_scheme(
        param=param,
        n_profile=n_profile,
        dt=dt,
        mms=mms
    )

    r_mesh_mms, time_array_mms, c_mms_hist = mms_iteration(
        param=param,
        n_profile=n_profile,
        dt=dt,
        mms=mms
    )

    if not np.allclose(r_mesh, r_mesh_mms):
        raise ValueError(
            "Les maillages radiaux de la solution numérique et de la MMS ne correspondent pas."
        )

    if not np.allclose(time_array, time_array_mms):
        raise ValueError(
            "Les vecteurs temps de la solution numérique et de la MMS ne correspondent pas."
        )

    error_hist = c_num_hist - c_mms_hist

    extent = (
        float(r_mesh[0]),
        float(r_mesh[-1]),
        float(time_array[0]),
        float(time_array[-1])
    )

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14.0, 4.2), dpi=100)

    im_num = axes[0].imshow(
        c_num_hist,
        aspect="auto",
        origin="lower",
        extent=extent
    )
    axes[0].set_title("Solution numérique")
    axes[0].set_xlabel("r [m]")
    axes[0].set_ylabel("t [s]")
    fig.colorbar(im_num, ax=axes[0])

    im_mms = axes[1].imshow(
        c_mms_hist,
        aspect="auto",
        origin="lower",
        extent=extent
    )
    axes[1].set_title("Solution MMS")
    axes[1].set_xlabel("r [m]")
    axes[1].set_ylabel("t [s]")
    fig.colorbar(im_mms, ax=axes[1])

    im_err = axes[2].imshow(
        error_hist,
        aspect="auto",
        origin="lower",
        extent=extent
    )
    axes[2].set_title("Erreur (num - MMS)")
    axes[2].set_xlabel("r [m]")
    axes[2].set_ylabel("t [s]")
    fig.colorbar(im_err, ax=axes[2])

    fig.tight_layout()
    plt.show()

def plot_original_problem_profiles(
    param: ProblemParameters,
    n_profile: int,
    dt: float,
    times_to_plot: list[float]
) -> None:
    """
    Trace les profils de concentration du problème original (sans MMS)
    pour plusieurs temps donnés.

    Parameters
    ----------
    param : ProblemParameters
        Paramètres physiques du problème original.
    n_profile : int
        Nombre de noeuds radiaux.
    dt : float
        Pas de temps utilisé pour la simulation.
    times_to_plot : list[float]
        Liste des temps auxquels on veut tracer les profils.
    """
    r_mesh, time_array, c_hist = solve_unsteady_scheme(
        param=param,
        n_profile=n_profile,
        dt=dt,
        mms=None
    )

    plt.figure(dpi=100)

    for t_val in times_to_plot:
        idx = int(np.argmin(np.abs(time_array - t_val)))
        plt.plot(r_mesh, c_hist[idx, :], label=f"t={time_array[idx]:.2e} s")

    plt.xlabel("r [m]")
    plt.ylabel("C(r,t)")
    plt.title("Solution du problème original")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
