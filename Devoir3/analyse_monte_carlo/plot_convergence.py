import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def richardson_convergence(
        srq_list: list[float],
        dx_list: list[float],
        p_est: float = 3.0,
        max_n_iter: int = 100,
        tol: float = 5e-3
    ) -> float:
    r12 = dx_list[-2] / dx_list[-1]
    r23 = dx_list[-3] / dx_list[-2]

    f1 = srq_list[-1]
    f2 = srq_list[-2]
    f3 = srq_list[-3]

    eps12 = f2 - f1
    eps23 = f3 - f2

    if abs(eps12) < 1e-30:
        return p_est

    # Convergence oscillatoire
    if eps23 / eps12 < 0:
        return np.log(abs(eps23) / abs(eps12)) / np.log(r12)

    p_k = p_est
    for _ in range(max_n_iter):
        up_term = (r12 ** p_k - 1) * eps23 / eps12 + r12 ** p_k
        if up_term <= 0:
            return np.log(abs(eps23) / abs(eps12)) / np.log(r12)
        p_kp1 = np.log(up_term) / np.log(r12 * r23)
        if abs(p_kp1 - p_k) < tol:
            return p_kp1
        p_k = p_kp1
    return p_k


def gci_factor(f_fine, f_coarse, r, p_hat, p_formal=2.0):
    if abs((p_hat - p_formal) / p_formal) < 0.01:
        f_s = 1
        p = p_formal
        GCI = f_s * abs(f_coarse - f_fine) / (r ** p - 1)
        u_num = 0
    elif abs((p_hat - p_formal) / p_formal) <= 0.10:
        f_s = 1.25
        p = p_formal
        GCI = f_s * abs(f_coarse - f_fine) / (r ** p - 1)
        u_num = GCI
    else:
        f_s = 3.0
        p = min(max(0.5, p_hat), p_formal)
        GCI = f_s * abs(f_coarse - f_fine) / (r ** p - 1)
        u_num = GCI

    return GCI, u_num

def plot_convergence_and_gci(k_list, dx_list, p_formal=2.0):
    """
    Prend k_list et dx_list (ordonnés du plus grossier au plus fin)
    et génère les deux graphiques.
    """
    dx_arr = np.array(dx_list)
    k_arr = np.array(k_list)

    # --- Richardson ---
    p_hat = richardson_convergence(k_list, dx_list, p_est=p_formal)

    f1, f2 = k_arr[-1], k_arr[-2]
    r = dx_arr[-2] / dx_arr[-1]
    denom = r ** p_hat - 1
    k_exact = f1 + (f1 - f2) / denom if abs(denom) > 1e-30 else f1

    err = np.abs(k_arr - k_exact)

    # --- GCI ---
    gci_list = [0.0]
    u_num_list = []
    for i in range(1, len(dx_list)):
        r_i = dx_list[i - 1] / dx_list[i]
        # gci_i = gci_factor(k_list[i], k_list[i - 1], r_i, p_hat, p_formal)
        gci_i = gci_factor(k_list[i], k_list[i - 1], r_i, p_hat, 2.0)
        gci_list.append(gci_i[0])
        u_num_list.append(gci_i[1])
    gci_arr = np.array(gci_list)

    # u_num = gci_list[-1]
    u_num = u_num_list[-1]

    # --- Print résumé ---
    print(f"p_hat = {p_hat:.4f},  k_exact = {k_exact:.6f} µm²,  u_num = {u_num:.6f} µm²")
    for i, (dx, k, gci) in enumerate(zip(dx_list, k_list, gci_list)):
        print(f"  Mesh {i+1}: dx={dx:.4e}, k={k:.4f}, GCI={gci:.4f}")

    # ====================================================================
    # GRAPHIQUE 1 : Log-log  (k vs dx  +  erreur vs dx)
    # ====================================================================
    fig1, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax1 = axes[0]
    ax1.loglog(dx_arr, k_arr, 'o-', color='#2563eb', ms=8, lw=1.5, label='$k$ (LBM)')
    ax1.axhline(k_exact, color='#dc2626', ls='--', lw=1,
                label=f'$k_{{exact}}$ = {k_exact:.2f} µm²')
    for dx_i, k_i in zip(dx_arr, k_arr):
        ax1.annotate(f'{k_i:.2f}', (dx_i, k_i), xytext=(8, 8),
                     textcoords='offset points', fontsize=9)
    ax1.set_xlabel(r'$\Delta x$ (m)')
    ax1.set_ylabel(r'$k$ (µm²)')
    ax1.set_title('SRQ vs $\\Delta x$')
    ax1.legend()
    ax1.grid(True, which='both', alpha=0.3)

    ax2 = axes[1]
    mask = err > 1e-30
    if mask.sum() >= 2:
        ax2.loglog(dx_arr[mask], err[mask], 's-', color='#2563eb', ms=8, lw=1.5,
                   label='$|k - k_{exact}|$')
        dx_ref = dx_arr[mask]
        C = err[mask][-1] / dx_ref[-1] ** p_hat
        ax2.loglog(dx_ref, C * dx_ref ** p_hat, '--', color='#16a34a', lw=1.5,
                   label=f'$\\hat{{p}}$ = {p_hat:.2f}')
        C2 = err[mask][-1] / dx_ref[-1] ** p_formal
        ax2.loglog(dx_ref, C2 * dx_ref ** p_formal, ':', color='#9333ea', lw=1.5,
                   label=f'$p_{{formel}}$ = {p_formal:.1f}')
    ax2.set_xlabel(r'$\Delta x$ (m)')
    ax2.set_ylabel(r'$|k - k_{exact}|$ (µm²)')
    ax2.set_title('Ordre de convergence')
    ax2.legend()
    ax2.grid(True, which='both', alpha=0.3)

    fig1.tight_layout()
    path1 = str(SCRIPT_DIR / "convergence_loglog.png")
    fig1.savefig(path1, dpi=200, bbox_inches='tight')
    plt.close(fig1)
    print(f"[SAVED] {path1}")

    # ====================================================================
    # GRAPHIQUE 2 : GCI barres d'erreur
    # ====================================================================
    fig2, ax = plt.subplots(figsize=(8, 5.5))

    ax.errorbar(dx_arr, k_arr, yerr=gci_arr, fmt='o', color='#2563eb',
                ms=8, capsize=6, capthick=1.5, elinewidth=1.5, label='$k$ ± GCI')
    ax.axhline(k_exact, color='#dc2626', ls='--', lw=1.2,
               label=f'$k_{{exact}}$ = {k_exact:.2f} µm²')

    for dx_i, k_i, gci_i in zip(dx_arr, k_arr, gci_arr):
        if gci_i > 0:
            ax.annotate(f'GCI = {gci_i:.2f}', (dx_i, k_i + gci_i),
                        xytext=(5, 8), textcoords='offset points', fontsize=9)

    ax.set_xscale('log')
    ax.set_xlabel(r'$\Delta x$ (m)')
    ax.set_ylabel(r'$k$ (µm²)')
    ax.set_title('GCI — Barres d\'incertitude numérique')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    fig2.tight_layout()
    path2 = str(SCRIPT_DIR / "gci_barres_erreur.png")
    fig2.savefig(path2, dpi=200, bbox_inches='tight')
    plt.close(fig2)
    print(f"[SAVED] {path2}")


# ============================================================================
if __name__ == "__main__":
    # ---- COLLE TES RÉSULTATS ICI ----
    # Ordonnés du plus grossier au plus fin
    dx_list = [2e-5, 1e-5, 2e-5/3]
    k_list  = [29.928743, 26.798465, 28.165698]

    plot_convergence_and_gci(k_list, dx_list, p_formal=2.0)