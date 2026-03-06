"""
Script principal exécutant la simulation et les post-traitements du problème de diffusion.
"""
from mesh_and_parameters import ProblemParameters
from mms_solution import MMSParams

from finite_differences_schemes import solve_unsteady_scheme

from post_processing import (
    convergence_space,
    convergence_time,
    plot_mms_solution_profiles,
    plot_mms_source_profiles,
    plot_error_convergence_space,
    plot_error_convergence_time,
    plot_heatmaps_num_mms_error,
)

def main():
    """
    Point d’entrée du programme: définition des paramètres, résolution et visualisation.
    """
    # Paramètres
    r = 0.5            # m -> D = 1
    d_eff = 1 # 1e-10      # m^2/s
    k_reac = 4 # 4e-9      # 1/s
    c_e = 20.0         # mol/m^3
    t_final = 1.0 # 2.0    # s
    dt = 1.0e-3         # s

    param = ProblemParameters(
    r=r,
    d_eff=d_eff,
    k=k_reac,
    dt=dt,
    t_final=t_final,
    c_e=c_e,
    )

    print("=== Paramètres du problème ===")
    print(f"R       = {param.r:.6g} m")
    print(f"D_eff   = {param.d_eff:.6e} m^2/s")
    print(f"k       = {param.k:.6e} 1/s")
    print(f"dt      = {param.dt:.6e} s")
    print(f"t_final = {param.t_final:.6e} s")
    print(f"C_e     = {param.c_e:.6g}")
    print("==============================\n")

    c0 = 20.0       # mol/m^3
    amp = 2.0       
    omega = 3.14  # rad/s

    mms = MMSParams(C0=c0, A=amp, omega=omega)

    print("=== Paramètres MMS ===")
    print(f"C0    = {mms.C0:.6g}")
    print(f"A     = {mms.A:.6g}")
    print(f"omega = {mms.omega:.6e} rad/s")
    print("======================\n")

    times_to_plot = [
        0.0,
        0.25 * param.t_final,
        0.50 * param.t_final,
        0.75 * param.t_final,
        param.t_final,
    ]

    plot_mms_solution_profiles(param, mms, num_nodes=200, times_to_plot=times_to_plot)
    plot_mms_source_profiles(param, mms, num_nodes=200, times_to_plot=times_to_plot)

    n_list_space = [5, 10, 50, 100, 500, 1000] #, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    dt_fixed = 1e-4

    space_results = convergence_space(
        problem=param,
        mms=mms,
        num_nodes_list=n_list_space,
        dt_fixed=dt_fixed,
    )

    print("")
    plot_error_convergence_space(space_results)

    dt_list_time = [1e-1, 5e-2] #, 2.5e-2, 1.25e-2]
    n_fixed_time = 30

    time_results = convergence_time(
        problem=param,
        mms=mms,
        dt_list=dt_list_time,
        num_nodes_fixed=n_fixed_time,
    )
    plot_error_convergence_time(time_results)

    n_heatmap = 20
    r_mesh, time_array, c_hist = solve_unsteady_scheme(param, n_heatmap, mms)
    plot_heatmaps_num_mms_error(c_hist, r_mesh, time_array, param, mms)



if __name__ == "__main__":
    main()
