"""
Script principal exécutant la simulation et les post-traitements du problème de diffusion.
"""
import numpy as np
from mesh_and_parameters import ProblemParameters
from mms_solution import MMSParams, mms_iteration

from finite_differences_schemes import solve_unsteady_scheme

from post_processing import (
    error_norms,
    compute_convergence_orders,
    plot_error_convergence_space,
    plot_error_convergence_time,
    plot_heatmaps_num_mms_error,
    plot_mms_solution_profiles,
    plot_mms_source_profiles, 
    plot_original_problem_profiles
)


def main():
    param_etude_conv = ProblemParameters(
        r=0.5,
        d_eff=1,
        k=1,
        c_e=20.0,
        t_final=1.0
    )
    param_normal = ProblemParameters(
        r=0.5,
        d_eff=1e-10,
        k=4e-9,
        c_e=20,
        t_final=4e9,
    )
    mms_param = MMSParams(
        C0=20.0,
        A=2.0,
        omega=1
    )

    """ PLOT MMS AND SOURCE TERM """
    times_to_plot = [
        0.0,
        0.25 * param_etude_conv.t_final,
        0.50 * param_etude_conv.t_final,
        0.75 * param_etude_conv.t_final,
        param_etude_conv.t_final,
    ]
    plot_mms_solution_profiles(param_etude_conv, mms_param, num_nodes=200, times_to_plot=times_to_plot)
    plot_mms_source_profiles(param_etude_conv, mms_param, num_nodes=200, times_to_plot=times_to_plot)

    """ SPACE CONVERGENCE """
    n_profile_list = [5, 10, 20, 50, 100, 150, 200, 250, 300]
    dt_space_convergence = 1e-4
    # n_profile_list = [5, 10, 20, 50, 100, 200, 400, 500]
    # dt_fixed = 1e-5

    L1_space = {}
    L2_space = {}
    L_inf_space = {}
    for n_profile in n_profile_list:
        # mesh array, time array, concentration array (time x n_profile)
        r_mesh, time_array, num_results = solve_unsteady_scheme(
            param=param_etude_conv,
            n_profile=n_profile,
            dt=dt_space_convergence,
            mms=mms_param
        )
        # mesh array, time array, concentration array (time x n_profile)
        _, _, mms_results = mms_iteration(
            param=param_etude_conv,
            n_profile=n_profile,
            dt=dt_space_convergence,
            mms=mms_param
        )
        # L1, L2, L_inf
        errors_tmp = error_norms(
            c_num_hist=num_results,
            c_mms_hist=mms_results,
            r_mesh=r_mesh,
            time_array=time_array
        )

        # calculate and store errors
        h = float(r_mesh[1] - r_mesh[0])
        L1_space[h] = errors_tmp[0]
        L2_space[h] = errors_tmp[1]
        L_inf_space[h] = errors_tmp[2]

    _, _, _, _ = compute_convergence_orders(
        l1_errors_dict=L1_space,
        l2_errors_dict=L2_space,
        linf_errors_dict=L_inf_space
    )
    plot_error_convergence_space(
        l1_errors_dict=L1_space,
        l2_errors_dict=L2_space,
        linf_errors_dict=L_inf_space
    )


    """ TIME CONVERGENCE """
    dt_list_time = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4]
    n_profile_time = 800

    L1_time = {}
    L2_time = {}
    L_inf_time = {}
    for dt in dt_list_time:
        # mesh array, time array, concentration array (time x n_profile)
        r_mesh, time_array, num_results = solve_unsteady_scheme(
            param=param_etude_conv,
            n_profile=n_profile_time,
            dt=dt,
            mms=mms_param
        )
        # mesh array, time array, concentration array (time x n_profile)
        _, _, mms_results = mms_iteration(
            param=param_etude_conv,
            n_profile=n_profile_time,
            dt=dt,
            mms=mms_param
        )
        # L1, L2, L_inf
        errors_tmp = error_norms(
            c_num_hist=num_results,
            c_mms_hist=mms_results,
            r_mesh=r_mesh,
            time_array=time_array
        )

        # calculate and store errors
        L1_time[float(dt)] = errors_tmp[0]
        L2_time[float(dt)] = errors_tmp[1]
        L_inf_time[float(dt)] = errors_tmp[2]

    _, _, _, _ = compute_convergence_orders(
        l1_errors_dict=L1_time,
        l2_errors_dict=L2_time,
        linf_errors_dict=L_inf_time
    )
    plot_error_convergence_time(
        l1_errors_dict=L1_time,
        l2_errors_dict=L2_time,
        linf_errors_dict=L_inf_time
    )

    """ HEATMAP """
    n_profile_heatmap = 800
    dt_heatmap = 5e-4
    plot_heatmaps_num_mms_error(
        param=param_etude_conv,
        n_profile=n_profile_heatmap,
        dt=dt_heatmap,
        mms=mms_param
    )

    """ ORIGINAL PROBLEM PLOT  """

    times_original = np.linspace(0, param_normal.t_final, 10)

    plot_original_problem_profiles(
        param=param_normal,
        n_profile=11,
        dt=5e6,
        times_to_plot=times_original
    )

if __name__ == "__main__":
    main()
