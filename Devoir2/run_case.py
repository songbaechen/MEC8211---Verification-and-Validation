"""
Exécute un seul cas de convergence MMS et écrit les erreurs dans un fichier CSV.

Ce fichier est volontairement conçu pour être modifié automatiquement par un
script Bash via substitution de chaînes de caractères, dans l'esprit du labo.
"""

import csv

from mesh_and_parameters import ProblemParameters
from mms_solution import MMSParams, mms_iteration
from finite_differences_schemes import solve_unsteady_scheme
from post_processing import error_norms


MODE = "MODE_PLACEHOLDER"
N_PROFILE = N_PROFILE_PLACEHOLDER
DT = DT_PLACEHOLDER
OUTPUT_CSV = "OUTPUT_CSV_PLACEHOLDER"


def main() -> None:
    """
    Exécute un seul cas numérique et ajoute les erreurs au fichier CSV.
    """
    param_etude_conv = ProblemParameters(
        r=0.5,
        d_eff=1.0,
        k=1.0,
        c_e=20.0,
        t_final=1.0
    )

    mms_param = MMSParams(
        C0=20.0,
        A=2.0,
        omega=1.0
    )

    r_mesh, time_array, num_results = solve_unsteady_scheme(
        param=param_etude_conv,
        n_profile=N_PROFILE,
        dt=DT,
        mms=mms_param
    )

    _, _, mms_results = mms_iteration(
        param=param_etude_conv,
        n_profile=N_PROFILE,
        dt=DT,
        mms=mms_param
    )

    l1_err, l2_err, linf_err = error_norms(
        c_num_hist=num_results,
        c_mms_hist=mms_results,
        r_mesh=r_mesh,
        time_array=time_array
    )

    h = float(r_mesh[1] - r_mesh[0])

    write_header = False
    try:
        with open(OUTPUT_CSV, "r", encoding="utf-8"):
            pass
    except FileNotFoundError:
        write_header = True

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        if write_header:
            writer.writerow(["mode", "n_profile", "dt", "h", "L1", "L2", "Linf"])

        writer.writerow([
            MODE,
            N_PROFILE,
            DT,
            h,
            l1_err,
            l2_err,
            linf_err
        ])

    print(
        f"[{MODE}] n_profile={N_PROFILE}, dt={DT:.6e}, h={h:.6e}, "
        f"L1={l1_err:.6e}, L2={l2_err:.6e}, Linf={linf_err:.6e}"
    )


if __name__ == "__main__":
    main()