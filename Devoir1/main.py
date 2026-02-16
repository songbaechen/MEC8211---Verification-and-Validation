"""
Script principal exécutant la simulation et les post-traitements du problème de diffusion.
"""
from post_processing import plot_profiles, plot_error_norms
from mesh_and_parameters import ProblemParameters

def main():
    """
    Point d’entrée du programme: définition des paramètres, résolution et visualisation.
    """
    # Paramètres
    r = 0.5            # m -> D = 1
    s = 2e-8           # mol/m^3/s
    d_eff = 1e-10      # m^2/s
    c_e = 20.0         # mol/m^3
    param = ProblemParameters(r=r, s=s, d_eff=d_eff, c_e=c_e)

    print("=== Paramètres de la simulation ===")
    print(f"R     = {r} m")
    print(f"S     = {s} mol/m^3/s")
    print(f"D_eff = {d_eff} m^2/s")
    print(f"C_e   = {c_e} mol/m^3")
    print("==================================\n")

    # (a) Profil de concentration stationnaire
    plot_profiles(param=param, n_profile=5, plot_1=True, plot_2=True)

    # (b) Vérification du code : erreurs L1, L2, L_infini
    n_profil_list = [5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240]
    plot_error_norms(param, n_profil_list, plot_1=True, plot_2=True)


if __name__ == "__main__":
    main()
