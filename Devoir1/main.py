from post_processing import plot_profiles, plot_error_norms

def main():
    # Paramètres 
    R = 0.5            # m -> D = 1
    S = 2e-8           # mol/m^3/s
    D_eff = 1e-10      # m^2/s
    C_e = 20.0         # mol/m^3

    print("=== Paramètres de la simulation ===")
    print(f"R     = {R} m")
    print(f"S     = {S} mol/m^3/s")
    print(f"D_eff = {D_eff} m^2/s")
    print(f"C_e   = {C_e} mol/m^3")
    print("==================================\n")

    # (a) Profil de concentration stationnaire
    N_profile = 101
    plot_profiles(R, D_eff, S, C_e, N_profile)

    # (b) Vérification du code : erreurs L1, L2, L_infini
    N_list = [20, 40, 80, 160, 320] 
    plot_error_norms(R, D_eff, S, C_e, N_list)


if __name__ == "__main__":
    main()