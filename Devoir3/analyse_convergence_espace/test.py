from Devoir3.problem_definition.lbm_devoir3 import Generate_sample, LBM

if __name__ == "__main__":
    filename     = 'fiber_mat.tiff'
    seed         = 101
    deltaP       = 0.1
    NX           = 100
    poro         = 0.9
    mean_fiber_d = 12.5
    std_d        = 2.85
    dx           = 2e-6

    d_equivalent = Generate_sample(seed, filename, mean_fiber_d, std_d, poro, NX, dx)
    LBM(filename, NX, deltaP, dx, d_equivalent)