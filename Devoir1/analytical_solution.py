import numpy as np

def analytic_solution(r, R, D_eff, S, C_e): 
    
    r = np.asarray(r, dtype = float)   
    
    sol = C_e + (S / (4 * D_eff) ) * (r**2 - R**2)

    return sol