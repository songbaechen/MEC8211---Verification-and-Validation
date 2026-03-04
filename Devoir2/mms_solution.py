

import numpy as np
from dataclasses import dataclass

@dataclass
class MMSParams: 
    """
    Classe contenant les paramètres de la solution manufacturée.
    """

    C0: float   # Dirichlet au bord
    A:float     # Amplitude
    omega:float # pulsation [rad/s]

def mms_function(r:float, t:float, R:float, p:MMSParams) -> float: 
    """
    Solution manufacturée :
    C_mms(r,t) = C0 + A*(1 - (r/R)^2)*sin(omega*t)
    """

    C0 = p.C0
    A = p.A
    omega = p.omega
    mms_sol = C0 + A * (1.0 - (r/R) ** 2) * np.sin(omega * t)

    return mms_sol

def dirichlet_bord_mms(t: float, R: float, p: MMSParams) -> float: 
    """
    Condition de Dirichlet au bord issue de la MMS :
    C(R,t)
    """
    return mms_function(R, t, R, p)

def source_term_MMS(r: float, t: float, R: float, D: float, k: float, p: MMSParams) -> float: 
    """
    Terme source de la MMS
    """
    C0 = p.C0
    A = p.A
    omega = p.omega
    C = mms_function(r, t, R, p)

    # dC/dt
    dC_dt = A * (1.0 - (r / R) ** 2) * omega * np.cos(omega * t)
    
    # Dérivéees en r 
    Cr = A * (-2.0 * r / (R ** 2)) * np.sin(omega * t)
    Crr = A * (-2.0 / (R ** 2)) * np.sin(omega * t)

    # quand r -> 0, on fait la limite 
    terme_r_0 = (-4.0 * A / (R ** 2)) * np.sin(omega * t)

    if abs(r) < 1e-14: 
        term_2 = terme_r_0
    else: 
        term_2 = Crr + (1.0 / r) * Cr

    return dC_dt - D * term_2 + k * C