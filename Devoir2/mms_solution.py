

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
    omerga = p.omega
    mms_sol = C0 + A * (1.0 - (r/R) ** 2) * np.sin(omerga * t)

    return mms_sol

def dirichlet_bord_mms(t: float, R: float, p: MMSParams) -> float: 
    """
    Condition de Dirichlet au bord issue de la MMS :
    C(R,t)
    """
    return mms_function(R, t, R, p)