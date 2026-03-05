

import numpy as np
import sympy as sp
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
    dirichlet_BC_curve = R
    neumann_BC_curve = 0.0
    mms_sol = C0 + A * (1.0 - (r/R) ** 2) * np.sin(omega * t) * (dirichlet_BC_curve - r)**1 * (neumann_BC_curve - r)**2

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

    # variables symboliques
    rs, ts, Rs, C0s, As, omegas = sp.symbols('rs ts Rs C0 A omega')

    # MMS 
    C = C0s + As*(1 - (rs/Rs)**2)*sp.sin(omegas*ts)*(Rs-rs)*(rs**2)

    # dérivées symboliques
    dC_dt = sp.diff(C, ts)
    Cr = sp.diff(C, rs)
    Crr = sp.diff(Cr, rs)

    term2 = Crr + (1/rs)*Cr
    terme_r_0 = sp.limit(term2, rs, 0)

    # substitution valeurs
    subs = {rs:r, ts:t, Rs:R, C0s:C0, As:A, omegas:omega}

    C_val = float(C.subs(subs))
    dC_dt_val = float(dC_dt.subs(subs))
    Cr_val = float(Cr.subs(subs))
    Crr_val = float(Crr.subs(subs))
    terme_r_0_val = float(terme_r_0.subs({ts:t, Rs:R, C0s:C0, As:A, omegas:omega}))

    if abs(r) < 1e-14:
        term_2 = terme_r_0_val
    else:
        term_2 = Crr_val + (1.0/r)*Cr_val

    return dC_dt_val - D*term_2 + k*C_val