

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

def build_source_term_MMS():

    rs, ts, Rs, C0s, As, omegas, Ds, ks = sp.symbols(
        "rs ts Rs C0 A omega D k"
    )

    C = C0s + As*(1 - (rs/Rs)**2)*sp.sin(omegas*ts)*(Rs-rs)*(rs**2)

    dC_dt = sp.diff(C, ts)
    Cr = sp.diff(C, rs)
    Crr = sp.diff(Cr, rs)

    term2 = Crr + (1/rs)*Cr
    term_r0 = sp.limit(term2, rs, 0)

    lap = sp.Piecewise((term_r0, sp.Eq(rs,0)), (term2, True))

    S = sp.simplify(dC_dt - Ds*lap + ks*C)

    return sp.lambdify(
        (rs, ts, Rs, Ds, ks, C0s, As, omegas),
        S,
        modules="numpy"
    )


source_term = build_source_term_MMS()

def source_term_MMS(r, t, R, D, k, p) -> float:

    if abs(r) < 1e-14:
        r = 0.0

    return float(
        source_term( r,t,R,D,k,p.C0,p.A, p.omega)
    )