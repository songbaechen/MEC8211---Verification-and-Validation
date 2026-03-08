"""
Définition de la solution manufacturée (MMS) et du terme source associé.

Ce module contient :
- une structure de données pour les paramètres de la MMS ;
- la fonction analytique de la solution manufacturée ;
- la condition de Dirichlet au bord dérivée de cette solution ;
- la construction symbolique du terme source permettant de satisfaire
  exactement l'équation différentielle.
"""

from dataclasses import dataclass

import numpy as np
import sympy as sp

from mesh_and_parameters import *


@dataclass
class MMSParams:
    """
    Paramètres de la solution manufacturée.

    Attributes
    ----------
    C0 : float
        Constante associée à la condition de Dirichlet au bord, soit la
        concentration à la surface.
    A : float
        Amplitude de la perturbation temporelle.
    omega : float
        Pulsation temporelle de la solution manufacturée, en rad/s.
    """

    C0: float
    A: float
    omega: float


def mms_function(r: float, t: float, R: float, p: MMSParams) -> float:
    """
    Évalue la solution manufacturée en un point de l'espace et du temps.

    La forme choisie est construite de manière à respecter :
    - une condition de symétrie au centre ;
    - une condition de Dirichlet imposée au bord externe.

    La solution utilisée est :

    C_mms(r, t) = C0 + A * (1 - (r / R)^2) * sin(omega * t)
                  * (R - r) * (0 - r)^2

    Parameters
    ----------
    r : float
        Position radiale.
    t : float
        Temps.
    R : float
        Rayon externe du domaine.
    p : MMSParams
        Paramètres de la solution manufacturée.

    Returns
    -------
    float
        Valeur de la solution manufacturée au point (r, t).
    """
    C0 = p.C0
    A = p.A
    omega = p.omega

    mms_sol = (
        C0
        + A
        * (1.0 - (r / R) ** 2)
        * np.sin(omega * t)
        * (R - r) ** 1
        * (0.0 - r) ** 2
    )

    return mms_sol


def dirichlet_bord_mms(t: float, R: float, p: MMSParams) -> float:
    """
    Évalue la condition de Dirichlet au bord issue de la MMS.

    Cette condition correspond simplement à la valeur de la solution
    manufacturée au bord externe, soit C(R, t).

    Parameters
    ----------
    t : float
        Temps.
    R : float
        Rayon externe du domaine.
    p : MMSParams
        Paramètres de la solution manufacturée.

    Returns
    -------
    float
        Valeur de la concentration au bord externe au temps t.
    """
    return mms_function(R, t, R, p)


def build_source_term_MMS():
    """
    Construit symboliquement le terme source associé à la MMS.

    Le terme source est obtenu en injectant la solution manufacturée dans
    l'équation de diffusion-réaction radiale. Le calcul est effectué
    symboliquement avec SymPy, puis converti en fonction numérique avec
    `lambdify`.

    Une attention particulière est portée au point r = 0, puisque
    l'expression du laplacien radial contient un terme en 1 / r. La valeur
    au centre est donc obtenue en prenant la limite analytique lorsque
    r tend vers 0.

    Returns
    -------
    callable
        Fonction numérique évaluant le terme source MMS sous la forme :

        S(r, t, R, D, k, C0, A, omega)
    """
    rs, ts, Rs, C0s, As, omegas, Ds, ks = sp.symbols(
        "rs ts Rs C0 A omega D k"
    )

    # Définition symbolique de la solution manufacturée
    C = C0s + As * (1 - (rs / Rs) ** 2) * sp.sin(omegas * ts) * (Rs - rs) * (rs ** 2)

    # Dérivées nécessaires à la construction du terme source
    dc_dt = sp.diff(C, ts)
    dc_dr = sp.diff(C, rs)
    dc_dr2 = sp.diff(dc_dr, rs)

    # Laplacien radial : d²C/dr² + (1/r) dC/dr
    term2 = dc_dr2 + (1 / rs) * dc_dr

    # Valeur régulière du laplacien au centre obtenue par passage à la limite
    term_r0 = sp.limit(term2, rs, 0)

    # Définition pièce par pièce pour éviter la singularité en r = 0
    laplacian = sp.Piecewise((term_r0, sp.Eq(rs, 0)), (term2, True))

    # Terme source de l'équation MMS
    source = sp.simplify(dc_dt - Ds * laplacian + ks * C)

    return sp.lambdify(
        (rs, ts, Rs, Ds, ks, C0s, As, omegas),
        source,
        modules="numpy"
    )


source_term = build_source_term_MMS()


def source_term_MMS(r, t, R, D, k, p) -> float:
    """
    Évalue numériquement le terme source associé à la MMS.

    Une petite correction numérique est appliquée près de r = 0 afin
    d'éviter les effets liés aux erreurs d'arrondi lors de l'évaluation
    de la fonction symbolique convertie.

    Parameters
    ----------
    r : float
        Position radiale.
    t : float
        Temps.
    R : float
        Rayon externe du domaine.
    D : float
        Coefficient de diffusion.
    k : float
        Coefficient de réaction.
    p : MMSParams
        Paramètres de la solution manufacturée.

    Returns
    -------
    float
        Valeur du terme source au point (r, t).
    """
    if abs(r) < 1e-14:
        r = 0.0

    return float(
        source_term(r, t, R, D, k, p.C0, p.A, p.omega)
    )


def mms_iteration(
    param: ProblemParameters,
    n_profile: int,
    dt: float,
    mms: MMSParams
):
    """
    Construit l'historique exact de la solution manufacturée sur tout le maillage
    et pour tous les pas de temps.

    Le format retourné est identique à celui de `solve_unsteady_scheme`, soit
    un tableau `c_hist` de dimension `(nt, n_profile)` contenant la solution
    analytique évaluée à chaque nœud radial et à chaque instant.

    Parameters
    ----------
    param : ProblemParameters
        Paramètres physiques du problème.
    n_profile : int
        Nombre de nœuds du maillage radial.
    dt : float
        Pas de temps.
    mms : MMSParams
        Paramètres de la solution manufacturée.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Un tuple contenant :
        - `r_mesh` : le maillage radial ;
        - `t` : le vecteur temps ;
        - `c_hist_exact` : l'historique exact de la solution manufacturée,
          de dimension `(nt, n_profile)`.
    """
    # Construction du maillage radial
    r_mesh, _ = create_mesh(param.r, n_profile)

    # Construction du vecteur temps
    t_final = float(param.t_final)
    nt = int(np.ceil(t_final / dt)) + 1
    t = np.linspace(0.0, dt * (nt - 1), nt)

    # Évaluation de la MMS à tous les temps et à tous les nœuds
    c_hist_exact = np.zeros((nt, n_profile), dtype=np.float64)

    for n, t_n in enumerate(t):
        c_hist_exact[n, :] = np.array(
            [mms_function(float(r_i), float(t_n), float(param.r), mms) for r_i in r_mesh],
            dtype=np.float64
        )

    return r_mesh, t, c_hist_exact
