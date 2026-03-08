"""
Schéma numérique implicite d'ordre 2 en espace pour la diffusion radiale.

Ce module contient la fonction de résolution transitoire du problème de
diffusion-réaction en coordonnées radiales, avec possibilité d'utiliser une
solution manufacturée (MMS) pour la vérification du code.
"""

import numpy as np

from mesh_and_parameters import ProblemParameters, create_mesh
from mms_solution import MMSParams, mms_function, dirichlet_bord_mms, source_term_MMS


def solve_unsteady_scheme(
    param: ProblemParameters,
    n_profile: int,
    dt: float,
    mms: MMSParams | None = None
):
    """
    Résout le problème transitoire de diffusion-réaction radiale.

    La discrétisation temporelle est implicite, tandis que la discrétisation
    spatiale utilise un schéma centré d'ordre 2 pour les nœuds intérieurs.
    La condition de symétrie au centre est imposée par une condition de
    Neumann, et la condition au bord externe est de type Dirichlet.

    Si un jeu de paramètres MMS est fourni, la condition initiale, le terme
    source et la condition de bord sont construits à partir de la solution
    manufacturée.

    Parameters
    ----------
    param : ProblemParameters
        Structure contenant les paramètres physiques du problème
        (rayon, diffusivité effective, coefficient de réaction, temps final,
        concentration au bord, etc.).
    n_profile : int
        Nombre de nœuds du maillage radial.
    dt : float
        Pas de temps utilisé pour l'intégration temporelle.
    mms : MMSParams | None, optional
        Paramètres associés à la solution manufacturée. Si `None`, le problème
        physique standard est résolu.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Un tuple contenant :
        - `r_mesh` : le maillage radial ;
        - `t` : le vecteur temps ;
        - `c_hist` : l'historique temporel de la concentration, de dimension
          `(nt, n_profile)`.
    """
    # Initialisation des paramètres physiques
    D = float(param.d_eff)
    k = float(param.k)
    t_final = float(param.t_final)

    # Construction du maillage radial et du vecteur temps
    r_mesh, dr = create_mesh(param.r, n_profile)
    nt = int(np.ceil(t_final / dt)) + 1
    t = np.linspace(0.0, dt * (nt - 1), nt)

    # Condition initiale
    # Problème physique : C(r, 0) = 0
    # Cas MMS : C(r, 0) = C0 ; imposée par la solution manufacturée
    if mms is None:
        cn = np.zeros(n_profile, dtype=np.float64)
    else:
        cn = np.array(
            [mms_function(float(r_i), 0.0, float(param.r), mms) for r_i in r_mesh],
            dtype=np.float64
        )

    # Stockage de l'historique temporel de la solution
    c_hist = np.zeros((nt, n_profile), dtype=np.float64)
    c_hist[0, :] = cn.copy()

    for n in range(nt - 1):
        t_np1 = float(t[n + 1])

        # Initialisation du système linéaire A * C^{n+1} = b
        a = np.zeros((n_profile, n_profile), dtype=np.float64)
        b = np.zeros(n_profile, dtype=np.float64)

        # Condition de Neumann au centre (symétrie)
        # Approximation décentrée d'ordre 2 :
        # dC/dr = 0 => -3C0 + 4C1 - C2 = 0
        a[0, 0] = -3.0
        a[0, 1] = 4.0
        a[0, 2] = -1.0
        b[0] = 0.0

        # Assemblage des équations aux nœuds intérieurs
        for i in range(1, n_profile - 1):
            ri = r_mesh[i]

            l_i_moins_1 = (1.0 / dr**2) - (1.0 / (2.0 * dr * ri))
            l_i = -2.0 / dr**2
            l_i_plus_1 = (1.0 / dr**2) + (1.0 / (2.0 * dr * ri))

            a[i, i - 1] = -D * l_i_moins_1
            a[i, i] = (1.0 / dt + k) - D * l_i
            a[i, i + 1] = -D * l_i_plus_1

            b[i] = (1.0 / dt) * cn[i]

            # Ajout du terme source dans le cas MMS
            if mms is not None:
                b[i] += source_term_MMS(
                    ri,
                    t_np1,
                    float(param.r),
                    D,
                    k,
                    mms,
                )

        # Condition de Dirichlet au bord externe
        a[n_profile - 1, :] = 0.0
        a[n_profile - 1, n_profile - 1] = 1.0

        if mms is None:
            b[n_profile - 1] = float(param.c_e)
        else:
            b[n_profile - 1] = dirichlet_bord_mms(t_np1, float(param.r), mms)

        # Résolution du système linéaire au temps n+1
        cn_plus_1 = np.linalg.solve(a, b)

        # Mise à jour de l'historique et de la solution courante
        c_hist[n + 1, :] = cn_plus_1
        cn = cn_plus_1

    return r_mesh, t, c_hist
