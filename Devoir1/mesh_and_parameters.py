"""
Module de création du maillage radial et de définition des paramètres du problème
de diffusion en géométrie cylindrique.

Contient la fonction de génération du maillage uniforme et la classe regroupant
les paramètres de la simulation.
"""
from dataclasses import dataclass
import numpy as np


def create_mesh(r, n_profile):
    """
    Fonction pour créer le maillage. Le maillage est une grille radiale uniforme en 1D [0, r].
    Retourne le maillage ainsi que le pas entre les nœuds.

    :param r: Rayon total du profil/cylindre [float]
    :param n_profile: Nombre de nœuds [int]

    :return: r, dr [tuple[np.array, float]]
    """
    if n_profile < 3:
        raise ValueError("Il faut avoir au moins 3 noeuds pour résoudre le problème")

    r = np.linspace(0.0, r, n_profile, dtype=np.float64)
    dr = r[1] - r[0]

    return r, dr


@dataclass
class ProblemParameters:
    """
    Classe contenant les paramètres de la simulation.
    """
    r: float        # Rayon du cylindre/profil
    d_eff: float    # Coefficient de diffusion
    s: float        # Quantité de sel
    c_e: float      # Concentration à la surface
