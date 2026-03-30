from dataclasses import dataclass

@dataclass
class SampleParameters:
    """
    Parameters of function Generate_sample.
    """
    seed     : int      # graine du générateur aléatoire (0 = aléatoire)
    filename : str      # nom du fichier TIFF de sortie
    mean_d   : float    # diamètre moyen des fibres [µm]
    std_d    : float    # écart-type des diamètres  [µm]
    poro     : float    # porosité cible
    nx       : int      # taille latérale du domaine [cellules]
    dx       : float    # taille d'une cellule [m]


@dataclass
class ProblemParameters:
    """
    Parameters of LBM function.
    """
    filename     : str      # fichier TIFF de la structure de fibres
    NX           : int      # taille du domaine (carré NX×NX)
    deltaP       : float    # chute de pression [Pa]
    dx           : float    # taille d'une cellule [m]
    d_equivalent : float    # diamètre équivalent des fibres [µm]
