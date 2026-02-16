# Diffusion radiale – Méthodes aux différences finies

## Description

Ce projet résout numériquement le problème de diffusion radiale stationnaire dans un cylindre infini avec un terme source constant.

Objectifs :

- Calcul du profil de concentration C(r)
- Comparaison des schémas numériques avec la solution analytique
- Calcul des erreurs L1, L2 et L∞
- Vérification de l’ordre de convergence

---

## Structure du projet
├── main.py <br>
├── mesh_and_parameters.py <br>
├── analytical_solution.py <br>
├── finite_differences_schemes.py <br>
├── post_processing.py <br>
├── requirements.txt <br>
└── README.md <br>

### Description des modules :

- `main.py` : script principal exécutant la simulation
- `mesh_and_parameters.py` : création du maillage et définition des paramètres
- `analytical_solution.py` : calcul de la solution analytique
- `finite_differences_schemes.py` : implémentation des schémas numériques
- `post_processing.py` : calcul des erreurs, ordres de convergence et plots
---

## Schémas numériques

Deux schémas aux différences finies sont implémentés :

- Schéma 1 : discrétisation standard
- Schéma 2 : discrétisation d’ordre supérieur au centre

Le système linéaire est résolu avec :

```python
np.linalg.solve(A, b)