# Diffusion radiale – Vérification par MMS

Ce projet implémente un schéma numérique implicite pour résoudre un problème de diffusion-réaction radiale.

La vérification du code est réalisée avec la **Method of Manufactured Solutions (MMS)** afin d'étudier la **convergence en espace et en temps**.

Le projet permet de :

- résoudre le problème numérique
- comparer avec la solution MMS
- calculer les normes d'erreur L1, L2 et L∞
- analyser l'ordre de convergence
- générer automatiquement les graphes de convergence

---

# Installation

Créer un environnement Python (recommandé **Python 3.10**) puis installer les dépendances :

```bash
pip install -r requirements.txt
```

Packages requis :

- numpy
- matplotlib
- pandas
- sympy
- pylint

---

# Structure du projet

```
.
├── main.py
├── finite_differences_schemes.py
├── mesh_and_parameters.py
├── mms_solution.py
├── post_processing.py
│
├── run_case.py
├── analyse_de_convergence.py
├── show_convergence_plots.py
│
├── run_convergence.sh
├── space_cases.txt
├── time_cases.txt
│
└── requirements.txt
```

### Modules principaux

**solve_unsteady_scheme**

Résout l'équation diffusion-réaction radiale avec un schéma implicite.

**mms_solution.py**

Contient :

- la solution manufacturée
- le terme source associé
- les fonctions permettant d'évaluer la solution exacte.

**post_processing.py**

Permet de :

- calculer les normes d'erreur
- calculer les ordres de convergence
- tracer les graphes de convergence.

---

# Script principal

Le fichier `main.py` permet de :

- tracer la solution MMS
- effectuer l'analyse de convergence en espace
- effectuer l'analyse de convergence en temps
- générer des heatmaps de l'erreur
- résoudre le problème physique original.

Exécution :

```bash
python main.py
```

---

# Analyse de convergence automatisée

L'analyse peut être automatisée avec le script bash :

```
run_convergence.sh
```

Ce script :

1. lit les cas dans  
   `space_cases.txt` et `time_cases.txt`
2. exécute les simulations pour chaque cas
3. écrit les erreurs dans un fichier CSV
4. génère les graphes de convergence
5. affiche les figures à la fin.

Résultats générés :

```
convergence_space.png
convergence_time.png
results_space.csv
results_time.csv
```

Les graphes sont **sauvegardés automatiquement** puis **affichés à la fin** de l'exécution.

---

# Comment lancer l'analyse complète

Sous Linux / WSL / Git Bash :

```bash
bash run_convergence.sh
```

Le script va :

- exécuter tous les cas de convergence en espace
- exécuter tous les cas de convergence en temps
- générer les figures de convergence
- afficher les figures finales.

Les figures affichées à la fin sont chargées par :

```
show_convergence_plots.py
```

---

# Définition des cas de convergence

Les cas sont définis dans deux fichiers texte.

---

## Convergence en espace

Fichier :

```
space_cases.txt
```

Format :

```
n_profile dt
```

Exemple :

```
5   1e-2
10  1e-2
20  1e-2
40  1e-2
80  1e-2
160 1e-2
```

Dans ce cas :

- le pas spatial varie
- le pas de temps reste fixe.

---

## Convergence en temps

Fichier :

```
time_cases.txt
```

Format :

```
n_profile dt
```

Exemple :

```
400 1e-1
400 5e-2
400 2.5e-2
400 1.25e-2
400 6.25e-3
```

Dans ce cas :

- le pas de temps varie
- le maillage spatial reste fixe.

---

# Résultat attendu

Les graphes obtenus doivent montrer :

- **ordre 2 en espace**
- **ordre 1 en temps**

ce qui correspond aux propriétés théoriques du schéma :

- discrétisation **centrée d'ordre 2 en espace**
- schéma **Euler implicite d'ordre 1 en temps**

---

# Remarque

Les graphes sont :

- **sauvegardés automatiquement**
- **affichés seulement à la fin de l'analyse**

afin d'éviter que les figures bloquent l'exécution des simulations.