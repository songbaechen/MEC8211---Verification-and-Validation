# Devoir 3 — Validation d'un modèle LBM de perméabilité

MEC8211 · Hiver 2026 · Polytechnique Montréal

## Structure

```
Devoir3/
├── problem_definition/
│   ├── lbm_devoir3.py            # Code LBM (boîte noire) + génération de géométrie
│   └── problem_definition.py     # Dataclasses des paramètres
├── analyse_convergence_espace/
│   └── richardson.py             # (A) Convergence en maillage → u_num
├── analyse_monte_carlo/
│   ├── monte_carlo.py            # (B) Propagation d'incertitude → u_input
│   ├── incertitude_experimentale.py  # (C) Incertitude expérimentale → u_D
│   ├── simulation_erreor.py      # (D) Erreur de simulation E = S − D
│   └── plot_convergence.py       # Graphiques GCI / Richardson
├── main.py                       # (E) Assemblage V&V20 → δ_model
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

> Numba est requis pour l'accélération JIT du solveur LBM.

## Exécution

Chaque question peut être lancée indépendamment :

```bash
python -m analyse_convergence_espace.richardson   # Question A
python -m analyse_monte_carlo.monte_carlo          # Question B
python -m analyse_monte_carlo.incertitude_experimentale  # Question C
python -m analyse_monte_carlo.simulation_erreor    # Question D
python main.py                                     # Question E (tout assembler)
```

## Questions

| # | Description | Sortie clé |
|---|-------------|------------|
| A | Convergence en maillage (GCI) | u_num |
| B | Monte-Carlo sur la porosité | u_input⁻, u_input⁺ |
| C | Incertitude expérimentale | u_D |
| D | Erreur de simulation | E = S − D |
| E | Intervalle δ_model (V&V20) | δ_lower ≤ δ_model ≤ δ_upper |