# Changelog

Toutes les modifications notables de ce projet seront documentées dans ce fichier.

## [Unreleased]

## [2026-01-09] - Maelig
### Feat
- `optuna` required pour l'optimisation des modèles (`af063a4`).

### Docs
- Nettoyage des commentaires et docstrings (`5ac0f17`).
- Mise à jour README (`829f345`).

### Fix
- Correction des tests liés au parallélisme (`e6474f6`).

### Test
- Tests des méthodes `fit`, `predict`, `eval` (`3e9138e`).

### Hotfix
- Correction `fit`/`predict` avant merge V1 (`bc021e3`).

### Merge
- Fusion des branches `automl-feat` et `refacto` dans `main` (`df8549e`, `33c4f4d`).
- Mise à jour générale des tests (`ebaf5d0`).

## [2026-01-06] - s2200959
### Feat
- Ajout de la méthode `predict` (`841c9a9`).
- Ajout de `optimize_model_library` (`497bcdf`).

## [2025-12-09] - s2200959 / Maelig
### Feat
- `fit` + `eval` des modèles (`bc76f05`).
- Ajout des méthodes `fit`, `predict`, `evaluate` (`75f1dc6`).

### Refacto
- Modifications mineures (`e4d74b0`).

### Docs
- Changements mineurs (`353b3d8`).

### Merge
- Fusion de `main` dans `automl-feat` (`6b9eaf2`).

## [2025-11-22] - 2025-11-21 - Arab / Maelig
### Feat / WIP
- Ajout du module `clean` (`859820f`).
- Amélioration GD et entraînement (`f64cb47`).
- Nouvelles méthodes de comparaison (`f4b354d`).

### Fix
- Corrections mineures dans la sélection de modèles (`34684f4`).
- Ajout de fichiers ignorés (`4a41b58`).

### Docs
- Mise à jour du README (`9faa199`).

## [2025-11-12] - Maelig
### Feat
- Choix de méthode d'entraînement via `argparse` (`13e50a1`).
- Gestion de la version via `.env` (`6dcc793`, `6d0f3a0`).

### Docs
- Avertissement pour violation d'accès (`be8be80`, `4954aa5`).

## [2025-10-17] - 2025-10-16 - Maelig
### Feat
- Ajout de runners spécifiques (`2ed155a`).
- Support `.env` dataset directory (`2b63055`).
- Auto removal `__pycache__` (`0264997`).

### Fix
- Problèmes de comparaison (`7038333`).
- Global fixes et refactor (`2fdb06c`).
- Correction warnings et deprecated (`8bc764d`).

### Style / WIP
- Code PEP8 et refactoring pour cleanup (`3cfc291`, `fe24b00`).
- Main AutoML model selection (`7da02a4`).

### Docs
- README et changelog updates (`3fe06c8`, `d45cb6e`, `2045c3d`).

## [2025-10-15] - Maelig
### WIP
- Main AutoML model selection (`7da02a4`).

### Fix
- Suppression de la préparation des datas (`f851685`).
- Models fix (`24c30b4`).

### Docs
- Documentation hosting et auto-generation (`a32be46`, `85cf2ac`, `8488745`, `18cd7af`, `41f82cd`, `29fb8ff`, `c5ae064`).

## [2025-10-10] - Maelig
### Docs
- CI/CD pipeline documentation (`7dab6fc`, `e240302`, `cb09445`).

### Fix
- CI/CD pipeline fixes (`f1b1442`, `d5c0918`, `912b58b`).

### Feat
- Pipeline pour exécution automatique des tests (`732a984`).
- CHANGELOG.md création (`b0b8748`).
- Ajout LICENSE (`dc17164`).

### Docs
- Initial doc updates (`95ddb4b`).

## [2025-10-09] - Maelig
### Feat
- `automl` load_data test (`d75c0a3`).
- `model_selector` integration & update (`42ab946`).
- Test duration tracking (`87d29f8`).
- Ridge classifier placeholder (`0f249cf`).
- Models class placeholder (`af1e68a`).
- `model_selector` module (`68670ef`).
- `automl.py` v0.1 et package (`2863f30`, `a7e4fcf`).

### Fix
- Support for sparse datasets (`bfbbe1b`).
- `model_selector` ridge update (`297bd96`).
- Struct. changes enable pip-package (`2a1aa39`).

### Docs
- Various README and documentation updates (`ce3c2b2`, `86a16e3`, `ac2105c`, `13d5b42`, `50f13eb`, `ea336411`, `ea86dc8`).

### Other
- `requirements.txt` creation (`e4b62d4`).
- UT `model_selector` (`ea4d56b`).
- `.gitignore` creation (`10dcb9a`).
- Structure changes (`1a26902`, `a6010cc`, `7d262e8`, `f858bca`).

### Merge
- Plusieurs merges depuis les branches (`4775fe6`, `d1e419a`, `3fb66be`, `f0af73e`, `b319fb4`, `4068cad`).

