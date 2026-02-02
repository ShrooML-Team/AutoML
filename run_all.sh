#!/bin/bash
# run_all.sh
# Run AutoML on all datasets except 'f'

# Chemin vers les datasets
DATASETS_DIR="/Users/pyxis/Desktop/datasets"

# Liste des datasets à exclure
EXCLUDE=("f")

# Boucle sur tous les dossiers de datasets
for dataset_path in "$DATASETS_DIR"/*; do
    dataset_name=$(basename "$dataset_path")
    
    # Ignorer les datasets exclus
    if [[ " ${EXCLUDE[@]} " =~ " $dataset_name " ]]; then
        echo "[INFO] Skipping dataset $dataset_name"
        continue
    fi

    data_file="$dataset_path/$dataset_name.data"
    solution_file="$dataset_path/$dataset_name.solution"

    # Vérifier que les fichiers existent
    if [[ ! -f "$data_file" || ! -f "$solution_file" ]]; then
        echo "[WARN] Missing .data or .solution for $dataset_name, skipping."
        continue
    fi

    echo "[INFO] Running AutoML on dataset $dataset_name..."
    python -m automl.automl \
        --data "$data_file" \
        --solution "$solution_file" \
        --name "$dataset_name" \
        --train-method h # replace with choosen train-method
done
