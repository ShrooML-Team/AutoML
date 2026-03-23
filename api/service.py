import os
import pandas as pd
from automl.automl import AutoML

_model: AutoML | None = None
_preprocessing_pipeline = None  # ✅ Sauvegarder le pipeline

BASE_TMP_DIR = os.path.join(os.path.dirname(__file__), "tmp_automl")
os.makedirs(BASE_TMP_DIR, exist_ok=True)

TRAIN_PREFIX = os.path.join(BASE_TMP_DIR, "train")
TEST_PREFIX = os.path.join(BASE_TMP_DIR, "test")

COLUMN_ORDER = [
    "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color", "stalk-shape",
    "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
    "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
    "ring-number", "ring-type", "spore-print-color", "population", "habitat"
]


def _write_data(prefix: str, X, y=None):
    """Sauvegarder les données en format .data"""
    df = pd.DataFrame.from_records(X)
    df = df[COLUMN_ORDER]
    df = df.apply(pd.to_numeric, errors="coerce")
    
    data_path = prefix + ".data"
    df.to_csv(data_path, index=False, header=False, sep=" ")
    
    if y is not None:
        solution_path = prefix + ".solution"
        pd.Series(y).to_csv(solution_path, index=False, header=False)
    
    return data_path


def fit_model(X, y, automl_params=None):
    """
    Entraîner le modèle AutoML.
    
    ✅ Important: Sauvegarder aussi les colonnes AVANT preprocessing
    pour que predict() puisse charger les données correctement.
    """
    global _model, _preprocessing_pipeline

    print(f"\n[FIT] === DÉBUT DU FIT ===")
    print(f"[FIT] Samples: {len(X)}, Labels: {len(y)}")
    
    # Écrire les données
    data_path = _write_data(TRAIN_PREFIX, X, y)
    print(f"[FIT] Fichier créé: {data_path}")
    
    # Créer et entraîner AutoML
    _model = AutoML()
    _model.fit(TRAIN_PREFIX)
    
    print(f"[FIT] AutoML.fit() terminé")
    
    # ✅ CRITICAL: Sauvegarder les informations nécessaires pour predict()
    
    # 1. Sauvegarder le nombre de colonnes originales
    _model.original_n_features = len(COLUMN_ORDER)
    print(f"[FIT] Colonnes originales: {_model.original_n_features}")
    
    # 2. Sauvegarder feature_columns et convertir en strings
    if hasattr(_model, 'feature_columns'):
        # Les feature_columns sont peut-être des entiers (0, 1, 2, ...)
        # ou des strings ("0", "1", "2", ...)
        # Il faut les convertir en strings pour le reindex
        original_fc = _model.feature_columns
        _model.feature_columns = [str(c) for c in original_fc]
        
        print(f"[FIT] Feature columns (original): {original_fc}")
        print(f"[FIT] Feature columns (converted to str): {_model.feature_columns}")
        print(f"[FIT] Nombre de feature columns: {len(_model.feature_columns)}")
    
    # 3. Sauvegarder le DataFrame d'entraînement (avant preprocessing)
    # pour avoir accès aux colonnes d'entraînement
    if hasattr(_model, 'df'):
        _model.original_columns = list(_model.df.columns)
        print(f"[FIT] Colonnes du DF d'entraînement: {_model.original_columns}")
    
    print(f"[FIT] === FIN DU FIT ===\n")
    
    return _model


def predict_model(X):
    """Prédire avec le patch pour preprocess"""
    global _model

    if _model is None:
        raise RuntimeError("Model not trained. Call /fit first.")

    test_file = _write_data(TEST_PREFIX, X)

    from automl.clean import preprocess_from_profile
    
    X_test = pd.read_csv(test_file, sep=r"\s+", header=None, engine="python", on_bad_lines="skip")
    X_test.columns = X_test.columns.astype(str)
    
    X_test = preprocess_from_profile(X_test)
    
    X_test = X_test.reindex(columns=_model.feature_columns, fill_value=0)
    
    preds = _model.model.predict(X_test)
    return [p.item() if hasattr(p, "item") else p for p in preds]

def eval_model():
    """Évaluer le modèle sur les données d'entraînement."""
    if _model is None:
        raise RuntimeError("Model not trained. Call /fit first.")

    return _model.eval()