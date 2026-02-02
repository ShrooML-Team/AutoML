"""
Module `train`
----------------------

Handles dataset splitting, model fitting, and evaluation.
Supports multi-output tasks and advanced hyperparameter optimization
with progress bars via tqdm.
"""

import warnings
from functools import partial

import optuna
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import ParameterGrid, train_test_split, cross_val_score
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.linear_model import SGDClassifier, SGDRegressor
from tqdm import trange, tqdm

from automl.compare import evaluate_model




def train_and_evaluate_model(model, df, task_type, log_file=None, train_method="h"):
    """
    Handles train/test split, optional hyperparameter tuning,
    model training, and evaluation.
    Adds tqdm progress bars for long operations.
    """
    target_cols = [c for c in df.columns if "target" in c]
    X = df.drop(columns=target_cols)
    y = df[target_cols]

    stratify = y if task_type in ["classification", "multilabel_classification"] and y.shape[1] == 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    tqdm_print = partial(tqdm.write, file=log_file or None)
    tqdm_print(f"[INFO] Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    if y_train.shape[1] > 1:
        tqdm_print(f"[WARN] y has {y_train.shape[1]} columns -> enabling multi-output mode.")
        if task_type == "regression":
            model = MultiOutputRegressor(model)
        else:
            model = MultiOutputClassifier(model)
        task_type = "multilabel_classification"
    else:
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()

    train_methods = {
        "h": _optimize_model,
        "g": _optimize_model_gradient_descent,
        "l": _optimize_model_library,
    }

    if train_method in train_methods:
        model = train_methods[train_method](model, task_type, X_train, y_train, tqdm_print)
    else:
        raise ValueError(f"Unknown training method: {train_method}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tqdm_print("[INFO] Training model ...")
        model.fit(X_train, y_train)

    tqdm_print("[INFO] Model training completed.")
    y_pred = model.predict(X_test)

    metrics = evaluate_model(y_test, y_pred, task_type)
    return model, metrics



def _optimize_model_gradient_descent(model, task_type, X_train, y_train, tqdm_print,
                                     lr=0.01, n_epochs=50, batch_size=None):
    """
    Perform gradient descent optimization for models that support it (SGDClassifier / SGDRegressor).

    For other models, warns and returns the original model without modification.

    Parameters
    ----------
    model : sklearn-like estimator
        Model to optimize.
    task_type : str
        'classification' or 'regression'.
    X_train : pd.DataFrame or np.ndarray
        Training features.
    y_train : pd.Series, pd.DataFrame or np.ndarray
        Training labels.
    tqdm_print : callable
        Function for logging (partial(tqdm.write)).
    lr : float
        Learning rate for gradient descent.
    n_epochs : int
        Number of epochs to run.
    batch_size : int or None
        Mini-batch size. If None, full-batch is used.

    Returns
    -------
    model : sklearn-like estimator
        Trained model.
    """
    model_name = model.__class__.__name__

    # Only SGD models are compatible
    if not isinstance(model, (SGDClassifier, SGDRegressor)):
        tqdm_print(f"[WARN] Model '{model_name}' not compatible with gradient descent. Skipping 'g'.")
        return model

    # Prepare classes for classification
    classes = np.unique(y_train) if task_type in ["classification", "multilabel_classification"] and y_train.ndim == 1 else None

    n_samples = X_train.shape[0]
    if batch_size is None:
        batch_size = n_samples  # full-batch

    tqdm_print(f"[INFO] Starting gradient descent ({model_name}) for {n_epochs} epochs ...")

    for epoch in trange(n_epochs, desc=f"Gradient Descent {model_name}"):
        # Shuffle the data
        idx = np.random.permutation(n_samples)
        X_shuffled = X_train.iloc[idx] if isinstance(X_train, (pd.DataFrame, pd.Series)) else X_train[idx]
        y_shuffled = y_train.iloc[idx] if isinstance(y_train, (pd.DataFrame, pd.Series)) else y_train[idx]

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # Partial fit
            if task_type in ["classification", "multilabel_classification"]:
                model.partial_fit(X_batch, y_batch, classes=classes)
            else:
                model.partial_fit(X_batch, y_batch)


    tqdm_print(f"[INFO] Gradient descent finished for {model_name}")
    return model


def _optimize_model_library(model, task_type, X_train, y_train, tqdm_print):
    """
    Optionally tune key hyperparameters depending on model type.
    Uses optuna for hyperparameter optimization.
    """

    model_name = model.__class__.__name__.lower()
    tqdm_print(f"[INFO] Démarrage de l'optimisation Optuna pour {model_name} ...")

    def objective(trial):
        param_grid = {}

        if "randomforest" in model_name:
            param_grid = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=100),
                "max_depth": trial.suggest_int("max_depth", 3, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            }

        elif "xgb" in model_name:
            param_grid = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 5),
            }

        elif "lgbm" in model_name:
            param_grid = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 255),
                "max_depth": trial.suggest_int("max_depth", -1, 20),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            }

        elif "logistic" in model_name:
            param_grid = {
                "C": trial.suggest_loguniform("C", 1e-3, 10),
                "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
                "max_iter": trial.suggest_int("max_iter", 100, 1000, step=100),
            }

        else:
            tqdm_print(f"[INFO] Aucun espace de recherche défini pour {model_name}, saut de l'optimisation.")
            return 0.0

        model.set_params(**param_grid)

        scoring = 'f1_weighted' if task_type in ["classification", "multilabel_classification"] else 'r2'

        try:
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring=scoring, n_jobs=-1)
            mean_score = np.mean(scores)
        except Exception as e:
            tqdm_print(f"[WARN] Erreur pendant l'évaluation : {e}")
            return -np.inf

        tqdm_print(f"[TRIAL {trial.number}] Score={mean_score:.4f} | Params={param_grid}")
        return mean_score

    study = optuna.create_study(direction="maximize")
    tqdm_print("[INFO] Lancement de l'étude Optuna (20 essais par défaut) ...")
    study.optimize(objective, n_trials=20, show_progress_bar=False)

    tqdm_print(f"[INFO] Meilleurs hyperparamètres trouvés : {study.best_params}")
    tqdm_print(f"[INFO] Meilleur score obtenu : {study.best_value:.4f}")

    model.set_params(**study.best_params)
    return model


def _optimize_model(model, X_train, y_train, name, task_type="classification", logger=None):
    """Optimize hyperparameters intelligently with proper scoring and fallback."""
    try:
        # Skip if multi-output
        if len(y_train.shape) > 1 and y_train.shape[1] > 1:
            if logger: logger.info("[INFO] Skipping hyperparameter tuning (multi-output detected).")
            return model
        
        # Choose appropriate scorer
        if task_type == "classification":
            scorer = make_scorer(f1_macro)
        elif task_type == "regression":
            scorer = make_scorer(r2_score)
        else:
            scorer = None
        
        # Define parameter grids per model
        param_grids = {
            "lgbmclassifier": {
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7, -1],
                "n_estimators": [100, 200, 300]
            },
            "randomforestclassifier": {
                "n_estimators": [100, 200, 500],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5, 10]
            },
            "xgbclassifier": {
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 6],
                "n_estimators": [100, 300]
            },
            "ridge": {
                "alpha": [0.1, 1.0, 10.0]
            },
            "lgbmregressor": {
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 7],
                "n_estimators": [100, 200]
            },
        }
        
        # Retrieve params for this model
        grid_params = param_grids.get(name.lower())
        if not grid_params:
            if logger: logger.info(f"[INFO] No tuning grid defined for {name}, using default params.")
            return model
        
        if logger: logger.info(f"[INFO] Running hyperparameter tuning for {name.lower()} ...")
        
        # Try tuning with multiprocessing
        try:
            grid = GridSearchCV(model, grid_params, scoring=scorer, cv=3, n_jobs=-1, verbose=0)
            grid.fit(X_train, y_train)
        except PicklingError:
            if logger: logger.warning(f"[WARN] {name} failed with PicklingError — retrying with n_jobs=1.")
            grid = GridSearchCV(model, grid_params, scoring=scorer, cv=3, n_jobs=1, verbose=0)
            grid.fit(X_train, y_train)
        
        best_model = grid.best_estimator_
        best_score = getattr(grid, "best_score_", None)
        if logger:
            if best_score is not None:
                logger.info(f"[INFO] Best params found: {grid.best_params_} with mean CV score {best_score:.4f}")
            else:
                logger.info(f"[INFO] Best params found: {grid.best_params_} (no valid CV score)")
        return best_model

    except Exception as e:
        if logger: logger.error(f"[ERROR] Tuning failed for {name}: {type(e).__name__} - {e}")
        return model
