"""
Module `select`
----------------------

Automatically selects the most suitable ML model for a dataset.
Supports classification, regression, and multilabel tasks.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.multioutput import MultiOutputClassifier

from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor

CLASSIFICATION_MODELS = {
    "knn": KNeighborsClassifier(),
    "logistic_regression": LogisticRegression(max_iter=500, n_jobs=-1),
    "random_forest": RandomForestClassifier(n_estimators=100, n_jobs=-1),
    "ada_boost": AdaBoostClassifier(n_estimators=50),
    "light_gbm": LGBMClassifier(n_estimators=100, verbose=-1),
    "xg_boost": XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', verbosity=0),
    "naive_bayes": GaussianNB(),
    "ridge_classifier": RidgeClassifier()
}

REGRESSION_MODELS = {
    "linear": LinearRegression(),
    "ridge": Ridge(),
    "random_forest": RandomForestRegressor(n_estimators=100, n_jobs=-1),
    "ada_boost": AdaBoostRegressor(n_estimators=50),
    "light_gbm": LGBMRegressor(n_estimators=100, verbose=-1),
    "xg_boost": XGBRegressor(n_estimators=100, verbosity=0)
}

def model_selector(dataset: pd.DataFrame, target_column: str = None, fast_mode: bool = True):
    """
    Method for automatic model selection based on dataset characteristics.
    Datasets are splited into subsamples for speed if fast_mode is enabled.
    """
    if target_column is None:
        target_column = dataset.columns[-1]

    X = dataset.drop(columns=[target_column])
    y = dataset[target_column]

    # Detect task type
    if isinstance(y, pd.DataFrame) and y.shape[1] > 1:
        task_type = 'multilabel_classification'
        from sklearn.tree import DecisionTreeClassifier
        base_model = DecisionTreeClassifier(random_state=42)
        return MultiOutputClassifier(base_model), task_type

    if y.dtype == 'object' or y.dtype.name == 'category' or len(y.unique()) <= 20:
        task_type = 'classification'
        models_dict = CLASSIFICATION_MODELS
        scoring = 'accuracy'
    else:
        task_type = 'regression'
        models_dict = REGRESSION_MODELS
        scoring = 'r2'

    # Subsample for speed
    if fast_mode and len(X) > 5000:
        sample_size = 5000
        X_sample = X.sample(sample_size, random_state=42)
        y_sample = y.loc[X_sample.index]
    else:
        X_sample, y_sample = X, y

    best_score = -np.inf
    best_model = None

    for name, model in models_dict.items():
        print(f"Testing model: {name}")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score = np.mean(cross_val_score(model, X_sample, y_sample, cv=3, scoring=scoring, n_jobs=-1))

        except Exception as e:
            if "PicklingError" in str(type(e).__name__):
                print(f"[WARN] {name} failed with PicklingError — retrying with n_jobs=1.")
                try:
                    score = np.mean(cross_val_score(model, X_sample, y_sample, cv=3, scoring=scoring, n_jobs=1))
                except Exception as e2:
                    print(f"[ERROR] Model {name} failed again: {type(e2).__name__} - {e2}")
                    continue
            else:
                print(f"[ERROR] Model {name} failed: {type(e).__name__} - {e}")
                continue

        if score > best_score:
            best_score = score
            best_model = model

    if best_model is None:
        raise RuntimeError("No suitable model could be selected from candidates.")

    return best_model, task_type
