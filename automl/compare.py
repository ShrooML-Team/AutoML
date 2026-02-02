"""
Module `compare`
----------------------

Handles baseline generation and performance evaluation
for different machine learning tasks (classification,
regression, multilabel classification).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, r2_score, mean_absolute_error
)


def baseline_predictions(y_true, task_type):
    """
    Generate baseline predictions (random and zero) for comparison.
    Works for multi-output regression as well.
    """
    n_samples = len(y_true)

    if task_type == "classification":
        classes = np.unique(y_true.values) if isinstance(y_true, pd.DataFrame) else np.unique(y_true)
        random_pred = (
            pd.DataFrame(np.random.choice(classes, size=y_true.shape), columns=y_true.columns)
            if isinstance(y_true, pd.DataFrame)
            else np.random.choice(classes, size=n_samples)
        )
        zero_pred = (
            pd.DataFrame(np.full(y_true.shape, classes[0]), columns=y_true.columns)
            if isinstance(y_true, pd.DataFrame)
            else np.full(n_samples, classes[0])
        )

    elif task_type == "regression":
        if isinstance(y_true, pd.DataFrame):
            random_pred = pd.DataFrame(
                {col: np.random.choice(y_true[col].values, size=n_samples, replace=True)
                 for col in y_true.columns},
                columns=y_true.columns
            )
            zero_pred = pd.DataFrame(
                {col: np.full(n_samples, y_true[col].mean()) for col in y_true.columns},
                columns=y_true.columns
            )
        else:
            random_pred = np.random.choice(y_true.values if isinstance(y_true, pd.Series) else y_true,
                                           size=n_samples, replace=True)
            zero_pred = np.full(n_samples, y_true.mean() if isinstance(y_true, (pd.Series, np.ndarray)) else 0)

    elif task_type == "multilabel_classification":
        random_pred = pd.DataFrame(np.random.randint(0, 2, size=y_true.shape), columns=y_true.columns)
        zero_pred = pd.DataFrame(np.zeros(y_true.shape, dtype=int), columns=y_true.columns)

    else:
        raise ValueError(f"Unknown task type: {task_type}")

    return random_pred, zero_pred


def evaluate_model(y_true, y_pred, task_type):
    """
    Evaluate model performance and compare against simple baselines.
    Returns a dictionary of metrics with macro/micro/weighted averages where relevant.
    """
    metrics = {}

    # CLASSIFICATION
    if task_type in ["classification", "multilabel_classification"]:
        metrics.update({
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        })

    # REGRESSION
    elif task_type == "regression":
        metrics.update({
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
        })

    # BASELINES 
    random_pred, zero_pred = baseline_predictions(y_true, task_type)

    if task_type in ["classification", "multilabel_classification"]:
        metrics.update({
            "random_accuracy": accuracy_score(y_true, random_pred),
            "zero_accuracy": accuracy_score(y_true, zero_pred),
            "random_f1_macro": f1_score(y_true, random_pred, average="macro", zero_division=0),
            "zero_f1_macro": f1_score(y_true, zero_pred, average="macro", zero_division=0),
        })

    elif task_type == "regression":
        random_r2 = max(r2_score(y_true, random_pred), 0.0)
        zero_r2 = max(r2_score(y_true, zero_pred), 0.0)
        metrics.update({
            "random_mse": mean_squared_error(y_true, random_pred),
            "zero_mse": mean_squared_error(y_true, zero_pred),
            "random_r2": random_r2,
            "zero_r2": zero_r2,
        })

    return metrics