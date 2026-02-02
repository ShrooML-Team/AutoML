"""
Module `automl`
----------------------

Defines the `AutoML` class, orchestrating automatic data loading,
model selection, training, and evaluation. Works generically for
classification, regression, and multilabel classification tasks.
"""

import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from automl.display import print_metrics_table
from automl.select import model_selector
from automl.train import train_and_evaluate_model
from automl.clean import preprocess_from_profile
from automl.compare import evaluate_model

class AutoML:
    """
    AutoML orchestrator class.
    Handles loading, model selection, training, evaluation, and baseline comparison.
    """

    def __init__(self):
        self.model = None
        self.task_type = None
        self.metrics = {}
        self.train_method = 'h'
        self.df = None

    # ------------------------------------------------------------------ #
    # |                  Load dataset (regular or sparse)               | #
    # ------------------------------------------------------------------ #
    def load_data(self, data_path: str, solution_path: str, dataset_name: str):
        """
        Load dataset from file paths. Supports sparse and dense formats.
        """
        y_df = pd.read_csv(solution_path, sep=r"\s+", header=None, engine="python", on_bad_lines="skip")

        if y_df.shape[1] == 1:
            y = y_df.iloc[:, 0].rename("target")
        else:
            y = y_df.copy()
            y.columns = [f"target_{i}" for i in range(y.shape[1])]

        with open(data_path, "r") as f:
            lines = f.readlines()

        if any(":" in line for line in lines):
            max_col = 0
            data_rows = []
            for line in lines:
                row = {}
                for entry in line.strip().split():
                    if ":" in entry:
                        idx, val = entry.split(":")
                        row[int(idx)] = float(val)
                        max_col = max(max_col, int(idx))
                data_rows.append(row)
            X = pd.DataFrame(0, index=np.arange(len(data_rows)), columns=np.arange(max_col + 1))
            for i, row in enumerate(data_rows):
                for col_idx, val in row.items():
                    X.at[i, col_idx] = val
        else:
            X = pd.read_csv(data_path, sep=r"\s+", header=None, engine="python", on_bad_lines="skip")

        X.columns = X.columns.astype(str)
        y = y.reset_index(drop=True)
        df = pd.concat([X, y], axis=1)

        print(f"{dataset_name}: X shape = {X.shape}, non-zero entries = {(X != 0).sum().sum()}")
        return df

    # ------------------------------------------------------------------ #
    # |                     Main AutoML Pipeline                        | #
    # ------------------------------------------------------------------ #
    def run(self, data_path: str, solution_path: str, dataset_name: str = "dataset", train_method: str = "h"):
        """
        Execute the AutoML pipeline:
        1. Load data
        2. Select model
        3. Train + Evaluate (delegated to train.py with tqdm)
        """
        print(f"\n=== Running AutoML on {dataset_name} ===")
        df = self.load_data(data_path, solution_path, dataset_name)

        print("[INFO] Nettoyage du dataset avec ydata_profiling ...")
        df = preprocess_from_profile(df)
        print("[INFO] Nettoyage terminé. New shape =", df.shape)

        self.model, self.task_type = model_selector(df)
        print(f"[INFO] Selected model: {self.model.__class__.__name__} for task: {self.task_type}")

        self.model, self.metrics = train_and_evaluate_model(
            self.model,
            df,
            self.task_type,
            log_file=None,
            train_method=train_method
        )

        print_metrics_table(self.metrics)
        print("[INFO] Metrics table displayed.")

        return self.model, self.task_type, self.metrics

    def fit(self, path: str, train_method: str = 'h'):
        """
        Train an AutoML model on a dataset.

        Loads data, preprocesses it, selects the best model,
        trains it, and stores training predictions for evaluation.
        """
        data_path = f"{path}.data"
        solution_path = f"{path}.solution"

        # Load + preprocess train data
        df = self.load_data(data_path, solution_path, "dataset")
        df = preprocess_from_profile(df)

        # Model selection
        self.model, self.task_type = model_selector(df)

        # Train model
        self.model, self.metrics = train_and_evaluate_model(
            self.model,
            df,
            self.task_type,
            log_file=None,
            train_method=train_method
        )

        # Store training data
        self.df = df
        self.feature_columns = [
            c for c in df.columns if not c.startswith("target")
        ]

        self.X_train = df[self.feature_columns]
        self.y_train = df[[c for c in df.columns if c.startswith("target")]]

        self.y_pred_train = self.model.predict(self.X_train)
        self.train_method = train_method

        return self.model

    
    def eval(self):
        """
        Evaluate the trained model on the training data.

        Computes task-specific metrics using stored predictions.
        """
        if self.model is None:
            raise ValueError("Model is not trained yet. Call fit() before eval().")
    
        y_true = self.y_train
        y_pred = self.y_pred_train

        metrics = evaluate_model(y_true, y_pred, task_type=self.task_type)
        return metrics


    def predict(self, test_path: str):
        """
        Predict outputs for a test dataset.

        Loads and preprocesses test data, aligns features with
        the training set, and returns model predictions.
        """

        if self.model is None:
            raise ValueError("Model is not trained yet. Call fit() before predict().")

        # Load test data
        with open(test_path, "r") as f:
            lines = f.readlines()

        if any(":" in line for line in lines):
            max_col = 0
            data_rows = []
            for line in lines:
                row = {}
                for entry in line.strip().split():
                    if ":" in entry:
                        idx, val = entry.split(":")
                        row[int(idx)] = float(val)
                        max_col = max(max_col, int(idx))
                data_rows.append(row)

            X = pd.DataFrame(
                0,
                index=np.arange(len(data_rows)),
                columns=np.arange(max_col + 1)
            )

            for i, row in enumerate(data_rows):
                for col_idx, val in row.items():
                    X.at[i, col_idx] = val
        else:
            X = pd.read_csv(
                test_path,
                sep=r"\s+",
                header=None,
                engine="python",
                on_bad_lines="skip"
            )

        X.columns = X.columns.astype(str)

        # Align columns with training data
        X = X.reindex(columns=self.feature_columns, fill_value=0)

        # Apply same preprocessing as training
        X = preprocess_from_profile(X)

        preds = self.model.predict(X)
        return preds



if __name__ == "__main__":
    import argparse
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run AutoML on a dataset.")
    parser.add_argument("--data", required=True, help="Path to the dataset (.data)")
    parser.add_argument("--solution", required=True, help="Path to the solution file (.solution)")
    parser.add_argument("--name", default="dataset", help="Dataset name for logs")
    parser.add_argument("--train-method", default="h", help="Training method to use (d: default(heuristic), g: gradient descent, l: library train)")
    args = parser.parse_args()

    version = os.getenv("VERSION", "unknown")
    print(f"[DEBUG] Starting AutoML v{version} CLI entrypoint")
    automl = AutoML()
    model, task_type, prepared = automl.run(args.data, args.solution, dataset_name=args.name, train_method=args.train_method)
    print("[DEBUG] AutoML finished.")
