import os
import sys
import time
import unittest

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from automl.select import model_selector
from tests.base_test import BaseTest
from tests.util import Utils

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestModelSelector(BaseTest):
    """
    Unit test for the `model_selector` function.
    Loads each dataset, performs model selection, and verifies that the returned
    task type and model class match the expected results.
    """

    @classmethod
    def setUpClass(cls):
        """
        Load all datasets into memory and store them in a dictionary for testing.
        """
        load_dotenv()
        cls.datasets_dir = os.getenv("DATASETS_DIR")
        cls.dataset_names = ["data_A", "data_B", "data_C", "data_D", "data_E", "data_F", "data_G"]
        cls.dataframes = {}

        for ds_name in cls.dataset_names:
            df = cls._load_dataset(cls.datasets_dir, ds_name)
            cls.dataframes[ds_name] = df

    @staticmethod
    def _load_dataset(datasets_dir, dataset_name):
        """
        Load a dataset and its solution file, convert to a DataFrame, and optionally
        downsample rows and columns for faster testing.
        """
        data_path = os.path.join(datasets_dir, dataset_name, f"{dataset_name}.data")
        solution_path = os.path.join(datasets_dir, dataset_name, f"{dataset_name}.solution")

        y = pd.read_csv(solution_path, sep=r'\s+', header=None, engine='python', on_bad_lines='skip').squeeze("columns")
        y.name = "target"

        with open(data_path, 'r') as f:
            lines = f.readlines()

        if any(':' in line for line in lines):
            max_col = 0
            data_rows = []
            for line in lines:
                row = {}
                entries = line.strip().split()
                for entry in entries:
                    if ':' in entry:
                        idx, val = entry.split(':')
                        idx, val = int(idx), float(val)
                        row[idx] = val
                        max_col = max(max_col, idx)
                data_rows.append(row)

            X = pd.DataFrame(0, index=np.arange(len(data_rows)), columns=np.arange(max_col + 1))
            for i, row in enumerate(data_rows):
                for col_idx, val in row.items():
                    X.at[i, col_idx] = val
        else:
            X = pd.read_csv(data_path, sep=r'\s+', header=None, engine='python', on_bad_lines='skip')

        if len(X) > 2000:
            sample_idx = np.random.choice(X.index, size=2000, replace=False)
            X = X.loc[sample_idx]
            y = y.loc[sample_idx].reset_index(drop=True)
            X = X.reset_index(drop=True)

        if X.shape[1] > 500:
            X = X.iloc[:, :500]

        df = pd.concat([X, y], axis=1)
        print(f"{dataset_name}: X shape = {X.shape}, non-zero entries = {(X != 0).sum().sum()}")
        return df

    def test_all_datasets_model_selection(self):
        """
        Test model selection on all datasets.
        Verify that the returned task type and model class match the expected results.
        """
        print("\n\n\n\n\n")
        print("############################################################")
        print("#                                                          #")
        print("#           TEST #4 : Automatic model selection            #")
        print("#                                                          #")
        print("############################################################")
        print("\n\n")
        expected_results = {
            "data_A": "multilabel_classification",
            "data_B": "regression",
            "data_C": "multilabel_classification",
            "data_D": "classification",
            "data_E": "classification",
            "data_F": "multilabel_classification",
            "data_G": "classification",
        }

        model_families = {
            "classification": ("Classifier",),
            "regression": ("Regressor",),
            "multilabel_classification": ("Classifier",),
        }

        for ds_name, df in self.dataframes.items():
            with self.subTest(dataset=ds_name):
                model, task_type = model_selector(df)
                expected_task = expected_results[ds_name]

                self.assertEqual(task_type, expected_task)

                family_keywords = model_families[task_type]
                self.assertTrue(any(k in model.__class__.__name__ for k in family_keywords),
                                f"{ds_name}: Model {model.__class__.__name__} is not compatible with task {task_type}")



if __name__ == "__main__":
    unittest.main()
    Utils.remove_all_pycache()