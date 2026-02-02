import os
import sys
import time
import unittest

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from automl.automl import AutoML
from tests.base_test import BaseTest
from tests.util import Utils

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestAutoMLLoadData(BaseTest):
    """
    Unit test for the `AutoML.load_data` function.
    
    Ensures that datasets are correctly loaded into pandas DataFrames,
    that target columns exist and contain no NaN values, and that
    DataFrames have valid rows and columns.
    """

    @classmethod
    def setUpClass(cls):
        """
        Initializes the datasets directory, dataset names, the AutoML instance,
        and a dictionary to store loaded DataFrames.
        """
        load_dotenv()
        cls.datasets_dir = os.getenv("DATASETS_DIR")
        cls.dataset_names = ["data_A", "data_B", "data_C", "data_D", "data_E", "data_F", "data_G"]
        cls.automl = AutoML()
        cls.dataframes = {}

    def test_load_data_all_datasets(self):
        """
        Loads all datasets and validates the following for each:
        - The result is a pandas DataFrame
        - The DataFrame has at least one row and more than one column
        - Target columns exist
        - Target columns contain no NaN values
        - Prints the DataFrame shape, number of non-zero entries, and loading time
        """
        print("\n\n\n\n\n")
        print("############################################################")
        print("#                                                          #")
        print("#              TEST #2 : Loading all datasets              #")
        print("#                                                          #")
        print("############################################################")
        print("\n\n")
        for ds_name in self.dataset_names:
            with self.subTest(dataset=ds_name):
                data_path = os.path.join(self.datasets_dir, ds_name, f"{ds_name}.data")
                solution_path = os.path.join(self.datasets_dir, ds_name, f"{ds_name}.solution")

                start_time = time.time()

                df = self.automl.load_data(data_path, solution_path, ds_name)
                self.dataframes[ds_name] = df

                elapsed_time = time.time() - start_time

                self.assertIsInstance(df, pd.DataFrame, f"{ds_name} -> The result is not a DataFrame.")
                self.assertGreater(df.shape[0], 0, f"{ds_name} -> DataFrame is empty (0 rows).")
                self.assertGreater(df.shape[1], 1, f"{ds_name} -> DataFrame has no columns.")

                target_cols = [c for c in df.columns if str(c).startswith("target")]
                self.assertTrue(len(target_cols) >= 1, f"{ds_name} -> No target columns found.")

                if len(target_cols) == 1:
                    y = df[target_cols[0]]
                    self.assertFalse(y.isna().any(), f"{ds_name} -> Target column contains NaN values.")
                else:
                    y_df = df[target_cols]
                    self.assertFalse(y_df.isna().any().any(), f"{ds_name} -> Some target columns contain NaN values.")

                non_target_df = df.drop(columns=target_cols)
                non_zero = (non_target_df != 0).sum().sum()
                print(f"{ds_name} loaded successfully: shape = {df.shape}, non-zero entries = {non_zero}, time = {elapsed_time:.2f}s")

if __name__ == "__main__":
    unittest.main()
    Utils.remove_all_pycache()
