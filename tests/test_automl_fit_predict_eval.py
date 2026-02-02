import os
import unittest
from dotenv import load_dotenv
import numpy as np

from automl.automl import AutoML


class TestAutoMLFitPredictEval(unittest.TestCase):
    """
    Integration tests for AutoML.fit, AutoML.predict and AutoML.eval
    using dataset A.
    """

    @classmethod
    def setUpClass(cls):
        """
        Load environment variables and prepare dataset paths.
        """

        print("\n\n\n\n\n")
        print("############################################################")
        print("#                                                          #")
        print("#         TEST #1 : fit, predict, eval methods test        #")
        print("#                                                          #")
        print("############################################################")
        print("\n\n")
        load_dotenv()

        datasets_dir = os.getenv("DATASETS_DIR")
        if datasets_dir is None:
            raise EnvironmentError("DATASETS_DIR is not defined in .env")

        cls.dataset_name = "data_A"
        cls.dataset_dir = os.path.join(datasets_dir, cls.dataset_name)

        cls.data_path = os.path.join(cls.dataset_dir, f"{cls.dataset_name}.data")
        cls.solution_path = os.path.join(cls.dataset_dir, f"{cls.dataset_name}.solution")

        if not os.path.exists(cls.data_path):
            raise FileNotFoundError(f"Dataset file not found: {cls.data_path}")
        if not os.path.exists(cls.solution_path):
            raise FileNotFoundError(f"Solution file not found: {cls.solution_path}")

    def setUp(self):
        """
        Create a fresh AutoML instance for each test.
        """
        self.automl = AutoML()

    def test_fit(self):
        """
        Test that fit() trains a model and stores training attributes.
        """
        model = self.automl.fit(
            path=os.path.join(self.dataset_dir, self.dataset_name),
            train_method="h"
        )

        self.assertIsNotNone(model, "Model should not be None after fit()")
        self.assertIsNotNone(self.automl.task_type, "task_type should be set after fit()")
        self.assertIsNotNone(self.automl.df, "Training dataframe should be stored")
        self.assertTrue(
            hasattr(self.automl, "X_train"),
            "X_train should be created during fit()"
        )
        self.assertTrue(
            hasattr(self.automl, "y_train"),
            "y_train should be created during fit()"
        )
        self.assertTrue(
            hasattr(self.automl, "y_pred_train"),
            "y_pred_train should be created during fit()"
        )

    def test_eval(self):
        """
        Test that eval() returns a non-empty metrics dictionary.
        """
        self.automl.fit(
            path=os.path.join(self.dataset_dir, self.dataset_name),
            train_method="h"
        )

        metrics = self.automl.eval()

        self.assertIsInstance(metrics, dict, "eval() should return a dict")
        self.assertGreater(
            len(metrics),
            0,
            "Metrics dictionary should not be empty"
        )

    def test_predict(self):
        """
        Test that predict() returns predictions of correct length.
        """
        self.automl.fit(
            path=os.path.join(self.dataset_dir, self.dataset_name),
            train_method="h"
        )

        preds = self.automl.predict(self.data_path)

        self.assertIsNotNone(preds, "Predictions should not be None")
        self.assertTrue(
            isinstance(preds, (list, np.ndarray)),
            "Predictions should be a list or numpy array"
        )
        self.assertEqual(
            len(preds),
            len(self.automl.X_train),
            "Number of predictions should match number of samples"
        )


if __name__ == "__main__":
    unittest.main()
