import os
import pandas as pd
from automl.automl import AutoML

_model: AutoML | None = None

BASE_TMP_DIR = os.path.join(os.path.dirname(__file__), "tmp_automl")
os.makedirs(BASE_TMP_DIR, exist_ok=True)

TRAIN_PREFIX = os.path.join(BASE_TMP_DIR, "train")
TEST_PREFIX = os.path.join(BASE_TMP_DIR, "test")


def _write_data(prefix: str, X, y=None):
    df = pd.DataFrame.from_records(X)
    df = df.apply(pd.to_numeric, errors="coerce")
    data_path = prefix + ".data"
    df.to_csv(data_path, index=False, header=False, sep=" ")
    if y is not None:
        solution_path = prefix + ".solution"
        pd.Series(y).to_csv(solution_path, index=False, header=False)
    return data_path



def fit_model(X, y, automl_params=None):
    global _model

    _write_data(TRAIN_PREFIX, X, y)

    _model = AutoML()
    _model.fit(TRAIN_PREFIX)


def predict_model(X):
    if _model is None:
        raise RuntimeError("Model not trained. Call /fit first.")

    test_file = _write_data(TEST_PREFIX, X)
    if test_file is None:
        raise RuntimeError("Failed to write test data file")

    preds = _model.predict(test_file)
    return [p.item() if hasattr(p, "item") else p for p in preds]

def eval_model():
    if _model is None:
        raise RuntimeError("Model not trained. Call /fit first.")

    return _model.eval()
