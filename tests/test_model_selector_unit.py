import numpy as np
import pandas as pd
import pytest

from automl import select


class FakePicklingError(Exception):
    pass


def test_model_selector_classification_selects_best_model(monkeypatch):
    model_a = object()
    model_b = object()

    monkeypatch.setattr(select, "CLASSIFICATION_MODELS", {"a": model_a, "b": model_b})

    def fake_cross_val_score(model, X_sample, y_sample, cv, scoring, n_jobs):
        assert scoring == "f1_micro"
        return np.array([0.2]) if model is model_a else np.array([0.9])

    monkeypatch.setattr(select, "cross_val_score", fake_cross_val_score)

    df = pd.DataFrame({
        "f1": [0, 1, 0, 1],
        "target": [0, 1, 0, 1],
    })

    model, task_type = select.model_selector(df, fast_mode=False)

    assert task_type == "classification"
    assert model is model_b


def test_model_selector_regression_selects_best_model(monkeypatch):
    model_a = object()
    model_b = object()

    monkeypatch.setattr(select, "REGRESSION_MODELS", {"a": model_a, "b": model_b})

    def fake_cross_val_score(model, X_sample, y_sample, cv, scoring, n_jobs):
        assert scoring == "r2"
        return np.array([0.1]) if model is model_a else np.array([0.8])

    monkeypatch.setattr(select, "cross_val_score", fake_cross_val_score)

    df = pd.DataFrame({
        "f1": list(range(30)),
        "target": np.linspace(0.1, 10.0, 30),
    })

    model, task_type = select.model_selector(df, fast_mode=False)

    assert task_type == "regression"
    assert model is model_b


def test_model_selector_retries_when_pickling_error(monkeypatch):
    model_a = object()
    model_b = object()

    monkeypatch.setattr(select, "CLASSIFICATION_MODELS", {"a": model_a, "b": model_b})

    def fake_cross_val_score(model, X_sample, y_sample, cv, scoring, n_jobs):
        if model is model_a and n_jobs == -1:
            raise FakePicklingError("parallel backend issue")
        if model is model_a and n_jobs == 1:
            return np.array([0.7])
        return np.array([0.3])

    monkeypatch.setattr(select, "cross_val_score", fake_cross_val_score)

    df = pd.DataFrame({
        "f1": [0, 1, 0, 1],
        "target": [0, 1, 0, 1],
    })

    model, task_type = select.model_selector(df, fast_mode=False)

    assert task_type == "classification"
    assert model is model_a


def test_model_selector_raises_when_all_models_fail(monkeypatch):
    monkeypatch.setattr(select, "CLASSIFICATION_MODELS", {"only": object()})

    def always_fail(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(select, "cross_val_score", always_fail)

    df = pd.DataFrame({
        "f1": [0, 1, 0, 1],
        "target": [0, 1, 0, 1],
    })

    with pytest.raises(RuntimeError, match="No suitable model"):
        select.model_selector(df, fast_mode=False)
