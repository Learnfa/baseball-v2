"""Shared model definitions for the Moneyball project.

These factories are reused across model experiment notebooks so the same model
name always means the same preprocessing and estimator settings.

These are screening/default definitions, not final tuned models.
"""

from __future__ import annotations

import numpy as np

from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


def make_ridge() -> Pipeline:
    """Create the Ridge screening pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=np.logspace(-3, 3, 25))),
        ]
    )


def make_elasticnet() -> Pipeline:
    """Create the ElasticNet screening pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                ElasticNetCV(
                    l1_ratio=[0.1, 0.5, 0.9],
                    cv=3,
                    max_iter=50_000,
                    tol=1e-4,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def make_random_forest() -> Pipeline:
    """Create the Random Forest screening pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    min_samples_leaf=2,
                    max_features=0.8,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def make_hgb() -> Pipeline:
    """Create the HistGradientBoosting screening pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_depth=5,
                    max_iter=200,
                    l2_regularization=1.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def make_lightgbm() -> Pipeline:
    """Create the LightGBM screening pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                LGBMRegressor(
                    objective="regression_l1",
                    n_estimators=300,
                    learning_rate=0.03,
                    num_leaves=15,
                    max_depth=-1,
                    min_child_samples=20,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            ),
        ]
    )
    
    
MODEL_FACTORIES = {
    "ridge": make_ridge,
    "elasticnet": make_elasticnet,
    "random_forest": make_random_forest,
    "hgb": make_hgb,
    "lightgbm": make_lightgbm,
}


def get_model_factory(model_name: str):
    """Return the factory for a named model."""
    try:
        return MODEL_FACTORIES[model_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model {model_name!r}. "
            f"Available models: {sorted(MODEL_FACTORIES)}"
        ) from exc


__all__ = [
    "RANDOM_STATE",
    "MODEL_FACTORIES",
    "get_model_factory",
    "make_ridge",
    "make_elasticnet",
    "make_random_forest",
    "make_hgb",
    "make_lightgbm",
]