from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import FunctionTransformer

import numpy as np

from xgboost import XGBClassifier


def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:
    """
    Build preprocessing pipeline for numerical and categorical features.
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor

def build_preprocessor_np1log(
    pos_num_cols: list[str],
    neg_num_cols: list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:

    log1p_transformer = Pipeline(steps=[
        ('log1p', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
        ('scaler', StandardScaler())
    ])

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('log1p_num', log1p_transformer, pos_num_cols),
            ('num', numeric_pipeline, neg_num_cols),
            ('cat', categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor


def build_logistic_regression_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> Pipeline:
    """
    Build a baseline logistic regression pipeline.
    """
    preprocessor = build_preprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    return pipeline

def build_logistic_regression_pipeline_np1log(
    pos_num_cols: list[str],
    neg_num_cols: list[str],
    categorical_cols: list[str],
) -> Pipeline:
    preprocessor = build_preprocessor_np1log(
        pos_num_cols = pos_num_cols,
        neg_num_cols = neg_num_cols,
        categorical_cols = categorical_cols,
    )

    model = LogisticRegression(
        max_iter = 1000,
        class_weight="balanced",
        random_state= 42
    )

    pipeline = Pipeline(steps =[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    return pipeline


def build_random_forest_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> Pipeline:
    """
    Build a random forest classifier pipeline.
    """
    preprocessor = build_preprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    return pipeline

def build_xgboost_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> Pipeline:
    """
    Build an XGBoost classifier pipeline.
    """
    preprocessor = build_preprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    return pipeline