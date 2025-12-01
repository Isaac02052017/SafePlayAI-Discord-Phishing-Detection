"""
model_train.py

Utility functions to build and evaluate models for phishing detection.
Used by experiment_ab_test.py and can also be imported into notebooks.
"""

from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


def train_test_split_basic(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    feature_cols_numeric = [
        "msg_timestamp",
        "usr_joined_at",
        "time_since_join",
        "message_length",
        "word_count",
        "has_link",
        "has_mention",
        "num_roles",
    ]

    X = df[feature_cols_numeric]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    return X_train, X_test, y_train, y_test


def build_model_A() -> Pipeline:
    """
    Model A: Numeric-only Logistic Regression with StandardScaler.
    """
    numeric_features = [
        "msg_timestamp",
        "usr_joined_at",
        "time_since_join",
        "message_length",
        "word_count",
        "has_link",
        "has_mention",
        "num_roles",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
        ]
    )

    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )

    return model


def build_model_B() -> Pipeline:
    """
    Model B: Numeric + TF-IDF text features.
    """
    numeric_features = [
        "msg_timestamp",
        "usr_joined_at",
        "time_since_join",
        "message_length",
        "word_count",
        "has_link",
        "has_mention",
        "num_roles",
    ]
    text_feature = "msg_content"

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("text", TfidfVectorizer(min_df=5, ngram_range=(1, 2)), text_feature),
        ]
    )

    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )

    return model


def evaluate_model(model: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame,
                   y_train: pd.Series, y_test: pd.Series) -> Dict[str, float]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    return metrics


def cross_val_f1(model: Pipeline, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> np.ndarray:
    scores = cross_val_score(model, X, y, scoring="f1", cv=cv)
    return scores
