from __future__ import annotations

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


_HOG = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)


def hog_descriptor(crop_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    return _HOG.compute(resized).reshape(-1).astype(np.float32)


def train_hog_svm(train_features: np.ndarray, train_labels: np.ndarray, seed: int = 42):
    model = make_pipeline(
        StandardScaler(),
        LinearSVC(class_weight="balanced", random_state=seed, dual="auto", max_iter=5000),
    )
    model.fit(train_features, train_labels)
    return model


def train_linear_classifier(train_features: np.ndarray, train_labels: np.ndarray, seed: int = 42):
    model = LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        random_state=seed,
        solver="liblinear",
    )
    model.fit(train_features, train_labels)
    return model


def model_scores(model, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    return np.asarray(model.decision_function(features), dtype=np.float64)

