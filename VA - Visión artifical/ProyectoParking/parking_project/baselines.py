from __future__ import annotations

import cv2
import numpy as np


def otsu_threshold_manual(gray: np.ndarray) -> int:
    """Compute Otsu's threshold from a uint8 grayscale image and its histogram."""
    values = np.asarray(gray, dtype=np.uint8)
    histogram = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    total = float(values.size)
    if total == 0:
        return 0
    intensity_sum = float(np.dot(np.arange(256), histogram))
    background_weight = 0.0
    background_sum = 0.0
    best_variance = -1.0
    best_threshold = 0
    for threshold in range(256):
        background_weight += histogram[threshold]
        if background_weight == 0:
            continue
        foreground_weight = total - background_weight
        if foreground_weight == 0:
            break
        background_sum += threshold * histogram[threshold]
        background_mean = background_sum / background_weight
        foreground_mean = (intensity_sum - background_sum) / foreground_weight
        variance = background_weight * foreground_weight * (background_mean - foreground_mean) ** 2
        if variance > best_variance:
            best_variance = variance
            best_threshold = threshold
    return int(best_threshold)


def foreground_difference_score(current_bgr: np.ndarray, empty_reference_bgr: np.ndarray) -> float:
    """Estimate changed-pixel area after Otsu thresholding and morphology."""
    size = (64, 32)
    current = cv2.resize(current_bgr, size, interpolation=cv2.INTER_AREA)
    reference = cv2.resize(empty_reference_bgr, size, interpolation=cv2.INTER_AREA)
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.GaussianBlur(current_gray, (3, 3), 0)
    reference_gray = cv2.GaussianBlur(reference_gray, (3, 3), 0)
    difference = cv2.absdiff(current_gray, reference_gray)
    threshold = otsu_threshold_manual(difference)
    _, mask = cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    minimum_area = max(4, int(mask.size * 0.005))
    foreground_area = sum(
        int(stats[index, cv2.CC_STAT_AREA])
        for index in range(1, count)
        if int(stats[index, cv2.CC_STAT_AREA]) >= minimum_area
    )
    return float(foreground_area / mask.size)


def best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, dict[str, float]]:
    """Choose an occupied-score threshold on validation data only."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    valid = np.isfinite(scores)
    scores, labels = scores[valid], labels[valid]
    if scores.size == 0:
        raise ValueError("No hay puntuaciones válidas para ajustar el umbral.")
    candidates = np.unique(np.concatenate(([0.0], scores, [1.0])))
    best: tuple[float, float, float, dict[str, float]] | None = None
    for threshold in candidates:
        predictions = scores >= threshold
        tp = int(np.sum(predictions & (labels == 1)))
        fp = int(np.sum(predictions & (labels == 0)))
        fn = int(np.sum(~predictions & (labels == 1)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics = {"precision": precision, "recall": recall, "f1": f1}
        candidate = (f1, precision, float(threshold), metrics)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    assert best is not None
    return best[2], best[3]


def binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=bool)
    predictions = np.asarray(predictions, dtype=bool)
    tp = int(np.sum(labels & predictions))
    tn = int(np.sum(~labels & ~predictions))
    fp = int(np.sum(~labels & predictions))
    fn = int(np.sum(labels & ~predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "support": int(labels.size),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }

