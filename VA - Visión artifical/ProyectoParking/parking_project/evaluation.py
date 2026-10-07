from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .types import Detection


def intersection_over_union(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def read_object_annotations(boxes_csv: str | Path, labeled_images_file: str | Path):
    boxes_by_image: dict[str, list[tuple[str, tuple[float, float, float, float]]]] = defaultdict(list)
    boxes_path = Path(boxes_csv)
    with boxes_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        expected = {"image_path", "class_name", "xmin", "ymin", "xmax", "ymax"}
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError(f"La anotación requiere columnas: {sorted(expected)}")
        for row in reader:
            if not any((value or "").strip() for value in row.values()):
                continue
            class_name = (row["class_name"] or "").strip().lower()
            if class_name not in {"person", "car", "motorcycle"}:
                raise ValueError(f"Clase inválida {class_name!r}; usa person, car o motorcycle.")
            try:
                coords = tuple(float(row[key]) for key in ("xmin", "ymin", "xmax", "ymax"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Coordenadas no numéricas para la clase {class_name!r}.") from exc
            if not np.isfinite(coords).all() or coords[2] <= coords[0] or coords[3] <= coords[1]:
                raise ValueError(f"Caja inválida para la clase {class_name!r}: {coords}")
            boxes_by_image[str(Path(row["image_path"]).expanduser().resolve())].append((class_name, coords))

    labeled_path = Path(labeled_images_file)
    labeled_images = set()
    for line in labeled_path.read_text(encoding="utf-8-sig").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            labeled_images.add(str(Path(value).expanduser().resolve()))
    unreviewed = set(boxes_by_image) - labeled_images
    if unreviewed:
        raise ValueError(
            "Todas las imágenes con cajas deben figurar también en object_labeled_images.txt. "
            f"Ejemplo: {sorted(unreviewed)[0]}"
        )
    return boxes_by_image, labeled_images


def object_detection_metrics(
    boxes_by_image: dict[str, list[tuple[str, tuple[float, float, float, float]]]],
    labeled_images: set[str],
    predictions_by_image: dict[str, list[Detection]],
    confidence: float,
    iou_threshold: float = 0.5,
) -> list[dict[str, float | int | str]]:
    results = []
    for class_name in ("person", "car", "motorcycle"):
        tp = fp = fn = 0
        support = 0
        for image_path in sorted(labeled_images):
            ground_truth = [
                box for name, box in boxes_by_image.get(image_path, []) if name == class_name
            ]
            predictions = sorted(
                [
                    det for det in predictions_by_image.get(image_path, [])
                    if det.class_name == class_name and det.confidence >= confidence
                ],
                key=lambda det: det.confidence,
                reverse=True,
            )
            support += len(ground_truth)
            matched: set[int] = set()
            for detection in predictions:
                candidates = [
                    (intersection_over_union(detection.xyxy, box), index)
                    for index, box in enumerate(ground_truth)
                    if index not in matched
                ]
                best_iou, best_index = max(candidates, default=(0.0, -1))
                if best_iou >= iou_threshold:
                    tp += 1
                    matched.add(best_index)
                else:
                    fp += 1
            fn += len(ground_truth) - len(matched)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        results.append(
            {
                "class_name": class_name,
                "confidence": confidence,
                "iou_threshold": iou_threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
    return results
