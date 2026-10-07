from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import platform
import time
from collections import defaultdict
from pathlib import Path

import cv2
import joblib
import numpy as np

from .baselines import best_f1_threshold, binary_metrics, foreground_difference_score
from .dataset import parse_pklot_xml, read_manifest
from .features import hog_descriptor, model_scores, train_hog_svm, train_linear_classifier
from .geometry import (
    homography_from_four_points,
    hough_ransac_orientation,
    polygon_edge_orientation,
    rectify_image,
    transform_polygon,
)
from .imaging import crop_space, draw_space_overlay
from .models import (
    extract_dinov2_features,
    infer_yolo,
    load_dinov2,
    load_yolo,
    resolve_torch_device,
    yolo_space_scores,
)
from .types import FrameRecord, SlotExample


def _read_examples(records: list[FrameRecord]) -> list[SlotExample]:
    examples: list[SlotExample] = []
    for record in records:
        image = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"No se pudo leer la imagen: {record.image_path}")
        spaces = parse_pklot_xml(record.xml_path)
        for space in spaces:
            crop = crop_space(image, space.polygon)
            examples.append(SlotExample(record, space, crop))
    if not examples:
        raise ValueError("No se encontraron plazas etiquetadas en las imágenes del manifiesto.")
    return examples


def _group_indices(examples: list[SlotExample]) -> dict[str, np.ndarray]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        groups[example.record.split].append(index)
    return {split: np.asarray(indices, dtype=np.int64) for split, indices in groups.items()}


def _fit_threshold(scores: np.ndarray, examples: list[SlotExample], indices: np.ndarray):
    labels = np.asarray([examples[int(i)].space.occupied for i in indices], dtype=np.int32)
    return best_f1_threshold(scores[indices], labels)


def _score_rows(
    examples: list[SlotExample],
    groups: dict[str, np.ndarray],
    score_columns: dict[str, np.ndarray],
    thresholds: dict[str, float],
) -> list[dict]:
    rows = []
    for split, indices in groups.items():
        for index in indices:
            example = examples[int(index)]
            row = {
                "image_path": str(example.record.image_path),
                "view": example.record.view,
                "weather": example.record.weather,
                "capture_date": example.record.capture_date,
                "split": split,
                "space_id": example.space.space_id,
                "ground_truth_occupied": int(example.space.occupied),
            }
            for method, scores in score_columns.items():
                score = float(scores[int(index)])
                row[f"{method}_score"] = score if math.isfinite(score) else ""
                row[f"{method}_occupied"] = (
                    int(score >= thresholds[method]) if math.isfinite(score) else ""
                )
            rows.append(row)
    return rows


def _write_dict_rows(path: Path, rows: list[dict], empty_fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if empty_fields:
            with path.open("w", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerow(empty_fields)
        else:
            path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _evaluate_occupancy(
    examples: list[SlotExample],
    groups: dict[str, np.ndarray],
    score_columns: dict[str, np.ndarray],
    thresholds: dict[str, float],
) -> list[dict]:
    rows: list[dict] = []
    for split, indices in groups.items():
        subgroup_keys: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index in indices:
            example = examples[int(index)]
            subgroup_keys[(example.record.view, example.record.weather)].append(int(index))
        categories: list[tuple[str, str, np.ndarray]] = [("ALL", "ALL", indices)]
        categories.extend(
            (view, weather, np.asarray(sub_indices, dtype=np.int64))
            for (view, weather), sub_indices in sorted(subgroup_keys.items())
        )
        for view, weather, category_indices in categories:
            for method, scores in score_columns.items():
                valid = np.isfinite(scores[category_indices])
                if not np.any(valid):
                    continue
                chosen = category_indices[valid]
                y_true = np.asarray([examples[int(i)].space.occupied for i in chosen], dtype=bool)
                y_pred = scores[chosen] >= thresholds[method]
                rows.append(
                    {
                        "split": split,
                        "view": view,
                        "weather": weather,
                        "method": method,
                        **binary_metrics(y_true, y_pred),
                    }
                )
    return rows


def _yolo_tune(
    examples: list[SlotExample],
    validation_indices: np.ndarray,
    detections_by_iou: dict[float, dict[str, list]],
    confidence_candidates: list[float],
    overlap_candidates: list[float],
):
    val_by_image: dict[str, list[SlotExample]] = defaultdict(list)
    for index in validation_indices:
        example = examples[int(index)]
        val_by_image[str(example.record.image_path)].append(example)

    best: tuple[tuple[float, float, float], dict] | None = None
    sweep_rows = []
    for nms_iou, image_detections in detections_by_iou.items():
        for confidence in confidence_candidates:
            for overlap in overlap_candidates:
                labels: list[bool] = []
                predictions: list[bool] = []
                for image_path, image_examples in val_by_image.items():
                    spaces = [example.space for example in image_examples]
                    scores = yolo_space_scores(
                        spaces,
                        image_detections.get(image_path, []),
                        confidence,
                        overlap,
                    )
                    for example in image_examples:
                        labels.append(example.space.occupied)
                        predictions.append(scores[example.space.space_id][0])
                metric = binary_metrics(np.asarray(labels), np.asarray(predictions))
                row = {
                    "nms_iou": nms_iou,
                    "confidence": confidence,
                    "box_overlap": overlap,
                    **metric,
                }
                sweep_rows.append(row)
                key = (float(metric["f1"]), float(metric["precision"]), float(metric["recall"]))
                if best is None or key > best[0]:
                    best = (key, row)
    if best is None:
        raise ValueError("No se pudo calibrar YOLO: el conjunto de validación está vacío.")
    return best[1], sweep_rows


def _geometry_outputs(
    records: list[FrameRecord],
    output_dir: Path,
    homography_sources: dict,
    birdseye_size: tuple[int, int],
) -> list[dict]:
    # Use one held-out image per view so geometry stays diagnostic, not a trained model.
    selected: dict[str, FrameRecord] = {}
    for record in records:
        if record.split in {"test", "test_external"}:
            selected.setdefault(record.view, record)
    rows = []
    for view, record in selected.items():
        image = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
        spaces = parse_pklot_xml(record.xml_path)
        line_stats = hough_ransac_orientation(image)
        edge_angles = [polygon_edge_orientation(space.polygon) for space in spaces]
        edge_angles = [angle for angle in edge_angles if math.isfinite(angle)]
        rows.append(
            {
                "view": view,
                "image_path": str(record.image_path),
                "weather": record.weather,
                "annotation_spaces": len(spaces),
                "median_space_edge_angle_degrees": float(np.median(edge_angles)) if edge_angles else "",
                **line_stats,
            }
        )
        slot_angle = float(np.median(edge_angles)) if edge_angles else float("nan")
        ransac_angle = line_stats["ransac_angle_degrees"]
        least_squares_angle = line_stats["least_squares_angle_degrees"]
        rows[-1]["ransac_vs_slot_angle_difference_degrees"] = (
            abs((float(ransac_angle) - slot_angle + 90.0) % 180.0 - 90.0)
            if ransac_angle is not None and math.isfinite(slot_angle)
            else ""
        )
        rows[-1]["least_squares_vs_slot_angle_difference_degrees"] = (
            abs((float(least_squares_angle) - slot_angle + 90.0) % 180.0 - 90.0)
            if least_squares_angle is not None and math.isfinite(slot_angle)
            else ""
        )
        source_points = homography_sources.get(view)
        if not source_points:
            continue
        homography, _ = homography_from_four_points(source_points, birdseye_size)
        warped = rectify_image(image, homography, birdseye_size)
        for space in spaces:
            transformed = transform_polygon(space.polygon, homography)
            color = (40, 40, 230) if space.occupied else (40, 200, 40)
            cv2.polylines(warped, [np.round(transformed).astype(np.int32)], True, color, 2, cv2.LINE_AA)
        destination = output_dir / "birdseye" / f"{view}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(destination), warped)
    return rows


def _run_yolo(
    records: list[FrameRecord],
    examples: list[SlotExample],
    groups: dict[str, np.ndarray],
    output_dir: Path,
    model_path: str,
    requested_device: str,
    image_size: int,
    nms_values: list[float],
    confidence_candidates: list[float],
    overlap_candidates: list[float],
):
    torch_device, yolo_device = resolve_torch_device(requested_device)
    model, yolo_device, _ = load_yolo(model_path, yolo_device)
    unique_records = {str(record.image_path): record for record in records}
    min_confidence = min(confidence_candidates)
    predictions: dict[float, dict[str, list]] = {value: {} for value in nms_values}
    inference_seconds = 0.0
    for image_path, record in unique_records.items():
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"No se pudo leer la imagen: {image_path}")
        for nms_iou in nms_values:
            start = time.perf_counter()
            detections = infer_yolo(
                model,
                image,
                yolo_device,
                confidence_floor=min_confidence,
                nms_iou=nms_iou,
                image_size=image_size,
            )
            inference_seconds += time.perf_counter() - start
            predictions[nms_iou][image_path] = detections

    validation_indices = groups.get("val", np.asarray([], dtype=np.int64))
    chosen, sweep_rows = _yolo_tune(
        examples,
        validation_indices,
        predictions,
        confidence_candidates,
        overlap_candidates,
    )
    selected_iou = float(chosen["nms_iou"])
    selected_detections = {
        image_path: [det for det in detections if det.confidence >= float(chosen["confidence"])]
        for image_path, detections in predictions[selected_iou].items()
    }
    slot_scores = np.zeros(len(examples), dtype=np.float32)
    slot_states = np.zeros(len(examples), dtype=bool)
    by_image_examples: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        by_image_examples[str(example.record.image_path)].append(index)
    for image_path, indices in by_image_examples.items():
        spaces = [examples[index].space for index in indices]
        mapped = yolo_space_scores(
            spaces,
            selected_detections.get(image_path, []),
            float(chosen["confidence"]),
            float(chosen["box_overlap"]),
        )
        for index in indices:
            occupied, confidence = mapped[examples[index].space.space_id]
            slot_states[index] = occupied
            slot_scores[index] = confidence

    # A small fixed sweep at the default class and overlap thresholds documents
    # the effect of NMS without using the held-out set to choose parameters.
    nms_report = []
    validation_by_image: dict[str, list[SlotExample]] = defaultdict(list)
    for index in validation_indices:
        example = examples[int(index)]
        validation_by_image[str(example.record.image_path)].append(example)
    for nms_iou, image_detections in predictions.items():
        labels, preds = [], []
        for image_path, frame_examples in validation_by_image.items():
            mapped = yolo_space_scores(
                [e.space for e in frame_examples],
                image_detections.get(image_path, []),
                confidence=float(chosen["confidence"]),
                overlap_threshold=float(chosen["box_overlap"]),
            )
            for example in frame_examples:
                labels.append(example.space.occupied)
                preds.append(mapped[example.space.space_id][0])
        nms_report.append({"nms_iou": nms_iou, **binary_metrics(np.asarray(labels), np.asarray(preds))})

    detection_rows = []
    for image_path, detections in selected_detections.items():
        for detection in detections:
            detection_rows.append(
                {
                    "image_path": image_path,
                    "class_name": detection.class_name,
                    "confidence": detection.confidence,
                    "xmin": detection.xyxy[0],
                    "ymin": detection.xyxy[1],
                    "xmax": detection.xyxy[2],
                    "ymax": detection.xyxy[3],
                }
            )
    _write_dict_rows(
        output_dir / "detections.csv",
        detection_rows,
        ["image_path", "class_name", "confidence", "xmin", "ymin", "xmax", "ymax"],
    )
    _write_dict_rows(output_dir / "yolo_validation_sweep.csv", sweep_rows)
    _write_dict_rows(output_dir / "yolo_nms_effect.csv", nms_report)
    return slot_scores, slot_states, selected_detections, chosen, inference_seconds


def run_experiment(
    manifest_path: str | Path,
    output_dir: str | Path,
    config: dict,
    requested_device: str = "auto",
    with_yolo: bool = True,
    with_dino: bool = True,
):
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = read_manifest(manifest_path)
    examples = _read_examples(records)
    groups = _group_indices(examples)
    if "train" not in groups or "val" not in groups or not ({"test", "test_external"} & groups.keys()):
        raise ValueError("El manifiesto debe contener train, val y al menos un conjunto test.")

    labels = np.asarray([example.space.occupied for example in examples], dtype=np.int32)
    train_indices = groups["train"]
    val_indices = groups["val"]
    score_columns: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    calibration_rows = []

    # Classical reference-difference baseline. Reference patches come only from
    # free training slots belonging to the same camera and slot identifier.
    references: dict[tuple[str, str], np.ndarray] = {}
    for index in train_indices:
        example = examples[int(index)]
        key = (example.record.view, example.space.space_id)
        if not example.space.occupied:
            references.setdefault(key, example.crop)
    classical_scores = np.full(len(examples), np.nan, dtype=np.float32)
    for index, example in enumerate(examples):
        reference = references.get((example.record.view, example.space.space_id))
        if reference is not None:
            classical_scores[index] = foreground_difference_score(example.crop, reference)
    threshold, calibration = _fit_threshold(classical_scores, examples, val_indices)
    score_columns["classical"] = classical_scores
    thresholds["classical"] = threshold
    calibration_rows.append({"method": "classical", "threshold": threshold, **calibration})

    # Classical hand-engineered HOG + linear SVM.
    hog_matrix = np.stack([hog_descriptor(example.crop) for example in examples])
    hog_model = train_hog_svm(hog_matrix[train_indices], labels[train_indices], config["seed"])
    hog_scores = model_scores(hog_model, hog_matrix)
    threshold, calibration = _fit_threshold(hog_scores, examples, val_indices)
    score_columns["hog_svm"] = hog_scores.astype(np.float32)
    thresholds["hog_svm"] = threshold
    calibration_rows.append({"method": "hog_svm", "threshold": threshold, **calibration})
    model_dir = output / "models"
    model_dir.mkdir(exist_ok=True)
    joblib.dump(hog_model, model_dir / "hog_svm.joblib")

    dino_model = None
    dino_features = None
    if with_dino:
        torch_device, _ = resolve_torch_device(requested_device)
        dino_model = load_dinov2(torch_device, config["dino_model"])
        dino_features = extract_dinov2_features(
            [example.crop for example in examples],
            dino_model,
            torch_device,
            int(config["dino_batch_size"]),
        )
        np.save(output / "dino_embeddings.npy", dino_features)
        del dino_model
        if torch_device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()
        dino_classifier = train_linear_classifier(
            dino_features[train_indices], labels[train_indices], config["seed"]
        )
        dino_scores = model_scores(dino_classifier, dino_features).astype(np.float32)
        threshold, calibration = _fit_threshold(dino_scores, examples, val_indices)
        score_columns["dino"] = dino_scores
        thresholds["dino"] = threshold
        calibration_rows.append({"method": "dino", "threshold": threshold, **calibration})
        joblib.dump(dino_classifier, model_dir / "dino_linear.joblib")

    selected_detections: dict[str, list] = {}
    yolo_seconds = 0.0
    yolo_choice = None
    if with_yolo:
        yolo_scores, yolo_states, selected_detections, yolo_choice, yolo_seconds = _run_yolo(
            records=records,
            examples=examples,
            groups=groups,
            output_dir=output,
            model_path=config["yolo_model"],
            requested_device=requested_device,
            image_size=int(config["yolo_image_size"]),
            nms_values=[float(v) for v in config["nms_iou_sweep"]],
            confidence_candidates=[float(v) for v in config["confidence_candidates"]],
            overlap_candidates=[float(v) for v in config["box_overlap_candidates"]],
        )
        # Detection confidence/overlap are already selected on validation.
        # Store assigned occupancy as a binary score for the fusion rule.
        score_columns["yolo"] = yolo_states.astype(np.float32)
        thresholds["yolo"] = 0.5
        calibration_rows.append(
            {
                "method": "yolo",
                "threshold": "detección asignada en validación",
                "nms_iou": yolo_choice["nms_iou"],
                "confidence": yolo_choice["confidence"],
                "box_overlap": yolo_choice["box_overlap"],
                "validation_f1": yolo_choice["f1"],
            }
        )

    if "dino" in score_columns and "yolo" in score_columns:
        fusion_scores = np.maximum(
            np.where(np.isfinite(score_columns["dino"]), score_columns["dino"], 0.0),
            np.where(np.isfinite(score_columns["yolo"]), score_columns["yolo"], 0.0),
        )
        fusion_threshold = thresholds["dino"]
        score_columns["fusion"] = fusion_scores
        thresholds["fusion"] = fusion_threshold
        calibration_rows.append(
            {
                "method": "fusion",
                "threshold": fusion_threshold,
                "rule": "occupied si DINOv2 o YOLO asignado indica ocupación",
            }
        )

    metric_rows = _evaluate_occupancy(examples, groups, score_columns, thresholds)
    prediction_rows = _score_rows(examples, groups, score_columns, thresholds)
    _write_dict_rows(output / "occupancy_metrics.csv", metric_rows)
    _write_dict_rows(output / "slot_predictions.csv", prediction_rows)
    _write_dict_rows(output / "validation_thresholds.csv", calibration_rows)

    # Save representative held-out overlays, including object boxes and the
    # fused occupancy decision. The number is capped to keep outputs compact.
    overlay_dir = output / "overlays"
    overlay_dir.mkdir(exist_ok=True)
    unique_records = {str(record.image_path): record for record in records}
    saved = 0
    examples_by_image: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        if example.record.split in {"test", "test_external"}:
            examples_by_image[str(example.record.image_path)].append(index)
    for image_path, indices in examples_by_image.items():
        if saved >= int(config["max_overlays"]):
            break
        record = unique_records[image_path]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        spaces = parse_pklot_xml(record.xml_path)
        states = {}
        if "fusion" in score_columns:
            overlay_method = "fusion"
        elif "yolo" in score_columns:
            overlay_method = "yolo"
        elif "dino" in score_columns:
            overlay_method = "dino"
        elif "hog_svm" in score_columns:
            overlay_method = "hog_svm"
        else:
            overlay_method = "classical"
        for index in indices:
            example = examples[index]
            score = score_columns[overlay_method][index]
            if math.isfinite(float(score)):
                states[example.space.space_id] = bool(score >= thresholds[overlay_method])
        canvas = draw_space_overlay(
            image,
            spaces,
            states,
            selected_detections.get(image_path, []),
        )
        output_path = overlay_dir / f"{record.view}_{record.capture_date}_{Path(image_path).stem}.jpg"
        cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        saved += 1

    homography_sources = config.get("homography_source_points_by_view", {})
    geometry_rows = _geometry_outputs(
        records,
        output,
        homography_sources,
        tuple(int(v) for v in config["birdseye_size"]),
    )
    _write_dict_rows(output / "geometry_report.csv", geometry_rows)

    object_metrics: list[dict] = []
    annotation_csv = config.get("object_annotations_csv", "")
    labeled_images_file = config.get("object_labeled_images_file", "")
    if bool(annotation_csv) != bool(labeled_images_file):
        raise ValueError(
            "Para evaluar cajas se necesitan tanto object_annotations_csv como "
            "object_labeled_images_file."
        )
    if annotation_csv and not with_yolo:
        raise ValueError("La evaluación de cajas requiere ejecutar YOLO; quita --without-yolo.")
    if annotation_csv and labeled_images_file:
        from .evaluation import object_detection_metrics, read_object_annotations

        boxes_by_image, labeled_images = read_object_annotations(annotation_csv, labeled_images_file)
        if labeled_images:
            test_paths = {
                str(record.image_path)
                for record in records
                if record.split in {"test", "test_external"}
            }
            outside_test = labeled_images - test_paths
            if outside_test:
                raise ValueError(
                    "Las imágenes anotadas para evaluación deben pertenecer a test/test_external. "
                    f"Ejemplo fuera de test: {sorted(outside_test)[0]}"
                )
            confidence = float(yolo_choice["confidence"]) if yolo_choice else 0.25
            object_metrics = object_detection_metrics(
                boxes_by_image,
                labeled_images,
                selected_detections,
                confidence,
                float(config["object_iou_threshold"]),
            )
            _write_dict_rows(output / "object_detection_metrics.csv", object_metrics)

    package_versions = {}
    package_candidates = {
        "numpy": ("numpy",),
        "opencv": ("opencv-python-headless", "opencv-python"),
        "scikit-learn": ("scikit-learn",),
        "torch": ("torch",),
        "torchvision": ("torchvision",),
        "ultralytics": ("ultralytics",),
    }
    for label, candidates in package_candidates.items():
        version = None
        for package in candidates:
            try:
                version = importlib.metadata.version(package)
                break
            except importlib.metadata.PackageNotFoundError:
                continue
        package_versions[label] = version

    summary = {
        "manifest": str(Path(manifest_path).resolve()),
        "seed": int(config["seed"]),
        "configuration": config,
        "python_version": platform.python_version(),
        "package_versions": package_versions,
        "frames": len(records),
        "labeled_parking_spaces": len(examples),
        "frames_by_split": {split: int(len(indices)) for split, indices in groups.items()},
        "thresholds": thresholds,
        "yolo_validation_choice": yolo_choice,
        "yolo_inference_seconds_all_nms_sweeps": yolo_seconds,
        "mean_yolo_seconds_per_frame_per_nms_setting": yolo_seconds / max(1, len(records) * len(config["nms_iou_sweep"])) if with_yolo else None,
        "object_box_metrics_available": any(int(row.get("support", 0)) > 0 for row in object_metrics),
        "notes": [
            "El conjunto test se usa solo para evaluación final.",
            "El XML PKLot aporta ocupación por plaza, no cajas por clase de objeto.",
            "La homografía se genera únicamente para vistas con cuatro puntos configurados.",
        ],
    }
    (output / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
