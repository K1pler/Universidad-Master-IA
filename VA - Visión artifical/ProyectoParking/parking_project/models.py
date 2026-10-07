from __future__ import annotations

import cv2
import numpy as np

from .imaging import pad_for_dino
from .types import Detection, ParkingSpace


def resolve_torch_device(requested: str = "auto") -> tuple[str, int | str]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Falta PyTorch. Instálalo siguiendo INSTALL.md.") from exc
    if requested == "auto":
        torch_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        torch_device = requested
    if torch_device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Se solicitó {torch_device}, pero CUDA no está disponible.")
        yolo_device: int | str = 0
    else:
        yolo_device = "cpu"
    return torch_device, yolo_device


def load_dinov2(device: str, model_name: str = "dinov2_vits14"):
    import torch

    model = torch.hub.load("facebookresearch/dinov2", model_name, trust_repo=True)
    model.to(device).eval()
    return model


def extract_dinov2_features(
    crops: list[np.ndarray], model, device: str, batch_size: int = 32
) -> np.ndarray:
    import torch

    if not crops:
        return np.empty((0, 384), dtype=np.float32)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    features: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(crops), batch_size):
            batch = np.stack([pad_for_dino(crop) for crop in crops[start : start + batch_size]])
            tensor = torch.from_numpy(batch).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
            tensor = (tensor - mean) / std
            embedding = model(tensor)
            if isinstance(embedding, dict):
                embedding = embedding["x_norm_clstoken"]
            features.append(embedding.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(features, axis=0)


def load_yolo(model_path: str, yolo_device: int | str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Falta Ultralytics. Instálalo siguiendo INSTALL.md.") from exc
    model = YOLO(model_path)
    names = model.names
    class_ids = [
        class_id
        for class_id, name in (names.items() if isinstance(names, dict) else enumerate(names))
        if str(name).lower() in {"person", "car", "motorcycle"}
    ]
    return model, yolo_device, class_ids


def infer_yolo(
    model,
    image_bgr: np.ndarray,
    yolo_device: int | str,
    confidence_floor: float,
    nms_iou: float,
    image_size: int = 640,
) -> list[Detection]:
    names = model.names
    class_ids = [
        class_id
        for class_id, name in (names.items() if isinstance(names, dict) else enumerate(names))
        if str(name).lower() in {"person", "car", "motorcycle"}
    ]
    result = model.predict(
        source=image_bgr,
        imgsz=image_size,
        conf=confidence_floor,
        iou=nms_iou,
        classes=class_ids,
        device=yolo_device,
        verbose=False,
    )[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    output: list[Detection] = []
    for box, score, class_id in zip(boxes, scores, classes):
        name = names[int(class_id)] if isinstance(names, dict) else names[int(class_id)]
        output.append(
            Detection(
                class_name=str(name).lower(),
                confidence=float(score),
                xyxy=tuple(float(value) for value in box),
            )
        )
    return output


def box_slot_overlap(detection: Detection, space: ParkingSpace) -> float:
    x1, y1, x2, y2 = detection.xyxy
    box_polygon = np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    slot_polygon = np.asarray(space.polygon, dtype=np.float32)
    slot_area = abs(float(cv2.contourArea(slot_polygon)))
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if slot_area <= 0.0 or box_area <= 0.0:
        return 0.0
    try:
        intersection, _ = cv2.intersectConvexConvex(slot_polygon, box_polygon)
    except cv2.error:
        return 0.0
    return float(intersection / box_area)


def yolo_space_scores(
    spaces: list[ParkingSpace],
    detections: list[Detection],
    confidence: float,
    overlap_threshold: float,
) -> dict[str, tuple[bool, float]]:
    vehicle_detections = [
        det for det in detections
        if det.class_name in {"car", "motorcycle"} and det.confidence >= confidence
    ]
    scores: dict[str, tuple[bool, float]] = {}
    for space in spaces:
        matches = [
            det.confidence
            for det in vehicle_detections
            if box_slot_overlap(det, space) >= overlap_threshold
        ]
        best = max(matches, default=0.0)
        scores[space.space_id] = (bool(matches), float(best))
    return scores

