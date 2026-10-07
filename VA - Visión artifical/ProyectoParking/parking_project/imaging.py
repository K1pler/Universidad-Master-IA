from __future__ import annotations

import cv2
import numpy as np


def order_quad(points: np.ndarray) -> np.ndarray:
    """Return four corners in clockwise image order, starting at top-left."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) != 4:
        hull = cv2.convexHull(points).reshape(-1, 2)
        if len(hull) != 4:
            rectangle = cv2.minAreaRect(points)
            points = cv2.boxPoints(rectangle)
        else:
            points = hull
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    start = int(np.argmin(ordered.sum(axis=1)))
    ordered = np.roll(ordered, -start, axis=0)

    first_edge = np.linalg.norm(ordered[1] - ordered[0])
    second_edge = np.linalg.norm(ordered[2] - ordered[1])
    if first_edge < second_edge:
        ordered = np.roll(ordered, -1, axis=0)
    return ordered.astype(np.float32)


def crop_space(
    image_bgr: np.ndarray, polygon: np.ndarray, output_size: tuple[int, int] = (96, 48)
) -> np.ndarray:
    """Perspective-normalize a parking-space polygon to a fixed-size BGR patch."""
    width, height = output_size
    corners = order_quad(polygon)
    target = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(corners, target)
    return cv2.warpPerspective(
        image_bgr,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def draw_space_overlay(
    image_bgr: np.ndarray,
    spaces: list,
    states: dict[str, bool],
    detections: list,
) -> np.ndarray:
    canvas = image_bgr.copy()
    for space in spaces:
        occupied = states.get(space.space_id, space.occupied)
        color = (40, 40, 230) if occupied else (40, 200, 40)
        polygon = np.round(space.polygon).astype(np.int32)
        cv2.polylines(canvas, [polygon], True, color, 2, cv2.LINE_AA)
        x, y = polygon[0]
        label = f"{space.space_id}: ocupada" if occupied else f"{space.space_id}: libre"
        cv2.putText(canvas, label, (int(x), max(18, int(y) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    class_colors = {"person": (255, 170, 0), "car": (0, 200, 255), "motorcycle": (230, 0, 230)}
    for detection in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in detection.xyxy)
        color = class_colors.get(detection.class_name, (255, 255, 255))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.putText(canvas, label, (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return canvas


def pad_for_dino(crop_bgr: np.ndarray) -> np.ndarray:
    """Resize a wide slot crop and pad it without changing its aspect ratio."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 112), interpolation=cv2.INTER_AREA)
    padded = np.zeros((224, 224, 3), dtype=np.uint8)
    padded[56:168, :, :] = resized
    return padded

