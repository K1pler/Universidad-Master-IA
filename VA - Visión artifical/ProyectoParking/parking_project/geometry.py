from __future__ import annotations

import math

import cv2
import numpy as np


def homography_from_four_points(source_points: list[list[float]], output_size: tuple[int, int]):
    if len(source_points) != 4:
        raise ValueError("La homografía necesita exactamente cuatro puntos de origen.")
    width, height = output_size
    source = np.asarray(source_points, dtype=np.float32)
    destination = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    return matrix, destination


def transform_polygon(polygon: np.ndarray, homography: np.ndarray) -> np.ndarray:
    points = np.asarray(polygon, dtype=np.float32).reshape(1, -1, 2)
    return cv2.perspectiveTransform(points, homography).reshape(-1, 2)


def rectify_image(image: np.ndarray, homography: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    width, height = output_size
    return cv2.warpPerspective(image, homography, (width, height))


def _angular_difference(a: float, b: float) -> float:
    return abs((a - b + math.pi / 2) % math.pi - math.pi / 2)


def _weighted_orientation(angles: np.ndarray, weights: np.ndarray) -> float:
    # Line orientation is periodic over pi, so average doubled angles.
    sine = float(np.sum(weights * np.sin(2.0 * angles)))
    cosine = float(np.sum(weights * np.cos(2.0 * angles)))
    return 0.5 * math.atan2(sine, cosine)


def hough_ransac_orientation(
    image_bgr: np.ndarray,
    canny_low: int = 60,
    canny_high: int = 160,
    angle_tolerance_degrees: float = 5.0,
    minimum_length: int = 35,
) -> dict[str, float | int | None]:
    """Estimate the dominant parking-marking direction with Hough + angular RANSAC."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=35,
        minLineLength=minimum_length,
        maxLineGap=12,
    )
    if lines is None:
        return {"segments": 0, "ransac_angle_degrees": None, "least_squares_angle_degrees": None, "inlier_fraction": 0.0}

    segments = lines.reshape(-1, 4).astype(np.float64)
    dx = segments[:, 2] - segments[:, 0]
    dy = segments[:, 3] - segments[:, 1]
    lengths = np.hypot(dx, dy)
    valid = lengths >= minimum_length
    segments, dx, dy, lengths = segments[valid], dx[valid], dy[valid], lengths[valid]
    if lengths.size == 0:
        return {"segments": 0, "ransac_angle_degrees": None, "least_squares_angle_degrees": None, "inlier_fraction": 0.0}

    angles = np.arctan2(dy, dx)
    least_squares = _weighted_orientation(angles, lengths)
    tolerance = math.radians(angle_tolerance_degrees)
    best_mask = np.zeros(len(angles), dtype=bool)
    best_weight = -1.0
    # Each Hough segment supplies one hypothesis for the common row direction.
    for hypothesis in angles:
        mask = np.asarray([_angular_difference(angle, hypothesis) <= tolerance for angle in angles])
        weight = float(np.sum(lengths[mask]))
        if weight > best_weight:
            best_weight, best_mask = weight, mask
    robust = _weighted_orientation(angles[best_mask], lengths[best_mask])
    total_weight = float(np.sum(lengths))
    return {
        "segments": int(len(angles)),
        "ransac_angle_degrees": math.degrees(robust),
        "least_squares_angle_degrees": math.degrees(least_squares),
        "inlier_fraction": float(best_weight / total_weight) if total_weight else 0.0,
    }


def polygon_edge_orientation(polygon: np.ndarray) -> float:
    points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    valid = lengths > 1.0
    if not np.any(valid):
        return float("nan")
    return math.degrees(_weighted_orientation(np.arctan2(edges[valid, 1], edges[valid, 0]), lengths[valid]))

