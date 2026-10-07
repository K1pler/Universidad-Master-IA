from __future__ import annotations

import csv
import json
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .types import FrameRecord, ParkingSpace

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
VIEW_ALIASES = {
    "pucpr": "PUCPR",
    "parking2": "PUCPR",
    "ufpr04": "UFPR04",
    "parking1a": "UFPR04",
    "ufpr05": "UFPR05",
    "parking1b": "UFPR05",
}
WEATHER_ALIASES = {
    "sunny": "sunny",
    "cloudy": "cloudy",
    "overcast": "cloudy",
    "rainy": "rainy",
}
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _canonical_view(parts: Iterable[str]) -> str:
    for part in parts:
        canonical = VIEW_ALIASES.get(part.lower())
        if canonical:
            return canonical
    return "UNKNOWN"


def _canonical_weather(parts: Iterable[str]) -> str:
    for part in parts:
        canonical = WEATHER_ALIASES.get(part.lower())
        if canonical:
            return canonical
    return "unknown"


def _capture_date(path: Path) -> str:
    for part in path.parts:
        if DATE_PATTERN.match(part):
            return part
    match = re.match(r"(\d{4}-\d{2}-\d{2})", path.stem)
    return match.group(1) if match else "undated"


def discover_records(dataset_root: str | Path) -> list[FrameRecord]:
    """Find full-frame PKLot images with a same-stem XML annotation."""
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"No existe el directorio del dataset: {root}")

    records: list[FrameRecord] = []
    for image_path in sorted(root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        parts_lower = {part.lower() for part in image_path.parts}
        if parts_lower.intersection({"pklotsegmented", "patches", "cropped"}):
            continue
        xml_path = image_path.with_suffix(".xml")
        if not xml_path.is_file():
            # Some extracted archives preserve uppercase extensions.
            xml_candidates = list(image_path.parent.glob(image_path.stem + ".XML"))
            if not xml_candidates:
                continue
            xml_path = xml_candidates[0]

        relative_parts = image_path.relative_to(root).parts
        records.append(
            FrameRecord(
                image_path=image_path,
                xml_path=xml_path,
                view=_canonical_view(relative_parts),
                weather=_canonical_weather(relative_parts),
                capture_date=_capture_date(image_path),
            )
        )
    if not records:
        raise FileNotFoundError(
            f"No se encontraron imágenes con XML asociado bajo {root}. "
            "Comprueba que dataset_root apunta al árbol extraído de PKLot."
        )
    unknown = sum(record.view == "UNKNOWN" for record in records)
    if unknown:
        raise ValueError(
            f"{unknown} imágenes no están bajo una carpeta PUCPR/UFPR04/UFPR05 "
            "(o parking1a/parking1b/parking2). Ajusta el árbol del dataset."
        )
    return records


def _tag_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1].lower()


def _parse_occupied(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "occupied", "busy"}:
        return True
    if normalized in {"0", "false", "vacant", "free", "empty"}:
        return False
    return None


def _rotated_rectangle(space_element: ET.Element) -> np.ndarray | None:
    rotated = next((e for e in space_element if _tag_name(e) == "rotatedrect"), None)
    if rotated is None:
        return None
    center = next((e for e in rotated if _tag_name(e) == "center"), None)
    size = next((e for e in rotated if _tag_name(e) == "size"), None)
    angle = next((e for e in rotated if _tag_name(e) == "angle"), None)
    if center is None or size is None:
        return None
    try:
        cx, cy = float(center.attrib["x"]), float(center.attrib["y"])
        width, height = float(size.attrib["w"]), float(size.attrib["h"])
        degrees = float(angle.attrib.get("d", 0.0)) if angle is not None else 0.0
    except (KeyError, ValueError):
        return None
    corners = cv2.boxPoints(((cx, cy), (width, height), degrees))
    return np.asarray(corners, dtype=np.float32)


def parse_pklot_xml(xml_path: str | Path) -> list[ParkingSpace]:
    """Read PKLot space polygons and binary occupancy labels from one XML file."""
    root = ET.parse(xml_path).getroot()
    spaces: list[ParkingSpace] = []
    generated_id = 0
    for element in root.iter():
        if _tag_name(element) != "space":
            continue
        occupied = _parse_occupied(element.attrib.get("occupied"))
        if occupied is None:
            continue
        generated_id += 1
        space_id = element.attrib.get("id", str(generated_id))

        contour = next((e for e in element if _tag_name(e) == "contour"), None)
        points: list[tuple[float, float]] = []
        if contour is not None:
            for point in contour:
                if _tag_name(point) != "point":
                    continue
                try:
                    points.append((float(point.attrib["x"]), float(point.attrib["y"])))
                except (KeyError, ValueError):
                    points = []
                    break
        polygon = np.asarray(points, dtype=np.float32) if len(points) >= 3 else _rotated_rectangle(element)
        if polygon is None or len(polygon) < 3 or not np.isfinite(polygon).all():
            continue
        spaces.append(ParkingSpace(space_id=space_id, polygon=polygon, occupied=occupied))
    return spaces


def _balanced_sample(
    records: list[FrameRecord], limit: int, seed: int, include_day: bool = False
) -> list[FrameRecord]:
    if limit <= 0:
        return []
    if len(records) <= limit:
        return list(records)
    rng = random.Random(seed)
    groups: dict[tuple[str, ...], list[FrameRecord]] = defaultdict(list)
    for record in records:
        key = (record.view, record.weather, record.capture_date) if include_day else (record.view, record.weather)
        groups[key].append(record)
    for group in groups.values():
        rng.shuffle(group)
    keys = sorted(groups)
    rng.shuffle(keys)
    selected: list[FrameRecord] = []
    while len(selected) < limit:
        progressed = False
        for key in keys:
            if groups[key] and len(selected) < limit:
                selected.append(groups[key].pop())
                progressed = True
        if not progressed:
            break
    return selected


def _assign_day_splits(records: list[FrameRecord]) -> list[FrameRecord]:
    # Keep a calendar day in one split across both development camera views.
    days = sorted({record.capture_date for record in records})
    n = len(days)
    if n < 3:
        raise ValueError(
            f"La muestra solo cubre {n} días. Aumenta max_development_frames "
            "para poder separar train/val/test por día."
        )
    n_train = max(1, int(round(0.70 * n)))
    n_val = max(1, int(round(0.15 * n)))
    if n_train + n_val >= n:
        n_train = n - 2
        n_val = 1
    split_by_day: dict[str, str] = {}
    for index, day in enumerate(days):
        if index < n_train:
            split_by_day[day] = "train"
        elif index < n_train + n_val:
            split_by_day[day] = "val"
        else:
            split_by_day[day] = "test"
    return [
        FrameRecord(
            image_path=r.image_path,
            xml_path=r.xml_path,
            view=r.view,
            weather=r.weather,
            capture_date=r.capture_date,
            split=split_by_day[r.capture_date],
        )
        for r in records
    ]


def _write_manifest(records: list[FrameRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["image_path", "xml_path", "view", "weather", "capture_date", "split"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "image_path": str(record.image_path),
                    "xml_path": str(record.xml_path),
                    "view": record.view,
                    "weather": record.weather,
                    "capture_date": record.capture_date,
                    "split": record.split,
                }
            )


def read_manifest(path: str | Path) -> list[FrameRecord]:
    manifest = Path(path).expanduser().resolve()
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        rows = csv.DictReader(stream)
        required = {"image_path", "xml_path", "view", "weather", "capture_date", "split"}
        if not required.issubset(rows.fieldnames or []):
            raise ValueError(f"El manifiesto debe contener las columnas: {sorted(required)}")
        records = [
            FrameRecord(
                image_path=Path(row["image_path"]).expanduser().resolve(),
                xml_path=Path(row["xml_path"]).expanduser().resolve(),
                view=row["view"],
                weather=row["weather"],
                capture_date=row["capture_date"],
                split=row["split"],
            )
            for row in rows
        ]
    if not records:
        raise ValueError("El manifiesto no contiene imágenes.")
    return records


def _write_object_label_templates(
    records: list[FrameRecord], output_dir: Path, count: int, seed: int
) -> None:
    candidates = [r for r in records if r.split in {"test", "test_external"}]
    selected = _balanced_sample(candidates, count, seed + 1, include_day=True)
    annotations_dir = output_dir / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    with (annotations_dir / "object_labeled_images.txt").open("w", encoding="utf-8") as stream:
        stream.write("# Una ruta de imagen por línea, incluyendo las imágenes revisadas sin objetos.\n")
        for record in selected:
            stream.write(str(record.image_path) + "\n")
    with (annotations_dir / "object_boxes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_path", "class_name", "xmin", "ymin", "xmax", "ymax"])


def prepare_dataset(
    dataset_root: str | Path,
    output_dir: str | Path,
    max_development_frames: int = 600,
    external_frames: int = 150,
    external_view: str = "PUCPR",
    label_frames: int = 100,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Create a balanced frame sample and day-separated manifest."""
    if max_development_frames < 1:
        raise ValueError("max_development_frames debe ser mayor que cero.")
    if external_frames < 1:
        raise ValueError("external_frames debe ser mayor que cero.")
    if label_frames < 0:
        raise ValueError("label_frames no puede ser negativo.")
    output = Path(output_dir).expanduser().resolve()
    records = discover_records(dataset_root)
    canonical_external = VIEW_ALIASES.get(external_view.lower(), external_view)
    external = [r for r in records if r.view == canonical_external]
    development = [r for r in records if r.view != canonical_external]
    if not external:
        raise ValueError(
            f"No se encuentra la vista externa {canonical_external}. "
            f"Vistas presentes: {sorted({r.view for r in records})}"
        )
    if len({r.view for r in development}) < 2:
        raise ValueError("Se esperaban dos vistas de desarrollo distintas a la vista externa.")

    # Sample evenly across camera, weather and capture day before making
    # chronological day-level splits, so one busy day cannot dominate the set.
    sampled_dev = _balanced_sample(development, max_development_frames, seed, include_day=True)
    sampled_external = _balanced_sample(external, external_frames, seed + 17, include_day=True)
    dev_with_splits = _assign_day_splits(sampled_dev)
    external_with_split = [
        FrameRecord(r.image_path, r.xml_path, r.view, r.weather, r.capture_date, "test_external")
        for r in sampled_external
    ]
    manifest_records = sorted(
        dev_with_splits + external_with_split,
        key=lambda r: (r.view, r.capture_date, r.image_path.name),
    )

    manifest_path = output / "manifest.csv"
    _write_manifest(manifest_records, manifest_path)
    _write_object_label_templates(manifest_records, output, label_frames, seed)
    counts = Counter((r.split, r.view, r.weather) for r in manifest_records)
    summary = {
        "dataset_root": str(Path(dataset_root).expanduser().resolve()),
        "seed": seed,
        "requested_development_frames": max_development_frames,
        "requested_external_frames": external_frames,
        "external_view": canonical_external,
        "total_frames": len(manifest_records),
        "frames_by_split_view_weather": {
            "|".join(key): value for key, value in sorted(counts.items())
        },
        "split_policy": "días globales cronológicos compartidos entre vistas de desarrollo; PUCPR reservado como test externo",
        "object_label_candidates": min(label_frames, sum(r.split.startswith("test") for r in manifest_records)),
    }
    summary_path = output / "preparation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path, summary_path
