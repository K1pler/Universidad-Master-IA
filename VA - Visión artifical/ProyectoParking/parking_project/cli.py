from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .dataset import prepare_dataset, read_manifest
from .pipeline import run_experiment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config.example.json"


def _read_config(path: str | Path) -> tuple[dict, Path]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as stream:
        return json.load(stream), config_path.parent


def _resolve_optional_path(value: str, config_directory: Path) -> str:
    if not value:
        return ""
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = config_directory / candidate
    return str(candidate.resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="parking-project",
        description="Prototipo reproducible de ocupación de plazas con PKLot.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Crear una muestra y un manifiesto sin fuga por día.")
    prepare.add_argument("--dataset-root", required=True, help="Raíz extraída del dataset PKLot.")
    prepare.add_argument("--output-dir", default="outputs/prepared")
    prepare.add_argument("--config", default=str(DEFAULT_CONFIG))

    run = subparsers.add_parser("run", help="Ejecutar baselines, YOLO, DINOv2 y evaluación.")
    run.add_argument("--manifest", required=True, help="CSV creado por el subcomando prepare.")
    run.add_argument("--output-dir", default="outputs/run")
    run.add_argument("--config", default=str(DEFAULT_CONFIG))
    run.add_argument("--device", default="auto", help="auto, cpu o cuda:0.")
    run.add_argument("--object-annotations", default="", help="CSV de cajas anotadas manualmente.")
    run.add_argument("--labeled-images", default="", help="Lista de todas las imágenes revisadas, incluso sin objetos.")
    run.add_argument("--without-yolo", action="store_true", help="Omitir detección de objetos.")
    run.add_argument("--without-dino", action="store_true", help="Omitir embeddings DINOv2.")

    calibrate = subparsers.add_parser("calibrate", help="Marcar cuatro puntos para una vista rectificada.")
    calibrate.add_argument("--manifest", required=True)
    calibrate.add_argument("--view", required=True, choices=["PUCPR", "UFPR04", "UFPR05"])
    calibrate.add_argument("--config", required=True, help="Config JSON del proyecto que se actualizará.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config, config_directory = _read_config(args.config)

    if args.command == "prepare":
        manifest, summary = prepare_dataset(
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            max_development_frames=int(config["max_development_frames"]),
            external_frames=int(config["external_frames"]),
            external_view=config["external_view"],
            label_frames=int(config["object_label_frames"]),
            seed=int(config["seed"]),
        )
        print(f"Manifiesto: {manifest}")
        print(f"Resumen de preparación: {summary}")
        print(f"Plantillas de cajas/personas: {manifest.parent / 'annotations'}")
        return 0

    if args.command == "calibrate":
        from matplotlib import pyplot as plt

        records = read_manifest(args.manifest)
        candidates = [record for record in records if record.view == args.view and record.split == "train"]
        if not candidates:
            raise ValueError(f"No hay imágenes train para la vista {args.view}.")
        image = cv2.imread(str(candidates[0].image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"No se pudo leer {candidates[0].image_path}")
        figure, axis = plt.subplots(figsize=(14, 8))
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axis.set_title(
            "Marca 4 esquinas del plano: arriba-izquierda, arriba-derecha, "
            "abajo-derecha, abajo-izquierda"
        )
        axis.axis("on")
        points = plt.ginput(4, timeout=0)
        plt.close(figure)
        if len(points) != 4:
            raise ValueError("Se necesitan exactamente cuatro puntos.")
        config["homography_source_points_by_view"] = config.get("homography_source_points_by_view", {})
        config["homography_source_points_by_view"][args.view] = [
            [float(x), float(y)] for x, y in points
        ]
        config_path = Path(args.config).expanduser().resolve()
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Puntos guardados para {args.view} en {config_path}")
        return 0

    if args.object_annotations:
        config["object_annotations_csv"] = str(Path(args.object_annotations).expanduser().resolve())
    else:
        config["object_annotations_csv"] = _resolve_optional_path(
            config.get("object_annotations_csv", ""), config_directory
        )
    if args.labeled_images:
        config["object_labeled_images_file"] = str(Path(args.labeled_images).expanduser().resolve())
    else:
        config["object_labeled_images_file"] = _resolve_optional_path(
            config.get("object_labeled_images_file", ""), config_directory
        )

    run_experiment(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        config=config,
        requested_device=args.device,
        with_yolo=not args.without_yolo,
        with_dino=not args.without_dino,
    )
    print(f"Resultados guardados en: {Path(args.output_dir).expanduser().resolve()}")
    return 0
