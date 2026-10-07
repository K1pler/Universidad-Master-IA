# Detección de ocupación y objetos en parking

Prototipo offline para el miniproyecto de Visión Artificial. Usa las anotaciones de plazas de PKLot para evaluar la ocupación y añade detecciones de personas, coches y motos con un detector COCO preentrenado.

## Alcance y regla de ocupación

- Cada imagen produce un estado por plaza: `libre` u `ocupada`.
- Las detecciones de coche y moto pueden marcar una plaza como ocupada. Las personas se dibujan y evalúan como clase independiente, pero no alteran el estado de la plaza.
- Se comparan un baseline por diferencia/Otsu, HOG+SVM, embeddings DINOv2+clasificador lineal y asignación geométrica de cajas YOLO.
- La fusión YOLO+DINOv2 se presenta como una alternativa a medir; no se presupone que mejore los baselines.
- PKLot proporciona la verdad terreno de ocupación por plaza, pero no cajas de persona/coche/moto. Para esas métricas se requiere la anotación manual de imágenes de prueba.

## Recursos del curso

Los notebooks de referencia son:

- `../Curso_pasado/10_YOLO_OD_ALOTextures_ACG.ipynb`: YOLOv8n, transferencia e inferencia.
- `../Curso_pasado/00_EjemplosUsoDINOv2.ipynb`: embeddings DINOv2 y clasificación lineal.
- `../Curso_pasado/05_OtsuContoursHomog2DWrappersC_AC.ipynb`: Otsu, contornos y homografías.
- `../Curso_pasado/06_RectasRANSACyHT_C.ipynb`: RANSAC y transformada de Hough.

El entorno base ya incluye OpenCV, NumPy, scikit-learn, matplotlib e ipywidgets. PyTorch, torchvision, Ultralytics y joblib se añaden con `requirements-project.txt`; consulta `INSTALL.md`.

## Datos

Descarga PKLot desde su [página oficial](https://web.inf.ufpr.br/vri/databases/parking-lot-database/) y extrae el árbol completo. `DATASET_ROOT` debe ser la carpeta donde se ven `parking1a`, `parking1b` y `parking2` (o sus equivalentes `UFPR04`, `UFPR05`, `PUCPR`). Dentro, el código busca las imágenes de escena completa y el XML del mismo nombre base. No apuntes a `PKLotSegmented`, que contiene recortes de plazas. Un ejemplo en Windows sería `Path("D:/Datos/PKLot/PKLot")` si dentro de esa carpeta aparece `parking1a`.

Si se usa CNRPark+EXT como validación complementaria, debe tratarse como un conjunto independiente de recortes etiquetados `free/busy`; el importador PKLot de este prototipo no lo convierte automáticamente y no permite evaluar detecciones por clase.

La preparación toma hasta 600 imágenes de `UFPR04`/`UFPR05` y hasta 150 de `PUCPR` para prueba externa. La muestra se distribuye por vista, clima y día; después separa los días cronológicamente en train/val/test y asigna un mismo día al mismo split en ambas vistas de desarrollo, para evitar que fotogramas casi iguales aparezcan en conjuntos distintos. La semilla y el manifiesto quedan guardados.

## Ejecución

Desde esta carpeta, copia `config.example.json` a `config.json` y ajusta la configuración solo si hace falta. El dataset no se copia al repositorio.

También se puede ejecutar el flujo por etapas en Jupyter abriendo `ProyectoParking.ipynb`. El notebook localiza e importa el paquete del proyecto, prepara el manifiesto, ejecuta los métodos y muestra las métricas y un overlay; edita `DATASET_ROOT` antes de preparar los datos.

```bash
python -m parking_project prepare \
  --dataset-root "/ruta/PKLot/PKLot" \
  --output-dir outputs/prepared \
  --config config.json

python -m parking_project run \
  --manifest outputs/prepared/manifest.csv \
  --output-dir outputs/run \
  --config config.json \
  --device auto
```

La ejecución completa calcula HOG, descarga y usa DINOv2-S/14, y ejecuta YOLOv8n con varios valores NMS. Para probar solo la parte clásica mientras se prepara el entorno de modelos:

```bash
python -m parking_project run --manifest outputs/prepared/manifest.csv \
  --output-dir outputs/classical --config config.json --without-yolo --without-dino
```

La primera ejecución con modelos necesita conexión para descargar los pesos. En ejecuciones posteriores se reutiliza la caché de Torch/Ultralytics.

## Anotaciones de objetos

`prepare` crea `outputs/prepared/annotations/object_labeled_images.txt` y `object_boxes.csv`.

1. Revisa todas las imágenes enumeradas en `object_labeled_images.txt`, incluidas las que no tengan objetos. Conserva solo imágenes de `test` o `test_external`.
2. En `object_boxes.csv`, añade una fila por caja con coordenadas de píxel y una de estas clases: `person`, `car`, `motorcycle`.
3. Ejecuta de nuevo con `--object-annotations outputs/prepared/annotations/object_boxes.csv --labeled-images outputs/prepared/annotations/object_labeled_images.txt`.

Si una clase tiene muy pocos ejemplos, informa su soporte y muestra ejemplos cualitativos; no interpretes una métrica con soporte insuficiente como rendimiento concluyente.

## Homografía y parámetros

Para generar una vista superior, añade a `homography_source_points_by_view` cuatro puntos en orden horario desde la esquina superior izquierda de la zona planar visible para esa vista. El destino es el rectángulo de `birdseye_size`. Si no se configura una vista, se omite su imagen rectificada; las demás etapas siguen funcionando.

La selección interactiva de puntos se puede guardar en `config.json` con:

```bash
python -m parking_project calibrate --manifest outputs/prepared/manifest.csv \
  --view UFPR04 --config config.json
```

Hough detecta segmentos de marcas viales y RANSAC estima su orientación dominante, comparándola con la media angular ponderada. Se guarda como análisis geométrico auxiliar.

## Resultados

`outputs/run/` incluye:

- `slot_predictions.csv`: verdad terreno, puntuaciones y estados de cada método por plaza.
- `occupancy_metrics.csv`: precisión, recall, F1, accuracy y matriz de confusión, separados por split, vista y clima.
- `validation_thresholds.csv`, `yolo_validation_sweep.csv` y `yolo_nms_effect.csv`: calibración y análisis de umbrales en validación.
- `detections.csv` y, cuando se facilitan cajas manuales, `object_detection_metrics.csv` por clase con IoU 0.5.
- `geometry_report.csv` con la diferencia angular de Hough/RANSAC respecto a las plazas anotadas, `birdseye/` cuando hay homografías configuradas, y hasta 20 ejemplos visuales en `overlays/`.
- `run_summary.json` con configuración, semilla y versiones instaladas, clasificadores lineales y embeddings DINOv2 para apoyar la reproducibilidad.

La evaluación final no fija una precisión mínima: se espera comparar métodos, explicar errores y documentar condiciones, parámetros, versiones y semillas.

## Referencias para la memoria

- Almeida et al., “PKLot – A robust dataset for parking lot classification”, *Expert Systems with Applications*, 42(11), 4937–4949, 2015. Citar además la [página oficial de descarga y licencia](https://web.inf.ufpr.br/vri/databases/parking-lot-database/).
- [Ultralytics: detector y clases COCO](https://docs.ultralytics.com/datasets/detect/coco).
- Oquab et al., “DINOv2: Learning Robust Visual Features without Supervision”, 2023; [repositorio/model card oficial](https://github.com/facebookresearch/dinov2).
