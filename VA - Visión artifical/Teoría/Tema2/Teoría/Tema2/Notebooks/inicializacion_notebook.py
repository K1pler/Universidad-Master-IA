# -*- coding: utf-8 -*-
"""
Plantilla de Inicialización para los Notebooks de Visión Artificial.
Este script contiene la celda de inicio recomendada para verificar
que el entorno interactivo (ipywidgets e ipympl) está bien configurado.
"""

import sys

# =============================================================================
# NOTA IMPORTANTE PARA JUPYTER NOTEBOOKS:
# Para que los gráficos interactivos funcionen, la PRIMERA línea de la primera
# celda de tu cuaderno de Jupyter DEBE ser la directiva mágica de ipympl:
#
# %matplotlib widget
# =============================================================================

print("=== Comprobando dependencias de Visión Artificial ===")

# 1. Comprobación de paquetes numéricos y de álgebra lineal
try:
    import numpy as np
    print(f"[OK] NumPy versión: {np.__version__}")
except ImportError:
    print("[CRÍTICO] NumPy no está instalado. Ejecuta: pip install numpy")

try:
    import scipy
    from scipy.linalg import rq, cholesky
    print(f"[OK] SciPy versión: {scipy.__version__}")
except ImportError:
    print("[CRÍTICO] SciPy no está instalado o faltan submódulos de álgebra (rq, Cholesky).")

# 2. Comprobación de visión artificial y gráficos
try:
    import cv2
    print(f"[OK] OpenCV (cv2) versión: {cv2.__version__}")
except ImportError:
    print("[CRÍTICO] OpenCV no está instalado. Ejecuta: pip install opencv-python")

try:
    import matplotlib
    import matplotlib.pyplot as plt
    print(f"[OK] Matplotlib versión: {matplotlib.__version__}")
except ImportError:
    print("[CRÍTICO] Matplotlib no está instalado. Ejecuta: pip install matplotlib")

# 3. Comprobación de interactividad en Jupyter (Jupyter Widgets)
try:
    import ipywidgets as widgets
    import ipympl
    print(f"[OK] ipywidgets versión: {widgets.__version__}")
    print("[OK] ipympl (backend interactivo) detectado correctamente.")
except ImportError:
    print("[ADVERTENCIA] ipywidgets o ipympl no están instalados.")
    print("              Los gráficos interactivos y la selección de puntos NO funcionarán.")
    print("              Instálalos ejecutando: pip install ipywidgets ipympl")

print("\n=== Instrucciones de inicialización para alumnos ===")
print("1. Copia este bloque de código en la primera celda de tu cuaderno.")
print("2. Escribe '%matplotlib widget' (sin comillas) en la línea 1 de la celda.")
print("3. Si todas las comprobaciones marcan [OK], tu entorno está 100% listo para las prácticas.")
