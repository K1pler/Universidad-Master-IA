# Instalación

El entorno base del curso está documentado en `../INSTALL.md` y `../requirements-base.txt`. Se recomienda Python 3.11 en Ubuntu/WSL, como en las instrucciones del curso.

1. Activa el entorno Python del curso e instala las dependencias base desde `Prácticas/`:

   ```bash
   python -m pip install -r requirements-base.txt
   ```

2. Para la RTX 2070 Super, instala PyTorch y torchvision con el selector oficial para la versión de CUDA disponible: <https://pytorch.org/get-started/locally/>.

3. Desde `Prácticas/ProyectoParking/`, instala las dependencias adicionales:

   ```bash
   python -m pip install -r requirements-project.txt
   ```

4. La primera ejecución descarga los pesos `yolov8n.pt` y el modelo DINOv2-S/14 de los repositorios correspondientes. Después, el procesamiento puede ejecutarse localmente.

Comprueba que CUDA está disponible con `python -c "import torch; print(torch.cuda.is_available())"`. Si devuelve `False`, se puede ejecutar con `--device cpu`, aunque será más lento.

