# Instalación local de los notebooks

Instrucciones para Ubuntu o WSL con Bash. Se recomienda Python 3.11.

## 1. Dependencias del sistema

```bash
sudo apt update
sudo apt install -y build-essential curl git make ffmpeg \
  libbz2-dev libffi-dev liblzma-dev libncursesw5-dev libreadline-dev \
  libsqlite3-dev libssl-dev libxml2-dev libxmlsec1-dev tk-dev xz-utils \
  zlib1g-dev
```

## 2. Instalar y configurar pyenv

```bash
curl -fsSL https://pyenv.run | bash

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init - bash)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.profile
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.profile
echo 'eval "$(pyenv init - bash)"' >> ~/.profile

exec "$SHELL"
```

## 3. Crear el entorno e instalar los paquetes

Desde `practicas/notebooks/`:

```bash
pyenv install 3.11
pyenv virtualenv 3.11 va-master-notebooks
pyenv activate va-master-notebooks

python -m pip install --upgrade pip
python -m pip install -r requirements-base.txt
jupyter lab
```

En sesiones posteriores, basta con ejecutar:

```bash
cd /ruta/al/repositorio/practicas/notebooks
pyenv activate va-master-notebooks
jupyter lab
```

## Futuros requirements adicionales

Los futuros archivos `requirements-*.txt` se instalan en el mismo entorno:

```bash
pyenv activate va-master-notebooks
python -m pip install -r requirements-NOMBRE.txt
```

