# Trabajo-3-aplicaciones-en-sistemas-de-recomendacion-e-imagenes

---

## ⚙️ Requisitos

- Python 3.8 o superior
- `pip` o entorno virtual como `.venv`

---

## Pasos para ejecutar el proyecto

### 📦 Requisitos adicionales

Este proyecto utiliza [Git Large File Storage (LFS)](https://git-lfs.com/) para manejar modelos grandes (.pkl).  
Antes de clonar el repositorio, asegúrate de tener Git LFS instalado:

```bash
# Instalar Git LFS (una sola vez en el sistema)
git lfs install
```

### 1. Clona el repositorio

```bash
git clone https://github.com/jaco8ar/Trabajo-3-aplicaciones-en-sistemas-de-recomendacion-e-imagenes
cd Trabajo-3-aplicaciones-en-sistemas-de-recomendacion-e-imagenes
```

**Nota:** el repositorio contiene archivos grandes, por lo que el clonado puede tomar más tiempo de lo esperado.

### 2. Crea y activa un entorno virtual

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt


# macOS/Linux
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

```bash
streamlit run web_page/Home.py
```
