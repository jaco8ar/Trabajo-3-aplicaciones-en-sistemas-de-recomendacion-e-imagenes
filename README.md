# Trabajo-3-aplicaciones-en-sistemas-de-recomendacion-e-imagenes

---

## ⚙️ Requisitos

- Python 3.8 o superior
- `pip` o entorno virtual como `.venv`

---

## Pasos para ejecutar el proyecto

### 1. Clona el repositorio

```bash
git clone https://github.com/tuusuario/proyecto-demanda-turistica.git
cd proyecto-demanda-turistica
```

## 📦 Requisitos adicionales

Este proyecto utiliza [Git Large File Storage (LFS)](https://git-lfs.com/) para manejar modelos grandes (.pkl).  
Antes de clonar el repositorio, asegúrate de tener Git LFS instalado:

```bash
# Instalar Git LFS (una sola vez en el sistema)
git lfs install
```

### 2. Crea y activa un entorno virtual

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
setup

# macOS/Linux
source .venv/bin/activate
source setup.sh
```

```bash
streamlit run app.py
```
