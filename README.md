# Trabajo 3 — Aplicaciones en Sistemas de Recomendación e Imágenes

---

##  Descripción del proyecto

Sistema inteligente que integra **predicción de demanda de transporte**, **clasificación de conducción distractiva mediante imágenes**, y **recomendación personalizada de destinos de viaje**, usando aprendizaje profundo.  

Se construyó una herramienta web interactiva para probar cada modelo y visualizar los resultados.

---

## Enlaces útiles

- [Video demostrativo](https://drive.google.com/file/d/1bsGW4kEb97XDH3IvKGYpgdKuUPilWsaL/view?usp=drive_link
)  
- [Informe técnico en QuartoPub](https://jochoara.quarto.pub/trabajo-3/)  
- [Sitio web (Streamlit)](https://trabajo-3-aplicaciones-en-sistemas-de-recomendacion-e-imagenes.streamlit.app/)


---

##  Estructura del repositorio

```plaintext
Trabajo-3-aplicaciones-en-sistemas-de-recomendacion-e-imagenes/
├── Informe_tecnico/    -         
├── Notebooks/          -
├── Recursos/           -
├── web_page/           -
│   ├── Home.py
│   └── pages/
│       ├── 1Prediccion_de_demanda.py
│       ├── 2Clasificación_de_conducción.py
│       ├── 3Recomendación_de_destinos.py
│       └── Sobre_Nosotros.py


##  Requisitos

- Python 3.8 o superior
- `pip` o entorno virtual como `.venv`

---

## Pasos para ejecutar el proyecto

###  Requisitos adicionales

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
