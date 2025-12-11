# Detector de Placas Vehiculares – Procesamiento Digital de Imágenes

**Autores:** Felipe Idárraga Quintero y  Julian Felipe Gutiérrez Ramírez

**Nombre de la Práctica:** Proyecto Final

**Curso:** Desarrollo de Sistemas IoT

**Departamento:** Departamento de Ingeniería Electrica, Electronica y Computacion

---

Este proyecto implementa un sistema completo de detección de placas vehiculares utilizando técnicas modernas de Deep Learning. Se desarrolla un modelo entrenado desde cero con un conjunto de datos de Roboflow, se evalúa su desempeño, se convierte a TorchScript para despliegue en dispositivos embebidos (como Raspberry Pi) y finalmente se integra con una API en un HuggingFace Space.

---

## Estructura del repositorio

```text
📂 Procesamiento_Digital__de_Imagenes/
│── 📄 README.md
│── 📓 deteccion-placas-v1.ipynb        # Notebook principal del proyecto
│── 📓 deployment_hf_space.ipynb        # Notebook de despliegue en HuggingFace
│── 📂 models/
│      ├── best_plate_detector.pth      # Pesos nativos de PyTorch
│      └── plate_detector_ts_cpu.pt     # Versión TorchScript optimizada
│── 📂 raspberry/
│      └── inferencia_raspberry.py      # Script de inferencia para Raspberry Pi
│── 📂 utils/
│      └── funciones_preprocesamiento.py (si aplica)
```

## 1. Descripción general del proyecto

El objetivo del proyecto es desarrollar un sistema de detección automática de placas vehiculares, partiendo desde el entrenamiento del modelo hasta su despliegue en entornos reales.

Las fases principales incluidas en este repositorio son:

**1. Carga y preparación del dataset** desde Roboflow.
**2. Diseño del modelo de detección** basado en PyTorch.
**3. Implementación de la función de pérdida personalizada.**
**4. Entrenamiento y validación del modelo detector.**
**5. Evaluación mediante métricas de desempeño.**
**6. Exportación a formato TorchScript** para despliegue eficiente
**7. Comparación entre modelo PyTorch original (.pth) y TorchScript (.pt)**
**8. Implementación de un script de inferencia en Raspberry Pi.**
**9. Despliegue del modelo en un HuggingFace Space mediante API.**

## 2. Dataset

El dataset fue obtenido desde **Roboflow** llamado ****, incluyendo imágenes anotadas de placas vehiculares en distintos entornos, con las siguientes características:

- Formato YOLO
- División train / valid / test
- Augmentaciones aplicadas:
  - Rotación
  - Brillo / contraste
  - Scale jitter
  - Flip horizontal

El dataset se carga directamente en el notebook mediante la API de Roboflow:

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("workspace").project("placas")
dataset = project.version(1).download("yolov8")
```

## 3. Modelo de detección

Se implementó un modelo de detección propio utilizando PyTorch, definiendo:

- Backbone convolucional
- Cabecera de predicción con anclas optimizadas
- Función de pérdida combinada (Obj + BBox + Class)
- Post-procesamiento con Non-Max Suppression (NMS)

El notebook incluye:

- Arquitectura del modelo
- Entrenamiento
- Validación
- Guardado de pesos .pth `.pth`
