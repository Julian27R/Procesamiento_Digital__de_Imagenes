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
📂 deteccion-placas/
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

