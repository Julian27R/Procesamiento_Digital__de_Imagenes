# Detector de Placas Vehiculares – Procesamiento Digital de Imágenes

**Autores:** Felipe Idárraga Quintero y  Julian Felipe Gutiérrez Ramírez

**Nombre de la Práctica:** Proyecto Final

**Curso:** Desarrollo de Sistemas IoT

**Departamento:** Departamento de Ingeniería Electrica, Electronica y Computacion

Este proyecto implementa un sistema completo de detección de placas vehiculares utilizando técnicas modernas de Deep Learning. Se desarrolla un modelo entrenado desde cero con un conjunto de datos de Roboflow, se evalúa su desempeño, se convierte a TorchScript para despliegue en dispositivos embebidos (como Raspberry Pi) y finalmente se integra con una API en un HuggingFace Space.

---

```text
## Estructura del repositorio

📂 Proyecto_Final/
├── README.md
├── deteccion-placas-v1.ipynb                 # Notebook principal: dataset, modelo, entrenamiento, métricas y exportación
│
├── 📂 Hugging_Face/
│   ├──  HFS_Proyecto_Final.ipynb              # Notebook de despliegue en HuggingFace Space (Colab)
│   └──  Inferencia_HuggingFace.py             # Cliente Python: consume /predict y dibuja detecciones
│
├── 📂 Pesos_del_Modelo/
│   ├──  best_plate_detector.pth               # Pesos del modelo en PyTorch (state_dict)
│   └──  plate_detector_ts_cpu.pt              # Modelo exportado a TorchScript (CPU) para despliegue
│
└── 📂 raspberry/
    └──  inferencia_raspberry.py               # Inferencia en Raspberry Pi (TorchScript + OpenCV + NMS)
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

## 3. Modelo de Detección

Se implementó un modelo de detección propio en PyTorch basado en una versión ligera de **YOLOv3 Mini mejorado**, diseñado específicamente para la detección de placas vehiculares. El modelo integra los siguientes componentes:

- **Backbone convolucional con bloques residuales**, encargado de extraer características robustas a partir de la imagen.
- **Módulo SPP (Spatial Pyramid Pooling)**, que aporta contexto multi-escala utilizando max-pooling con distintos tamaños de ventana.
- **Cabeza de detección con anchors optimizados**, donde cada celda de la grilla predice offsets de cajas, objectness y probabilidades de clase, empleando los anchors generados por k-means.
- **Función `decode` estilo YOLO**, responsable de transformar las predicciones de la red en coordenadas reales (píxeles), scores y clases antes del post-procesamiento.
- **Función de pérdida combinada** (Obj + BBox + Class) que entrena simultáneamente la localización, confianza y clasificación.

El notebook incluye:

- La **arquitectura completa del modelo** (backbone, SPP y head de detección).
- El **proceso de entrenamiento**.
- La **validación y métricas de desempeño**.
- El **guardado de pesos** del modelo en formato `.pth` para uso posterior en inferencia y exportación a TorchScript.

## 4. Función de pérdida para detección (`yolo_plate_loss`)

Se implementó `yolo_plate_loss`, una pérdida inspirada en **YOLO** y adaptada al modelo `ImprovedPlateDetector`. A partir de las cajas reales normalizadas, la función:

- Asigna cada placa a una **celda del grid y al mejor anchor** según IoU.
- Construye los **targets** para posición `(tx, ty)`, tamaño `(tw, th)`, confianza (`t_obj`) y clase (`t_cls`).
- Calcula:
  - **Pérdida de coordenadas** (posición y tamaño de la caja).
  - **Pérdida de confianza** para celdas con objeto y sin objeto.
  - **Pérdida de clasificación** sobre las celdas que contienen placas.

La pérdida total combina estos términos con factores de ponderación (`lambda_coord`, `lambda_obj`, `lambda_noobj`) y devuelve:

- Un **escalar `total_loss`** usado para el entrenamiento.
- Un **diccionario `loss_dict`** con el desglose: `coord_loss`, `conf_loss` y `cls_loss`.

## 5. Métricas de desempeño

Durante el proceso de evaluación del modelo se calcularon:

- **Pérdida promedio** en validación (loss total y desglosada por componentes).
- **AP@0.5 (AP50)** como métrica principal de detección de placas.
- **Precisión, recall y F1 por clase**, junto con conteo de TP, FP y FN.

Además, se generan visualizaciones de:

- **Evolución de la pérdida** (entrenamiento vs. validación) a lo largo de las épocas.
- **Evolución del AP@0.5** (entrenamiento vs. validación) en porcentaje.

## 6. Conversión a TorchScript

Para facilitar el despliegue del detector en entornos fuera de PyTorch “completo” (por ejemplo, inferencia en **CPU** y dispositivos embebidos), el modelo se exportó a **TorchScript**. En el notebook se realiza la conversión reconstruyendo el `ImprovedPlateDetector` con **los mismos parámetros y anchors del entrenamiento**, cargando los pesos `.pth` y generando una versión TorchScript mediante **trazado** (`torch.jit.trace`) con una entrada dummy.

Se exportan dos variantes:

- **`plate_detector_ts.pt`**: trazado usando el dispositivo disponible (GPU si está disponible).
- **`plate_detector_ts_cpu.pt`**: trazado y guardado específicamente para **CPU**, recomendado para despliegue (p. ej., Raspberry Pi).

Ejemplo (como se hace en el notebook):

```python
# Reconstruir el modelo con los mismos parámetros del entrenamiento
loaded_model = ImprovedPlateDetector(
    num_classes=1,
    image_size=(416, 416),
    num_anchors=3,
    anchors=anchors_kmeans.tolist()
).to(device)

# Cargar pesos y exportar a TorchScript por trace
loaded_model.load_state_dict(torch.load("best_plate_detector.pth", map_location=device))
loaded_model.eval()

dummy_input = torch.randn(1, 3, 416, 416, device=device)

with torch.no_grad():
    ts_model = torch.jit.trace(loaded_model, dummy_input)

ts_model.save("plate_detector_ts_cpu.pt")
print(" Modelo exportado a TorchScript (CPU)")
```

## 7. Comparación entre PyTorch (.pth) y TorchScript (.pt)

Para analizar el impacto de la exportación a TorchScript, se compararon ambas versiones del detector:

- **Modelo PyTorch (eager):** `best_plate_detector.pth`
- **Modelo TorchScript (CPU):** `plate_detector_ts_cpu.pt`

### Aspectos evaluados

1. **Tamaño en disco (MB)**  
   Se comparó el tamaño del archivo `.pth` frente al `.pt` para verificar si existía alguna reducción tras la conversión.

2. **Velocidad de inferencia (benchmark)**  
   Se midió el tiempo promedio de inferencia (forward) de ambos modelos usando la misma entrada dummy, incluyendo *warmup* y múltiples ejecuciones para estimar media y desviación estándar.

3. **Consistencia numérica (opcional / si aplica en el notebook)**  
   Cuando se realiza, se comparan las salidas de ambos modelos sobre las mismas entradas calculando:
   - `max |Δ|`: diferencia absoluta máxima
   - `mean |Δ|`: diferencia absoluta promedio

###  Resultados observados (ejemplo)

- **Tamaño:** no se evidenció reducción significativa (los archivos quedaron con tamaños similares).
- **Tiempo de inferencia:** el rendimiento fue comparable entre ambas versiones (speedup cercano a `1.0x`).

```text
 Tamaño best_plate_detector.pth      : 38.08 MB
 Tamaño plate_detector_ts_cpu.pt     : 38.27 MB

PyTorch (eager):      299.799 ± 4.276 ms
TorchScript (.pt):    303.697 ± 8.123 ms
Speedup aproximado:      0.99x
```
## 8. Inferencia en Raspberry Pi

El repositorio incluye un script de inferencia para ejecutar el detector de placas en **CPU** (ideal para Raspberry Pi) usando el modelo exportado a **TorchScript** (`.pt`). El script carga el modelo, procesa una imagen desde ruta y genera una salida con las detecciones dibujadas.

### Archivo incluido

`raspberry/inference_plate_detector_ts.py` *(ajusta el nombre/ruta si en tu repo es diferente)*

### Funcionalidades del script

- **Carga** un modelo TorchScript (`torch.jit.load`) desde una ruta (por defecto `plate_detector_ts_cpu.pt`).
- **Preprocesa** la imagen con OpenCV:
  - BGR → RGB
  - redimensiona a `416×416`
  - normaliza a `[0,1]`
  - convierte a tensor `[1,3,H,W]`
- **Realiza inferencia** en CPU.
- **Decodifica** la salida tipo YOLO (grid + anchors) para obtener cajas `(x1,y1,x2,y2)`, score y clase.
- **Filtra por confianza** y aplica **Non-Max Suppression (NMS)**.
- **Reescala** las cajas al tamaño original de la imagen.
- **Dibuja** las detecciones y **guarda** el resultado como `output_detection.jpg`.

### Ejecución

```bash
python3 inference_plate_detector_ts.py plate_detector_ts_cpu.pt ruta/imagen.jpg 0.25 0.5
```

**Parámetros:**
- `0.25`: umbral de confianza (*conf_threshold*).
- `0.5`: umbral IoU para NMS (*nms_iou_threshold*).

### Requisitos en Raspberry Pi

Instalar dependencias básicas:

```bash
sudo apt-get update
sudo apt-get install -y python3-opencv
pip3 install numpy
```

**Nota:** para ejecutar un modelo TorchScript .pt también necesitas PyTorch instalado en la Raspberry Pi (porque torch.jit.load depende de torch). La instalación en ARM varía según el modelo y el sistema operativo, pero TorchScript permite desplegar el modelo sin depender del código fuente original y facilita la ejecución en CPU.

## 9. Despliegue en HuggingFace Space

El proyecto incluye un notebook de despliegue en Google Colab para publicar el detector en **HuggingFace Spaces** mediante una **API en FastAPI**.

📄 `HFS_Proyecto_Final.ipynb` *

En el notebook se realiza el siguiente flujo:

1. **Implementación de la API (FastAPI)**  
   Se construye una aplicación que:
   - Carga el modelo de detección en formato **TorchScript** (`torch.jit.load`).
   - Define los esquemas de entrada/salida con Pydantic.
   - Incluye utilidades de **preprocesamiento** (imagen → tensor 416×416) y **decodificación** de predicciones (YOLO + anchors + NMS).
   - Expone endpoints:
     - `/` (ruta base)
     - `/health` (verificación de estado)
     - `/predict` (inferencia)

2. **Endpoint `/predict`**  
   Recibe una imagen codificada en **Base64** y retorna un JSON con las detecciones:

   - Coordenadas de caja: `(x1, y1, x2, y2)` *(en el sistema de 416×416)*
   - Puntaje de confianza: `score`
   - Identificador y nombre de clase: `class_id`, `class_name` (ej. `"LicensePlate"`)

3. **Dockerfile para el Space**  
   Se define un `Dockerfile` para el despliegue que:
   - Usa **Python 3.10**
   - Instala dependencias desde `requirements.txt`
   - Copia el código de la app al contenedor
   - Expone el puerto `7860`
   - Inicia el servidor con **Uvicorn** (`app:app`)

4. **Cliente Python para consumir la API**  
   Se incluye un script de ejemplo que:
   - Lee una imagen desde disco
   - La codifica en Base64
   - Realiza una petición POST al endpoint `/predict`
   - Dibuja y opcionalmente guarda las detecciones sobre la imagen

### Ejemplo de consumo desde un cliente Python

```python
import base64
import requests

url = "https://<tu-space>.hf.space/predict"

with open("placa_test.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {"image_base64": img_b64}
resp = requests.post(url, json=payload)

print(resp.json())
```

## 10. Flujo de uso recomendado

Para reproducir el proyecto de principio a fin, se recomienda seguir el siguiente flujo de trabajo:

### 1️⃣ Abrir el notebook principal

`deteccion-placas-v1.ipynb`

### 2️⃣ Ejecutar las secciones en orden

1. **Carga del dataset desde Roboflow.**
2. **Definición del modelo** y su **función de pérdida personalizada**.
3. **Entrenamiento** del modelo.
4. **Validación** y cálculo de las **métricas de desempeño**.
5. **Guardado de pesos** en formato `.pth`.
6. **Conversión del modelo a TorchScript** (`.pt`).
7. **Comparación entre las salidas de PyTorch y TorchScript**.

---

### 3️⃣ Probar el despliegue embebido (Raspberry Pi)

1. Copiar a la Raspberry Pi:
   - `plate_detector_ts_cpu.pt`
   - `inferencia_raspberry.py`
2. Ejecutar pruebas utilizando imágenes reales para validar el modelo.

---

### 4️⃣ Probar el despliegue en la nube (HuggingFace Space)

1. Abrir el notebook de despliegue en HuggingFace.
2. Subir el modelo TorchScript.
3. Verificar el funcionamiento de la API usando solicitudes de inferencia.
4. Comprobar que retorna detecciones correctas (coordenadas, score, clase).

---

Este flujo garantiza que el proyecto pueda reproducirse completamente desde el entrenamiento hasta el despliegue final, tanto en dispositivos locales como en la nube.

## 11. Requisitos

### 11.1. Entorno de entrenamiento (Kaggle / PC)

Requisitos mínimos:

- **Python 3.9+**
- Bibliotecas principales:
  - `torch`
  - `torchvision`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `tqdm`
  - `roboflow`
  - `opencv-python`
  - `fastapi` (para pruebas de API locales)
  - `uvicorn` (servidor ASGI)

#### Instalación típica

```bash
pip install torch torchvision roboflow opencv-python matplotlib seaborn tqdm fastapi uvicorn
```

### 11.2. Entorno en Raspberry Pi

**Requisitos mínimos:**

- **Python 3**

- **Bibliotecas necesarias:**
  - `numpy`
  - `opencv-python`

- **Compatibilidad con PyTorch / TorchScript** para arquitectura ARM.  
  *(Se puede instalar PyTorch para ARM o usar únicamente el runtime de TorchScript).*

#### Instalación de dependencias

```bash
pip3 install numpy opencv-python
```
