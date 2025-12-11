#!/usr/bin/env python
"""
Script de inferencia para el modelo de detección de placas desplegado en
HuggingFace Space (FastAPI).

Uso desde terminal:

    python infer_plate_api.py --image ./Carros3.png

Opcionalmente puedes cambiar la URL del endpoint:

    python infer_plate_api.py --image ./Carros3.png \
        --api-url https://JulianGR27-DetectionPlacas.hf.space/predict

Y/o guardar la imagen resultante:

    python infer_plate_api.py --image ./Carros3.png --save ./out.png
"""

import argparse
import base64
import json
import os

import requests
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURACIÓN BÁSICA
# ============================================================

# URL de tu Space (endpoint /predict del FastAPI)
API_URL_DEFAULT = "https://JulianGR27-DetectionPlacas.hf.space/predict"

# Tamaño de entrada del modelo (el mismo que usaste: 416x416)
MODEL_IMAGE_SIZE = (416, 416)  # (ancho, alto)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def encode_image_to_base64(image_path: str) -> str:
    """
    Lee la imagen desde disco y la convierte a un string base64 (UTF-8),
    listo para ser enviado al endpoint FastAPI.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"No se encontró la imagen: {image_path}")

    with open(image_path, "rb") as f:
        img_bytes = f.read()

    # Codificar a base64 y luego a string UTF-8
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_b64


def call_plate_api(image_path: str,
                   api_url: str = API_URL_DEFAULT,
                   timeout: int = 60) -> dict:
    """
    Prepara el payload con la imagen en base64 y realiza la petición POST
    al endpoint /predict de tu Space. Devuelve el JSON como diccionario.

    Estructura esperada de respuesta:
        {
          "detections": [
            {
              "x1": ..., "y1": ..., "x2": ..., "y2": ...,
              "score": ..., "class_id": ..., "class_name": ...
            },
            ...
          ]
        }
    """
    # 1. Convertir imagen a base64
    img_b64 = encode_image_to_base64(image_path)
    payload = {"image_base64": img_b64}

    # 2. Enviar petición POST
    res = requests.post(api_url, json=payload, timeout=timeout)

    # Lanzar excepción si hubo error HTTP (4xx, 5xx)
    res.raise_for_status()

    # 3. Convertir respuesta a dict
    data = res.json()
    return data


def draw_detections(
    image_path: str,
    detections: list,
    save_path: str | None = None,
    show: bool = True,
    model_image_size: tuple[int, int] = MODEL_IMAGE_SIZE,
):
    """
    Dibuja las cajas de detección sobre la imagen.

    IMPORTANTE:
    - La API procesa la imagen redimensionándola a 416x416.
    - Las coordenadas devueltas (x1, y1, x2, y2) están en ese sistema.
    - Por eso aquí redimensionamos también la imagen a 416x416 antes de dibujar.

    Si quieres dibujar en la resolución original, habría que reescalar las
    coordenadas manualmente.
    """
    # 1. Cargar imagen original
    img = Image.open(image_path).convert("RGB")

    # 2. Redimensionar a tamaño del modelo (416x416)
    img = img.resize(model_image_size)  # mismo proceso que en el servidor

    draw = ImageDraw.Draw(img)

    # 3. Dibujar cada detección
    for det in detections:
        x1 = det["x1"]
        y1 = det["y1"]
        x2 = det["x2"]
        y2 = det["y2"]
        score = det["score"]
        class_name = det.get("class_name", "obj")

        # Etiqueta con clase y score
        label = f"{class_name} {score:.2f}"

        # Rectángulo en rojo
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        # Texto ligeramente encima de la caja
        draw.text((x1, max(y1 - 10, 0)), label, fill="red")

    # 4. Guardar si se especifica ruta
    if save_path is not None:
        img.save(save_path)
        print(f"✅ Imagen con detecciones guardada en: {save_path}")

    # 5. Mostrar en pantalla si show=True
    if show:
        plt.imshow(img)
        plt.axis("off")
        plt.title("Detecciones de placa")
        plt.show()


# ============================================================
# FUNCIÓN PRINCIPAL (CLI)
# ============================================================

def main():
    # Parser de argumentos para poder pasar opciones desde la terminal
    parser = argparse.ArgumentParser(
        description="Script de inferencia para PlateDetectorAPI vía HuggingFace Space."
    )
    parser.add_argument(
        "--image",
        "-i",
        required=True,
        help="Ruta de la imagen a procesar (ejemplo: ./Carros3.png)",
    )
    parser.add_argument(
        "--api-url",
        default=API_URL_DEFAULT,
        help=f"URL del endpoint /predict de tu Space. Por defecto: {API_URL_DEFAULT}",
    )
    parser.add_argument(
        "--save",
        "-o",
        default=None,
        help="Ruta opcional para guardar la imagen con detecciones (ej: ./resultado.png)",
    )

    args = parser.parse_args()

    image_path = args.image
    api_url = args.api_url
    save_path = args.save

    print(f"🖼  Imagen: {image_path}")
    print(f"🌐 Endpoint: {api_url}")

    # 1. Llamar a la API
    try:
        data = call_plate_api(image_path, api_url=api_url)
    except requests.HTTPError as e:
        print("❌ Error HTTP al llamar a la API:")
        print(e)
        return
    except Exception as e:
        print("❌ Error inesperado al llamar a la API:")
        print(e)
        return

    # 2. Extraer detecciones
    detections = data.get("detections", [])
    print(f"📦 Detecciones recibidas: {len(detections)}")

    # Imprimir breve resumen en consola
    for idx, det in enumerate(detections):
        print(
            f"[{idx}] "
            f"Clase: {det['class_id']} ({det['class_name']}) | "
            f"Score: {det['score']:.3f} | "
            f"Caja: ({det['x1']:.1f}, {det['y1']:.1f}, "
            f"{det['x2']:.1f}, {det['y2']:.1f})"
        )

    # 3. Dibujar y mostrar resultados si hay al menos una detección
    if detections:
        draw_detections(
            image_path=image_path,
            detections=detections,
            save_path=save_path,
            show=True,
            model_image_size=MODEL_IMAGE_SIZE,
        )
    else:
        print("ℹ️ No se detectaron placas en la imagen (o score < umbral en el backend).")


if __name__ == "__main__":
    main()
