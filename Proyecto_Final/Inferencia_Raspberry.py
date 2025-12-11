import cv2
import numpy as np
import torch
import sys
import os

def preprocess_image(image_path, input_size=(416, 416)):
    """
    Carga una imagen, la convierte a RGB, la redimensiona y la prepara como tensor.
    Además devuelve la imagen redimensionada en BGR para dibujar las detecciones.
    """
    # Cargar imagen con OpenCV (BGR por defecto)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")
    
    # Redimensionar al tamaño de entrada del modelo
    img_resized_bgr = cv2.resize(img_bgr, input_size, interpolation=cv2.INTER_LINEAR)

    # Convertir a RGB para el modelo
    img_rgb = cv2.cvtColor(img_resized_bgr, cv2.COLOR_BGR2RGB)

    # Normalizar a [0, 1]
    img_normalized = img_rgb.astype(np.float32) / 255.0

    # Transponer de (H, W, C) a (C, H, W)
    img_transposed = np.transpose(img_normalized, (2, 0, 1))

    # Añadir batch dimension: (1, C, H, W)
    img_tensor = torch.from_numpy(img_transposed).unsqueeze(0)

    return img_tensor, img_resized_bgr


def main(model_path, image_path, input_size=(416, 416), score_thresh=0.3):
    # Verificar que los archivos existan
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Imagen no encontrada: {image_path}")

    # Cargar el modelo TorchScript
    print(f"Cargando modelo TorchScript desde: {model_path}")
    model = torch.jit.load(model_path)
    model.eval()  # modo evaluación

    # Preprocesar la imagen
    print(f"Procesando imagen: {image_path}")
    input_tensor, img_vis = preprocess_image(image_path, input_size=input_size)

    # Inferencia
    print("Ejecutando inferencia...")
    with torch.no_grad():
        output = model(input_tensor)

    # Esperamos que la salida sea un dict con la clave "detections"
    if not isinstance(output, dict) or "detections" not in output:
        raise ValueError(
            f"Salida del modelo no es del formato esperado. "
            f"Tipo: {type(output)}, claves: {getattr(output, 'keys', lambda: [])()}"
        )

    detections = output["detections"]

    print("\n=== Detecciones crudas ===")
    print(detections)

    # Filtrar por score
    filtered_dets = []
    for det in detections:
        score = float(det["score"])
        if score < score_thresh:
            continue

        x1 = int(det["x1"])
        y1 = int(det["y1"])
        x2 = int(det["x2"])
        y2 = int(det["y2"])
        class_id = int(det.get("class_id", -1))
        class_name = det.get("class_name", f"Clase_{class_id}")

        filtered_dets.append(
            {
                "bbox": (x1, y1, x2, y2),
                "score": score,
                "class_id": class_id,
                "class_name": class_name,
            }
        )

    # Imprimir resultados
    print("\n=== Detecciones (filtradas por score) ===")
    if not filtered_dets:
        print(f"No se detectó nada con score >= {score_thresh}")
    else:
        for i, det in enumerate(filtered_dets):
            x1, y1, x2, y2 = det["bbox"]
            print(
                f"[{i}] {det['class_name']} "
                f"| score={det['score']:.3f} "
                f"| bbox=({x1}, {y1}, {x2}, {y2})"
            )

    # Dibujar detecciones en la imagen
    for det in filtered_dets:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['class_name']} {det['score']:.2f}"

        # Caja
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Fondo para el texto
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            img_vis,
            (x1, y1 - th - baseline),
            (x1 + tw, y1),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            img_vis,
            label,
            (x1, y1 - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    # Guardar imagen con detecciones
    out_path = "detecciones_resultado.jpg"
    cv2.imwrite(out_path, img_vis)
    print(f"\nImagen con detecciones guardada en: {out_path}")

    # Mostrar en ventana (si estás en PC, no en servidor/headless)
    try:
        cv2.imshow("Detecciones", img_vis)
        print("Presiona cualquier tecla en la ventana de imagen para cerrar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        print("No se pudo abrir ventana gráfica (posible entorno sin GUI).")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso:")
        print("  python inference_detector_torchscript.py <ruta_modelo.pt> <ruta_imagen.jpg>")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]

    main(model_path, image_path, input_size=(416, 416), score_thresh=0.3)
