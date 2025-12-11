import cv2
import numpy as np
import torch
import sys
import os

from torchvision.ops import nms  # igual que en tu API

# ============================================================
# CONFIGURACIÓN DEL MODELO (igual que en tu API)
# ============================================================

MODEL_PATH = "./plate_detector_ts_cpu.pt"   # se puede sobreescribir por CLI
DEVICE = torch.device("cpu")

IMAGE_SIZE = (416, 416)  # (H, W)

ANCHORS = torch.tensor([
    [47.977596, 30.171726],
    [95.66016 , 75.83388 ],
    [357.70685, 279.4956 ]
], dtype=torch.float32)

NUM_CLASSES = 1
CLASS_NAMES = ["LicensePlate"]

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def preprocess_image_cv2(image_path, input_size=IMAGE_SIZE):
    """
    Carga una imagen desde ruta con OpenCV, la convierte a RGB,
    la redimensiona a input_size y la devuelve como tensor [1,3,H,W] en [0,1].

    Devuelve:
      - img_bgr_original: imagen original BGR (para dibujar cajas)
      - img_tensor: tensor listo para el modelo
      - (orig_w, orig_h): tamaño original en píxeles
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")

    orig_h, orig_w = img_bgr.shape[:2]

    # Convertir BGR -> RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Redimensionar a tamaño de entrada (416x416)
    img_resized = cv2.resize(
        img_rgb,
        (input_size[1], input_size[0]),
        interpolation=cv2.INTER_LINEAR
    )

    # Normalizar a [0,1]
    img_normalized = img_resized.astype(np.float32) / 255.0

    # (H,W,C) -> (C,H,W)
    img_transposed = np.transpose(img_normalized, (2, 0, 1))

    # Añadir dimensión batch
    img_tensor = torch.from_numpy(img_transposed).unsqueeze(0)  # [1,3,H,W]

    return img_bgr, img_tensor.to(DEVICE), (orig_w, orig_h)

def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convierte cajas (cx, cy, w, h) -> (x1, y1, x2, y2)
    boxes: [N, 4]
    """
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 4))
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    return torch.stack([x1, y1, x2, y2], dim=1)

def decode_predictions(pred,
                       image_size=IMAGE_SIZE,
                       anchors=ANCHORS,
                       conf_threshold=0.25,
                       nms_iou_threshold=0.5):
    """
    Decodifica la salida del modelo tipo YOLO:
      pred: [1, A, H, W, 5+C]

    Devuelve lista de dicts:
      [ {x1, y1, x2, y2, score, class_id, class_name}, ... ]
    en coordenadas del espacio de entrada (416x416).
    """
    device = pred.device
    H_img, W_img = image_size

    B, A, H, W, out_ch = pred.shape
    num_classes = out_ch - 5

    # Separar predicciones
    tx = torch.sigmoid(pred[..., 0])
    ty = torch.sigmoid(pred[..., 1])
    tw = pred[..., 2]
    th = pred[..., 3]
    obj = torch.sigmoid(pred[..., 4])
    cls_logits = pred[..., 5:]  # [B,A,H,W,C]

    if num_classes > 0:
        cls_prob = torch.sigmoid(cls_logits)
        max_cls_prob, class_pred = cls_prob.max(dim=-1)  # [B,A,H,W]
    else:
        max_cls_prob = torch.ones_like(obj)
        class_pred = torch.zeros_like(obj, dtype=torch.long)

    scores = obj * max_cls_prob  # [B,A,H,W]

    # Grid
    grid_x = torch.arange(W, device=device).repeat(H, 1)
    grid_y = torch.arange(H, device=device).repeat(W, 1).t()
    gx = grid_x.view(1, 1, H, W)
    gy = grid_y.view(1, 1, H, W)

    # Anchors
    anchors = anchors.to(device).view(1, A, 1, 1, 2)

    # Decodificación a píxeles (cx,cy,w,h)
    bx = (tx + gx) * (W_img / W)
    by = (ty + gy) * (H_img / H)
    bw = anchors[..., 0] * torch.exp(tw)
    bh = anchors[..., 1] * torch.exp(th)

    # Solo B=1
    bx_b = bx[0].reshape(-1)
    by_b = by[0].reshape(-1)
    bw_b = bw[0].reshape(-1)
    bh_b = bh[0].reshape(-1)
    scores_b = scores[0].reshape(-1)
    class_b  = class_pred[0].reshape(-1)

    # Filtrar por confianza
    keep_conf = scores_b > conf_threshold
    if keep_conf.sum() == 0:
        return []

    bx_f = bx_b[keep_conf]
    by_f = by_b[keep_conf]
    bw_f = bw_b[keep_conf]
    bh_f = bh_b[keep_conf]
    scores_f = scores_b[keep_conf]
    class_f  = class_b[keep_conf]

    boxes_cxcywh = torch.stack([bx_f, by_f, bw_f, bh_f], dim=1)
    boxes_xyxy = xywh_to_xyxy(boxes_cxcywh)  # [N,4]

    if boxes_xyxy.numel() == 0:
        return []

    # NMS
    keep = nms(boxes_xyxy, scores_f, nms_iou_threshold)

    boxes_xyxy = boxes_xyxy[keep]
    scores_f   = scores_f[keep]
    class_f    = class_f[keep]

    detections = []
    for i in range(boxes_xyxy.shape[0]):
        x1, y1, x2, y2 = boxes_xyxy[i].tolist()
        score = float(scores_f[i].item())
        cls_id = int(class_f[i].item())
        cls_name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else f"class_{cls_id}"

        detections.append({
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "score": score,
            "class_id": cls_id,
            "class_name": cls_name
        })

    return detections

# ============================================================
# LÓGICA PRINCIPAL
# ============================================================

def main(model_path, image_path,
         conf_threshold=0.25,
         nms_iou_threshold=0.5,
         output_path="output_detection.jpg"):
    # 1. Verificar rutas
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Imagen no encontrada: {image_path}")

    # 2. Cargar modelo TorchScript
    print(f"Cargando modelo TorchScript desde: {model_path}")
    model = torch.jit.load(model_path, map_location=DEVICE)
    model.eval()

    # 3. Preprocesar imagen
    img_bgr, input_tensor, (orig_w, orig_h) = preprocess_image_cv2(
        image_path,
        input_size=IMAGE_SIZE
    )

    # 4. Inferencia
    with torch.no_grad():
        preds = model(input_tensor)  # esperado [1,A,H,W,5+C]

    # 5. Decodificar detecciones en espacio 416x416
    dets = decode_predictions(
        preds,
        image_size=IMAGE_SIZE,
        anchors=ANCHORS,
        conf_threshold=conf_threshold,
        nms_iou_threshold=nms_iou_threshold
    )

    if len(dets) == 0:
        print("\n=== Sin detecciones sobre el umbral de confianza ===")
        return

    # 6. Reescalar cajas al tamaño original de la imagen
    scale_x = orig_w / IMAGE_SIZE[1]
    scale_y = orig_h / IMAGE_SIZE[0]

    print("\n=== Detecciones ===")
    for i, d in enumerate(dets):
        x1_416 = d["x1"]
        y1_416 = d["y1"]
        x2_416 = d["x2"]
        y2_416 = d["y2"]

        # Escalar a coordenadas originales
        x1 = int(x1_416 * scale_x)
        y1 = int(y1_416 * scale_y)
        x2 = int(x2_416 * scale_x)
        y2 = int(y2_416 * scale_y)

        score = d["score"]
        cls_id = d["class_id"]
        cls_name = d["class_name"]

        print(f"Det {i}: {cls_name} ({cls_id}) "
              f"score={score:.3f} "
              f"box416=({x1_416:.1f},{y1_416:.1f},{x2_416:.1f},{y2_416:.1f}) "
              f"boxOrig=({x1},{y1},{x2},{y2})")

        # 7. Dibujar caja en la imagen original
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{cls_name} {score:.2f}"
        cv2.putText(img_bgr, label, (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # 8. Guardar resultado
    cv2.imwrite(output_path, img_bgr)
    print(f"\nImagen con detecciones guardada en: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python inference_plate_detector_ts.py "
              "<ruta_modelo.pt> <ruta_imagen.jpg> [conf_threshold] [nms_iou]")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]

    conf_th = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
    nms_iou = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5

    main(model_path, image_path,
         conf_threshold=conf_th,
         nms_iou_threshold=nms_iou,
         output_path="output_detection.jpg")
