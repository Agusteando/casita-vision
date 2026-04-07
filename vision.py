import io
import hashlib
import logging
from typing import Dict, Any, Tuple
import numpy as np
import cv2
from PIL import Image
from rembg import remove, new_session

logger = logging.getLogger(__name__)

# NOTA IMPORTANTE: Usamos 'u2netp' (la versión ligera) en lugar de 'u2net'
# para prevenir caídas y reinicios del servidor por falta de memoria RAM (OOM) en VPS pequeños.
try:
    rembg_session = new_session("u2netp")
except Exception as e:
    logger.error(f"Error al inicializar el modelo de Rembg: {e}")
    rembg_session = None

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def hash_image(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()

def process_image(image_bytes: bytes) -> Tuple[Dict[str, Any], bytes]:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Datos de imagen no válidos. Asegúrese de que el archivo sea una imagen válida.")

    img_height, img_width = img.shape[:2]

    result = {
        "faceDetected": False,
        "faceConfidence": 0.0,
        "faceBox": None,
        "cropBox": None,
        "eyesDetected": False,
        "eyeConfidence": 0.0,
        "eyeBoxes": None,
        "backgroundRemoved": False,
        "debug": {
            "width": img_width,
            "height": img_height
        }
    }

    # 1. REMOCIÓN DE FONDO PRIMERO
    mask_bytes = None
    mask_cv2 = None
    if rembg_session is not None:
        try:
            logger.info("Iniciando remoción de fondo...")
            image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            # Forzamos una resolución máxima para evitar picos excesivos de RAM
            image_pil.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
            mask_img = remove(image_pil, session=rembg_session, only_mask=True)
            img_byte_arr = io.BytesIO()
            mask_img.save(img_byte_arr, format='PNG')
            mask_bytes = img_byte_arr.getvalue()
            result["backgroundRemoved"] = True
            logger.info("Fondo removido exitosamente.")
            
            # Decodificamos la máscara generada a un array de OpenCV (escala de grises)
            mask_nparr = np.frombuffer(mask_bytes, np.uint8)
            mask_cv2_decoded = cv2.imdecode(mask_nparr, cv2.IMREAD_GRAYSCALE)
            
            if mask_cv2_decoded is not None:
                # La máscara puede ser más pequeña por el thumbnail, la redimensionamos al tamaño original
                mask_cv2 = cv2.resize(mask_cv2_decoded, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
                
        except Exception as e:
            logger.error(f"Error al remover fondo: {e}")
            result["debug"]["bg_removal_error"] = str(e)
    else:
        result["debug"]["bg_removal_error"] = "Modelo rembg no disponible"

    # 2. DETECCIÓN FACIAL (Validada mediante la máscara)
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # MAGIA AQUI: Si tenemos la máscara de fondo, enmascaramos la imagen original.
        # Esto convierte todo el fondo (paredes, árboles, etc) en negro puro.
        # Haar Cascades NUNCA detectará una cara falsa en un fondo negro sólido.
        if mask_cv2 is not None:
            _, binary_mask = cv2.threshold(mask_cv2, 127, 255, cv2.THRESH_BINARY)
            gray = cv2.bitwise_and(gray, gray, mask=binary_mask)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        if len(faces) > 0:
            dominant_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = dominant_face
            
            result["faceDetected"] = True
            result["faceConfidence"] = 0.95 
            
            x_min, x_max = x / img_width, (x + w) / img_width
            y_min, y_max = y / img_height, (y + h) / img_height
            
            result["faceBox"] = {"xMin": x_min, "yMin": y_min, "xMax": x_max, "yMax": y_max}

            pad_w, pad_h = w * 0.5, h * 0.5
            result["cropBox"] = {
                "xMin": max(0.0, (x - pad_w) / img_width),
                "yMin": max(0.0, (y - pad_h) / img_height),
                "xMax": min(1.0, (x + w + pad_w) / img_width),
                "yMax": min(1.0, (y + h + pad_h) / img_height)
            }

            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15))

            if len(eyes) >= 2:
                result["eyesDetected"] = True
                result["eyeConfidence"] = 0.90
                
                sorted_eyes = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[:2]
                eye_boxes = []
                for (ex, ey, ew, eh) in sorted_eyes:
                    global_ex = x + ex
                    global_ey = y + ey
                    eye_boxes.append({
                        "xMin": global_ex / img_width,
                        "yMin": global_ey / img_height,
                        "xMax": (global_ex + ew) / img_width,
                        "yMax": (global_ey + eh) / img_height
                    })
                
                eye_boxes.sort(key=lambda box: box["xMin"])
                result["eyeBoxes"] = {
                    "leftEye": eye_boxes[0],
                    "rightEye": eye_boxes[1]
                }
            
            result["debug"]["pipeline"] = "opencv_cascade_success_masked"
            
        else:
            # SMART FALLBACK: No se detectó ninguna cara (o la falsa fue borrada por la máscara).
            # En lugar de fallar, usamos el "Bounding Box" de la máscara de fondo para encuadrar
            # el cuerpo del sujeto. Así el frontend siempre muestra un avatar bien centrado.
            if mask_cv2 is not None:
                coords = cv2.findNonZero(mask_cv2)
                if coords is not None:
                    mx, my, mw, mh = cv2.boundingRect(coords)
                    pad_w, pad_h = mw * 0.1, mh * 0.1 # 10% de padding para que no choque con los bordes
                    
                    result["cropBox"] = {
                        "xMin": max(0.0, (mx - pad_w) / img_width),
                        "yMin": max(0.0, (my - pad_h) / img_height),
                        "xMax": min(1.0, (mx + mw + pad_w) / img_width),
                        "yMax": min(1.0, (my + mh + pad_h) / img_height)
                    }
                    result["debug"]["pipeline"] = "opencv_cascade_failed_fallback_to_body_mask"

    except Exception as e:
        logger.error(f"Error en detección facial: {e}")
        result["debug"]["detection_error"] = str(e)

    return result, mask_bytes