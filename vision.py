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
    orig_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if orig_img is None:
        raise ValueError("Datos de imagen no válidos. Asegúrese de que el archivo sea una imagen válida.")

    orig_h, orig_w = orig_img.shape[:2]

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
            "original_width": orig_w,
            "original_height": orig_h
        }
    }

    # =====================================================================
    # ESTANDARIZACIÓN DE IMAGEN PARA OPENCV
    # Redimensionamos imágenes masivas a un estándar (máx 800px).
    # Esto soluciona de raíz las "alucinaciones" del Haar Cascade, ya que
    # evita que texturas diminutas (50px) en fotos gigantes (3000px) sean 
    # erróneamente identificadas como caras.
    # =====================================================================
    MAX_DETECTION_DIM = 800
    scale = 1.0
    
    if max(orig_w, orig_h) > MAX_DETECTION_DIM:
        scale = MAX_DETECTION_DIM / max(orig_w, orig_h)
        det_w = int(orig_w * scale)
        det_h = int(orig_h * scale)
        det_img = cv2.resize(orig_img, (det_w, det_h), interpolation=cv2.INTER_AREA)
    else:
        det_img = orig_img
        det_w, det_h = orig_w, orig_h

    result["debug"]["detection_width"] = det_w
    result["debug"]["detection_height"] = det_h

    # Detección Facial
    try:
        gray = cv2.cvtColor(det_img, cv2.COLOR_BGR2GRAY)
        
        # Ecualizamos la iluminación para mejorar el contraste en fotos oscuras
        gray = cv2.equalizeHist(gray)

        # En nuestra imagen estandarizada, una cara debe ocupar al menos ~10% del ancho
        min_face_size = max(40, int(det_w * 0.10))

        # Aumentamos minNeighbors a 6 para exigir más certeza del algoritmo y eliminar falsos positivos
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=6, 
            minSize=(min_face_size, min_face_size)
        )

        if len(faces) > 0:
            # Priorizar la cara más grande (área = w * h)
            dominant_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = dominant_face
            
            result["faceDetected"] = True
            result["faceConfidence"] = 0.95 
            
            # Coordenadas RELATIVAS (0.0 a 1.0) calculadas sobre la imagen estandarizada.
            # Se aplican matemáticamente perfecto a la imagen gigante original en el frontend.
            x_min, x_max = x / det_w, (x + w) / det_w
            y_min, y_max = y / det_h, (y + h) / det_h
            
            result["faceBox"] = {"xMin": x_min, "yMin": y_min, "xMax": x_max, "yMax": y_max}

            # Padding del 50% para el recorte final
            pad_w, pad_h = w * 0.5, h * 0.5
            result["cropBox"] = {
                "xMin": max(0.0, (x - pad_w) / det_w),
                "yMin": max(0.0, (y - pad_h) / det_h),
                "xMax": min(1.0, (x + w + pad_w) / det_w),
                "yMax": min(1.0, (y + h + pad_h) / det_h)
            }

            # Búsqueda de Ojos
            roi_gray = gray[y:y+h, x:x+w]
            
            # Los ojos suelen ser el ~15% del tamaño de la cara encontrada
            min_eye_size = max(15, int(w * 0.15))

            eyes = eye_cascade.detectMultiScale(
                roi_gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(min_eye_size, min_eye_size)
            )

            if len(eyes) >= 2:
                result["eyesDetected"] = True
                result["eyeConfidence"] = 0.90
                
                sorted_eyes = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[:2]
                eye_boxes = []
                for (ex, ey, ew, eh) in sorted_eyes:
                    global_ex = x + ex
                    global_ey = y + ey
                    eye_boxes.append({
                        "xMin": global_ex / det_w,
                        "yMin": global_ey / det_h,
                        "xMax": (global_ex + ew) / det_w,
                        "yMax": (global_ey + eh) / det_h
                    })
                
                eye_boxes.sort(key=lambda box: box["xMin"])
                result["eyeBoxes"] = {
                    "leftEye": eye_boxes[0],
                    "rightEye": eye_boxes[1]
                }
            
            result["debug"]["pipeline"] = "opencv_cascade_success_standardized"

    except Exception as e:
        logger.error(f"Error en detección facial: {e}")
        result["debug"]["detection_error"] = str(e)

    # Remoción de Fondo (usa la lógica original inalterada)
    mask_bytes = None
    if rembg_session is not None:
        try:
            logger.info("Iniciando remoción de fondo...")
            image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            # Forzamos una resolución máxima para evitar picos excesivos de RAM en imágenes gigantes
            image_pil.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
            mask_img = remove(image_pil, session=rembg_session, only_mask=True)
            img_byte_arr = io.BytesIO()
            mask_img.save(img_byte_arr, format='PNG')
            mask_bytes = img_byte_arr.getvalue()
            result["backgroundRemoved"] = True
            logger.info("Fondo removido exitosamente.")
        except Exception as e:
            logger.error(f"Error al remover fondo: {e}")
            result["debug"]["bg_removal_error"] = str(e)
    else:
        result["debug"]["bg_removal_error"] = "Modelo rembg no disponible"

    return result, mask_bytes