import io
import hashlib
from typing import Dict, Any, Tuple
import numpy as np
import cv2
from PIL import Image
from rembg import remove, new_session

# Initialize Rembg session for background removal
rembg_session = new_session("u2net")

# Initialize OpenCV Haar Cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def hash_image(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()

def process_image(image_bytes: bytes) -> Tuple[Dict[str, Any], bytes]:
    """Analyzes image, extracts metadata via OpenCV, and generates a mask."""
    
    # 1. Load image for OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Invalid image data. Ensure the file is a valid image.")

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

    # 2. Run Face & Eye Detection
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
            
            result["debug"]["pipeline"] = "opencv_cascade_success"

    except Exception as e:
        result["debug"]["detection_error"] = str(e)

    # 3. Background Removal (Mask Generation)
    mask_bytes = None
    try:
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        mask_img = remove(image_pil, session=rembg_session, only_mask=True)
        img_byte_arr = io.BytesIO()
        mask_img.save(img_byte_arr, format='PNG')
        mask_bytes = img_byte_arr.getvalue()
        result["backgroundRemoved"] = True
    except Exception as e:
        result["debug"]["bg_removal_error"] = str(e)

    return result, mask_bytes