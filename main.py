import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings
import httpx

from vision import process_image, hash_image

class Settings(BaseSettings):
    cors_origins: str = "*"  # Restaurado para evitar conflictos con el .env existente
    base_url: str = "http://localhost:8000"
    class Config:
        env_file = ".env"
        extra = "ignore"     # Ignora de forma segura cualquier otra variable obsoleta en el .env

settings = Settings()

# Configurar logging visible
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")

DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "cache"
MASKS_DIR = DATA_DIR / "masks"
ORIGINALS_DIR = DATA_DIR / "originals" 

for d in [CACHE_DIR, MASKS_DIR, ORIGINALS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Caché de Nivel 1 (Memoria): Almacena las respuestas JSON para acceso instantáneo
MEMORY_CACHE: Dict[str, dict] = {}
MAX_MEMORY_CACHE = 10000  # Límite seguro para no saturar RAM en organizaciones grandes

app = FastAPI(title="Avatar Vision Service", version="1.0.0")

# Middleware Universal de CORS
@app.middleware("http")
async def universal_cors_middleware(request: Request, call_next):
    if request.method == "OPTIONS":
        response = JSONResponse(status_code=200, content="OK")
    else:
        response = await call_next(request)
    
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    allow_headers = request.headers.get("access-control-request-headers", "Content-Type, Authorization, Accept")
    response.headers["Access-Control-Allow-Headers"] = allow_headers
    # Exponemos las cabeceras de caché al navegador para que el frontend pueda aprovecharlas
    response.headers["Access-Control-Expose-Headers"] = "Cache-Control, ETag"
    
    return response

# Captura global de errores
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Error crítico en el servidor: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"ok": False, "error": "Error interno del servidor", "debug": str(exc)},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*"
        }
    )

class AnalyzeResponse(BaseModel):
    ok: bool
    imageKey: Optional[str] = None
    faceDetected: bool = False
    faceConfidence: float = 0.0
    faceBox: Optional[dict] = None
    cropBox: Optional[dict] = None
    eyesDetected: bool = False
    eyeConfidence: float = 0.0
    eyeBoxes: Optional[dict] = None
    backgroundRemoved: bool = False
    maskAvailable: bool = False
    maskUrl: Optional[str] = None
    debug: Optional[dict] = None
    error: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    index_path = Path("index.html")
    if not index_path.exists():
        return "<h1>Error: No se encontró el archivo index.html</h1>"
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/image/{image_key}")
async def get_image(image_key: str):
    """Devuelve la imagen original con políticas agresivas de caché HTTP del lado del cliente."""
    img_path = ORIGINALS_DIR / image_key
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Imagen no encontrada")
    return FileResponse(
        img_path,
        headers={
            "Cache-Control": "public, max-age=31536000, immutable",
            "ETag": f'"{image_key}"'
        }
    )

@app.get("/mask/{image_key}")
async def get_mask(image_key: str):
    """Devuelve la máscara transparente generada con políticas agresivas de caché HTTP."""
    mask_path = MASKS_DIR / f"{image_key}.png"
    if not mask_path.exists():
        raise HTTPException(status_code=404, detail="Máscara no encontrada")
    return FileResponse(
        mask_path, 
        media_type="image/png",
        headers={
            "Cache-Control": "public, max-age=31536000, immutable",
            "ETag": f'"{image_key}-mask"'
        }
    )

async def get_image_bytes(imageUrl: Optional[str], file: Optional[UploadFile]) -> bytes:
    if file:
        return await file.read()
    if imageUrl:
        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                response = await client.get(imageUrl)
                response.raise_for_status()
                return response.content
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"No se pudo descargar la imagen: {str(e)}")
    raise HTTPException(status_code=400, detail="Debe proporcionar un enlace o subir un archivo de imagen")

def _prepare_response_with_cache_flag(base_data: dict, source: str) -> dict:
    """Clona el diccionario e inyecta la bandera de origen de caché sin mutar el original en memoria."""
    response_copy = dict(base_data)
    response_copy["debug"] = dict(response_copy.get("debug", {}))
    response_copy["debug"]["cache_source"] = source
    return response_copy

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image(
    imageUrl: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    try:
        # Descargamos o leemos los bytes primero para generar una firma única inmutable
        image_bytes = await get_image_bytes(imageUrl, file)
        image_key = hash_image(image_bytes)
        
        # 1. BÚSQUEDA EN CACHÉ DE NIVEL 1 (Memoria RAM ultra-rápida)
        if image_key in MEMORY_CACHE:
            logger.info(f"⚡ [Hit Memoria] Análisis instantáneo servido para: {image_key}")
            return AnalyzeResponse(**_prepare_response_with_cache_flag(MEMORY_CACHE[image_key], "memory"))

        # 2. BÚSQUEDA EN CACHÉ DE NIVEL 2 (Disco duro persistente)
        cache_file = CACHE_DIR / f"{image_key}.json"
        if cache_file.exists():
            logger.info(f"📂 [Hit Disco] Análisis recuperado de disco para: {image_key}")
            with open(cache_file, "r") as f:
                disk_data = json.load(f)
            
            # Subimos el dato a memoria para el próximo request
            if len(MEMORY_CACHE) >= MAX_MEMORY_CACHE:
                MEMORY_CACHE.clear()  # Previene fugas de memoria a largo plazo
            MEMORY_CACHE[image_key] = disk_data
            
            return AnalyzeResponse(**_prepare_response_with_cache_flag(disk_data, "disk"))

        # 3. PROCESAMIENTO NUEVO (Miss de caché)
        logger.info(f"⚙️ [Miss] Procesando y analizando nueva imagen: {image_key}")
        
        # Guardar imagen original para evadir el problema de "Tainted Canvas" en el UI Frontend
        original_path = ORIGINALS_DIR / image_key
        if not original_path.exists():
            with open(original_path, "wb") as f:
                f.write(image_bytes)

        # Análisis real mediante Inteligencia Artificial y OpenCV
        metadata, mask_bytes = process_image(image_bytes)
        
        mask_available = False
        mask_url = None
        if mask_bytes and metadata.get("backgroundRemoved"):
            mask_path = MASKS_DIR / f"{image_key}.png"
            with open(mask_path, "wb") as f:
                f.write(mask_bytes)
            mask_available = True
            mask_url = f"{settings.base_url.rstrip('/')}/mask/{image_key}"

        response_data = {
            "ok": True,
            "imageKey": image_key,
            "maskAvailable": mask_available,
            "maskUrl": mask_url,
            **metadata
        }

        # Guardar en Disco (Nivel 2)
        with open(cache_file, "w") as f:
            json.dump(response_data, f)
            
        # Guardar en Memoria (Nivel 1)
        if len(MEMORY_CACHE) >= MAX_MEMORY_CACHE:
            MEMORY_CACHE.clear()
        MEMORY_CACHE[image_key] = response_data

        logger.info(f"✅ Análisis completado y guardado permanentemente para: {image_key}")
        
        return AnalyzeResponse(**_prepare_response_with_cache_flag(response_data, "miss"))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fallo en el análisis: {str(e)}")
        return AnalyzeResponse(ok=False, error=str(e), debug={"exception": type(e).__name__})