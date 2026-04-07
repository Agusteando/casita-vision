import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings
import httpx

from vision import process_image, hash_image

class Settings(BaseSettings):
    cors_origins: str = "*"
    base_url: str = "http://localhost:8000"
    class Config:
        env_file = ".env"

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

app = FastAPI(title="Avatar Vision Service", version="1.0.0")

# Middleware CORS ultra-permisivo forzado
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=".*",  # Fuerza la coincidencia con cualquier origen sin importar el formato
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]      # Expone todas las cabeceras a la respuesta del navegador
)

# Captura cualquier error no controlado para evitar desconexiones que rompan el CORS
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Error crítico en el servidor: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"ok": False, "error": "Error interno del servidor", "debug": str(exc)},
        headers={"Access-Control-Allow-Origin": "*"}
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
    img_path = ORIGINALS_DIR / image_key
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Imagen no encontrada")
    return FileResponse(img_path, headers={"Access-Control-Allow-Origin": "*"})

@app.get("/mask/{image_key}")
async def get_mask(image_key: str):
    mask_path = MASKS_DIR / f"{image_key}.png"
    if not mask_path.exists():
        raise HTTPException(status_code=404, detail="Máscara no encontrada")
    return FileResponse(mask_path, media_type="image/png", headers={"Access-Control-Allow-Origin": "*"})

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

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image(
    imageUrl: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    try:
        logger.info("Recibiendo petición de análisis...")
        image_bytes = await get_image_bytes(imageUrl, file)
        image_key = hash_image(image_bytes)
        logger.info(f"Imagen procesada con clave: {image_key}")
        
        original_path = ORIGINALS_DIR / image_key
        if not original_path.exists():
            with open(original_path, "wb") as f:
                f.write(image_bytes)

        cache_file = CACHE_DIR / f"{image_key}.json"
        if cache_file.exists():
            logger.info("Devolviendo resultado desde caché.")
            with open(cache_file, "r") as f:
                return AnalyzeResponse(**json.load(f))

        logger.info("Iniciando análisis de visión...")
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

        with open(cache_file, "w") as f:
            json.dump(response_data, f)

        logger.info("Análisis completado exitosamente.")
        return AnalyzeResponse(**response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fallo en el análisis: {str(e)}")
        return AnalyzeResponse(ok=False, error=str(e), debug={"exception": type(e).__name__})