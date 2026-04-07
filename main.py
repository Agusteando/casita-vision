import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
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
logger = logging.getLogger("uvicorn.error")

DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "cache"
MASKS_DIR = DATA_DIR / "masks"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MASKS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Avatar Vision Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.cors_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
        return "<h1>Error: index.html not found!</h1>"
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "vision-service"}

@app.get("/mask/{image_key}")
async def get_mask(image_key: str):
    mask_path = MASKS_DIR / f"{image_key}.png"
    if not mask_path.exists():
        raise HTTPException(status_code=404, detail="Mask not found")
    headers = {"Access-Control-Allow-Origin": "*"}
    return FileResponse(mask_path, media_type="image/png", headers=headers)

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
            raise HTTPException(status_code=400, detail=f"Failed to fetch image: {str(e)}")
    raise HTTPException(status_code=400, detail="Must provide either imageUrl or file upload")

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image(
    imageUrl: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    try:
        image_bytes = await get_image_bytes(imageUrl, file)
        image_key = hash_image(image_bytes)
        cache_file = CACHE_DIR / f"{image_key}.json"
        
        if cache_file.exists():
            with open(cache_file, "r") as f:
                return AnalyzeResponse(**json.load(f))

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

        return AnalyzeResponse(**response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return AnalyzeResponse(ok=False, error=str(e), debug={"exception": type(e).__name__})