import cv2
import numpy as np
from fastapi import APIRouter, Form, UploadFile

from services.analyzer import Analyzer
# O que é o APIRouter?

router = APIRouter()
analyzer = Analyzer(samples_target=60)
camera_material_id = 1  # Temporário ou vindo do FrontEnd (MAIS VINDO DO FRONTED)

@router.post("/analyze")
async def analyze_image(
    file
)
