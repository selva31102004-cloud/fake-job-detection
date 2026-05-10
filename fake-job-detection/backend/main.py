"""
Fake Job Detection System - FastAPI Backend
Hybrid AI Model: ML (Random Forest + Logistic Regression) + NLP (TF-IDF + BERT-like features)
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import logging

from api.routes import router
from api.auth import router as auth_router, init_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fake Job Detection API",
    description="AI-powered system to detect fraudulent job postings",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")
app.include_router(auth_router, prefix="/api/v1")

@app.on_event("startup")
def on_startup():
    """Initialise MySQL tables on first run."""
    try:
        init_db()
        logger.info("✅  MySQL users table ready.")
    except Exception as e:
        logger.error(f"❌  DB init failed: {e}")

@app.get("/")
async def root():
    return {"message": "Fake Job Detection API is running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
