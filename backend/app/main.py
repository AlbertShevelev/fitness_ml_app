from fastapi import FastAPI
from app.api.routes.cv import router as cv_router

app = FastAPI(
    title="Fitness ML CV Service",
    version="0.1.0",
    description="MediaPipe Pose Landmarker backend for posture and body feature extraction.",
)

app.include_router(cv_router, prefix="/api/v1/cv", tags=["cv"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
