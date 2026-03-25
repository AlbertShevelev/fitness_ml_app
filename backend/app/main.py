from fastapi import FastAPI
from app.api.routes.cv import router as cv_router
from app.api.routes.surrogate import router as surrogate_router

app = FastAPI(
    title="Fitness ML CV Service",
    version="0.2.0",
    description="MediaPipe Pose Landmarker backend with surrogate biomechanical inference.",
)

app.include_router(cv_router, prefix="/api/v1/cv", tags=["cv"])
app.include_router(surrogate_router, prefix="/api/v1/surrogate", tags=["surrogate"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
