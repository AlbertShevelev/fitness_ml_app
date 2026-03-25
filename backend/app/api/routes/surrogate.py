from fastapi import APIRouter, HTTPException

from app.schemas.response import SurrogatePredictRequest, SurrogatePredictionResponse
from app.services.surrogate_service import (
    SurrogateArtifactsError,
    SurrogatePredictionError,
    predict_from_cv,
)

router = APIRouter()


@router.post("/predict", response_model=SurrogatePredictionResponse)
def predict_surrogate(payload: SurrogatePredictRequest):
    try:
        return predict_from_cv(payload)
    except SurrogateArtifactsError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except SurrogatePredictionError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        print("Unexpected surrogate error:", repr(e))
        raise HTTPException(status_code=500, detail="Внутренняя ошибка surrogate-модуля")
