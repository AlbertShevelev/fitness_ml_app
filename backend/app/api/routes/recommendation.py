from fastapi import APIRouter, HTTPException

from app.schemas.response import RecommendationGenerateRequest, RecommendationResponse
from app.services.recommendation_service import RecommendationError, generate_recommendation

router = APIRouter()


@router.post('/generate', response_model=RecommendationResponse)
def generate_recommendation_endpoint(payload: RecommendationGenerateRequest):
    try:
        return generate_recommendation(payload)
    except RecommendationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        print('Unexpected recommendation error:', repr(e))
        raise HTTPException(status_code=500, detail='Внутренняя ошибка recommendation-модуля')
