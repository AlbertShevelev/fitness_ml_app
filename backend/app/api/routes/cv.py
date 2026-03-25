from fastapi import APIRouter, File, Form, UploadFile, HTTPException

from app.schemas.response import CVAnalysisResponse
from app.services.pose_service import analyze_pose, PoseServiceError
from app.services.quality_service import build_quality
from app.services.feature_service import build_features, FeatureServiceError

router = APIRouter()


@router.post("/analyze", response_model=CVAnalysisResponse)
async def analyze_cv(
    image: UploadFile = File(...),
    gender: str = Form(...),
    age: int = Form(...),
):
    if image is None or not image.filename:
        raise HTTPException(status_code=400, detail="Требуется изображение")

    try:
        image_bytes = await image.read()

        pose_result = analyze_pose(image_bytes)
        quality = build_quality(pose_result)
        features = build_features(pose_result)

        warnings = []
        if not quality.body_fully_visible:
            warnings.append("Тело не полностью видно в кадре.")
        if quality.keypoint_confidence_mean < 0.7:
            warnings.append("Низкая уверенность распознавания ключевых точек.")

        return CVAnalysisResponse(
            status="ok",
            quality=quality,
            keypoints=pose_result.keypoints,
            features=features,
            warnings=warnings,
            metadata={
                "engine": "mediapipe_pose_landmarker",
                "stub_mode": False,
                "gender": gender,
                "age": age,
            },
        )

    except PoseServiceError as e:
        raise HTTPException(status_code=422, detail=str(e))

    except FeatureServiceError:
        raise HTTPException(
            status_code=422,
            detail="Не удалось надежно определить ключевые точки тела. "
                "Требуется фронтальная фотография в полный рост, "
                "где полностью видны плечи, таз, колени и голеностоп."
        )

    except Exception as e:
        print("Unexpected CV error:", repr(e))
        raise HTTPException(status_code=500, detail="Внутренняя ошибка CV-модуля")