from __future__ import annotations

from app.schemas.response import FeatureSet, QualityResult


VISIBLE_THRESHOLD = 0.5



from app.schemas.response import QualityResult

REQUIRED_FULL_BODY = [
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

def _is_in_frame(kp) -> bool:
    return 0.0 <= kp.x <= 1.0 and 0.0 <= kp.y <= 1.0

def build_quality(pose_result) -> QualityResult:
    points = pose_result.keypoints

    if not points:
        return QualityResult(
            body_detected=False,
            body_fully_visible=False,
            photo_quality_score=0.0,
            keypoint_confidence_mean=0.0,
            visible_keypoints_ratio=0.0,
        )

    confidences = [kp.confidence for kp in points.values()]
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0

    visible_required = 0
    for name in REQUIRED_FULL_BODY:
        if name in points:
            kp = points[name]
            if kp.confidence > 0.5 and _is_in_frame(kp):
                visible_required += 1

    visible_keypoints_ratio = visible_required / len(REQUIRED_FULL_BODY)
    body_fully_visible = visible_keypoints_ratio >= 1.0

    photo_quality_score = min(1.0, 0.7 * mean_conf + 0.3 * visible_keypoints_ratio)

    return QualityResult(
        body_detected=True,
        body_fully_visible=body_fully_visible,
        photo_quality_score=photo_quality_score,
        keypoint_confidence_mean=mean_conf,
        visible_keypoints_ratio=visible_keypoints_ratio,
    )



def build_warnings(quality: QualityResult, features: FeatureSet) -> list[str]:
    warnings: list[str] = []
    if not quality.body_fully_visible:
        warnings.append("Тело не полностью или ненадёжно определено в кадре.")
    if quality.keypoint_confidence_mean < 0.65:
        warnings.append("Низкая уверенность детекции ключевых точек.")
    if quality.photo_quality_score < 0.7:
        warnings.append("Желательно повторить фото при лучшем освещении и на более простом фоне.")
    if features.shoulder_asymmetry > 0.03:
        warnings.append("Выявлена заметная асимметрия плеч по фронтальному снимку.")
    if features.pelvis_asymmetry > 0.03:
        warnings.append("Выявлена заметная асимметрия таза по фронтальному снимку.")
    return warnings
