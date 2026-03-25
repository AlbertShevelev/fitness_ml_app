from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Keypoint(BaseModel):
    x: float
    y: float
    z: Optional[float] = None
    confidence: float = Field(ge=0.0, le=1.0)


class QualityResult(BaseModel):
    body_detected: bool
    body_fully_visible: bool
    photo_quality_score: float = Field(..., ge=0.0, le=1.0)
    keypoint_confidence_mean: float = Field(..., ge=0.0, le=1.0)
    visible_keypoints_ratio: float = Field(..., ge=0.0, le=1.0)


class FeatureSet(BaseModel):
    torso_tilt_deg: float
    shoulder_tilt_deg: float
    pelvis_tilt_deg: float
    left_knee_angle_deg: float
    right_knee_angle_deg: float
    left_hip_angle_deg: float
    right_hip_angle_deg: float
    shoulder_width_ratio: float
    hip_width_ratio: float
    torso_length_ratio: float
    leg_length_ratio: float
    shoulder_asymmetry: float
    pelvis_asymmetry: float


class CVAnalysisResponse(BaseModel):
    status: str
    quality: QualityResult
    keypoints: Dict[str, Keypoint]
    features: FeatureSet
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
