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


class Questionnaire(BaseModel):
    gender: str
    age: int = Field(..., ge=10, le=100)
    height_cm: float = Field(..., gt=100.0, le=250.0)
    weight_kg: float = Field(..., gt=25.0, le=300.0)
    goal: str = Field(..., description="fat_loss | maintenance | muscle_gain | recomposition")
    experience_level: str = Field(default="beginner", description="beginner | intermediate | advanced")


class SurrogateInput(BaseModel):
    Lx: float = Field(..., ge=0.5, le=1.5)
    Ly: float = Field(..., ge=0.1, le=0.4)
    Lz: float = Field(..., ge=0.1, le=0.4)
    E: float = Field(..., ge=5e4, le=5e5)
    nu: float = Field(..., ge=0.2, le=0.45)
    tx: float = Field(..., ge=100.0, le=5000.0)
    ty: float = 0.0
    tz: float = 0.0


class SurrogateOutput(BaseModel):
    umax: float
    U: float
    sigmavm_max: float
    Rx: float


class SurrogateInterpretation(BaseModel):
    load_score: int = Field(..., ge=0, le=100)
    level: str
    summary: str
    progression_hint: str


class SurrogatePredictRequest(BaseModel):
    questionnaire: Questionnaire
    features: FeatureSet


class SurrogatePredictionResponse(BaseModel):
    status: str
    surrogate_input: SurrogateInput
    prediction: SurrogateOutput
    interpretation: SurrogateInterpretation
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
