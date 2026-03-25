from __future__ import annotations
import numpy as np
import io
import os
from dataclasses import dataclass
from typing import Any

from PIL import Image

from app.schemas.response import Keypoint

MODEL_PATH = os.getenv("POSE_LANDMARKER_MODEL_PATH", "models/pose_landmarker.task")
USE_STUB = os.getenv("USE_STUB_POSE", "0") == "1"


class PoseServiceError(RuntimeError):
    pass


@dataclass
class PoseResult:
    keypoints: dict[str, Keypoint]
    raw_result: Any | None = None


MEDIAPIPE_TO_APP_NAMES = {
    0: "nose",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
}


_cached_detector = None



def _stub_pose() -> PoseResult:
    return PoseResult(
        keypoints={
            "nose": Keypoint(x=0.50, y=0.08, z=0.0, confidence=0.95),
            "left_shoulder": Keypoint(x=0.42, y=0.22, z=-0.08, confidence=0.93),
            "right_shoulder": Keypoint(x=0.58, y=0.22, z=-0.07, confidence=0.94),
            "left_elbow": Keypoint(x=0.38, y=0.34, z=-0.07, confidence=0.91),
            "right_elbow": Keypoint(x=0.62, y=0.34, z=-0.07, confidence=0.91),
            "left_wrist": Keypoint(x=0.36, y=0.48, z=-0.06, confidence=0.89),
            "right_wrist": Keypoint(x=0.64, y=0.48, z=-0.06, confidence=0.89),
            "left_hip": Keypoint(x=0.45, y=0.49, z=-0.03, confidence=0.90),
            "right_hip": Keypoint(x=0.55, y=0.50, z=-0.04, confidence=0.91),
            "left_knee": Keypoint(x=0.46, y=0.71, z=-0.02, confidence=0.89),
            "right_knee": Keypoint(x=0.54, y=0.71, z=-0.02, confidence=0.88),
            "left_ankle": Keypoint(x=0.47, y=0.93, z=0.00, confidence=0.87),
            "right_ankle": Keypoint(x=0.53, y=0.93, z=0.00, confidence=0.86),
        },
        raw_result=None,
    )



def _get_detector():
    global _cached_detector
    if _cached_detector is not None:
        return _cached_detector

    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except Exception as exc:  # pragma: no cover
        raise PoseServiceError("MediaPipe не установлен. Установите зависимости из requirements.txt.") from exc

    if not os.path.exists(MODEL_PATH):
        raise PoseServiceError(
            f"Не найден файл модели Pose Landmarker: {MODEL_PATH}. Скачайте pose_landmarker.task и укажите путь через POSE_LANDMARKER_MODEL_PATH."
        )

    base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    _cached_detector = vision.PoseLandmarker.create_from_options(options)
    return _cached_detector



def analyze_pose(image_bytes: bytes) -> PoseResult:
    if USE_STUB:
        return _stub_pose()

    detector = _get_detector()

    try:
        import mediapipe as mp
    except Exception as exc:
        raise PoseServiceError("MediaPipe не установлен.") from exc

    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise PoseServiceError("Не удалось открыть изображение.") from exc

    try:
        numpy_image = np.asarray(pil_image, dtype=np.uint8)
        if not numpy_image.flags["C_CONTIGUOUS"]:
            numpy_image = np.ascontiguousarray(numpy_image)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=numpy_image,
        )

        result = detector.detect(mp_image)
    except Exception as exc:
        raise PoseServiceError(f"Ошибка MediaPipe при обработке изображения: {exc}") from exc
    finally:
        try:
            pil_image.close()
        except Exception:
            pass

    if not result.pose_landmarks:
        raise PoseServiceError("Человек или поза не обнаружены на изображении.")

    landmarks = result.pose_landmarks[0]
    keypoints: dict[str, Keypoint] = {}
    for idx, name in MEDIAPIPE_TO_APP_NAMES.items():
        lm = landmarks[idx]
        keypoints[name] = Keypoint(
            x=float(lm.x),
            y=float(lm.y),
            z=float(getattr(lm, "z", 0.0)),
            confidence=float(getattr(lm, "visibility", 1.0)),
        )

    return PoseResult(keypoints=keypoints, raw_result=result)