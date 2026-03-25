from __future__ import annotations

from app.schemas.response import FeatureSet, Keypoint
from app.utils.geometry import angle_at_point, distance, figure_height, midpoint, segment_angle_deg

class FeatureServiceError(ValueError):
    pass

REQUIRED = [
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

def _require_keypoints(points: dict) -> None:
    missing = [name for name in REQUIRED if name not in points]
    if missing:
        raise FeatureServiceError(
            f"Недостаточно ключевых точек для расчёта признаков: {', '.join(missing)}"
        )

    out_of_frame = [
        name for name in REQUIRED
        if not _is_in_frame(points[name]) or points[name].confidence < 0.5
    ]
    if out_of_frame:
        raise FeatureServiceError(
            f"Часть обязательных ключевых точек вне кадра или определена ненадёжно: {', '.join(out_of_frame)}"
        )

def build_features(pose_result) -> FeatureSet:
    points = pose_result.keypoints
    _require_keypoints(points)

    ls = points["left_shoulder"]
    rs = points["right_shoulder"]
    lh = points["left_hip"]
    rh = points["right_hip"]
    lk = points["left_knee"]
    rk = points["right_knee"]
    la = points["left_ankle"]
    ra = points["right_ankle"]

    fig_h = figure_height(points)

    ms_x, ms_y = midpoint(ls, rs)
    mh_x, mh_y = midpoint(lh, rh)
    mid_shoulder = Keypoint(x=ms_x, y=ms_y, z=None, confidence=1.0)
    mid_hip = Keypoint(x=mh_x, y=mh_y, z=None, confidence=1.0)

    left_leg_len = distance(lh, lk) + distance(lk, la)
    right_leg_len = distance(rh, rk) + distance(rk, ra)

    return FeatureSet(
        torso_tilt_deg=segment_angle_deg(mid_hip, mid_shoulder, reference="vertical"),
        shoulder_tilt_deg=segment_angle_deg(ls, rs, reference="horizontal"),
        pelvis_tilt_deg=segment_angle_deg(lh, rh, reference="horizontal"),
        left_knee_angle_deg=angle_at_point(lh, lk, la),
        right_knee_angle_deg=angle_at_point(rh, rk, ra),
        left_hip_angle_deg=angle_at_point(ls, lh, lk),
        right_hip_angle_deg=angle_at_point(rs, rh, rk),
        shoulder_width_ratio=distance(ls, rs) / fig_h,
        hip_width_ratio=distance(lh, rh) / fig_h,
        torso_length_ratio=distance(mid_shoulder, mid_hip) / fig_h,
        leg_length_ratio=((left_leg_len + right_leg_len) / 2.0) / fig_h,
        shoulder_asymmetry=abs(ls.y - rs.y) / fig_h,
        pelvis_asymmetry=abs(lh.y - rh.y) / fig_h,
    )
