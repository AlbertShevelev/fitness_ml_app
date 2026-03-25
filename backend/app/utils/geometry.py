from __future__ import annotations

import math
from typing import Mapping

from app.schemas.response import Keypoint


PointMap = Mapping[str, Keypoint]


def distance(a: Keypoint, b: Keypoint) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)



def midpoint(a: Keypoint, b: Keypoint) -> tuple[float, float]:
    return ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)



def angle_at_point(a: Keypoint, b: Keypoint, c: Keypoint) -> float:
    bax_x, bax_y = a.x - b.x, a.y - b.y
    bcx_x, bcx_y = c.x - b.x, c.y - b.y
    dot = bax_x * bcx_x + bax_y * bcx_y
    norm_ba = math.hypot(bax_x, bax_y)
    norm_bc = math.hypot(bcx_x, bcx_y)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    cos_theta = max(-1.0, min(1.0, dot / (norm_ba * norm_bc)))
    return math.degrees(math.acos(cos_theta))



def segment_angle_deg(a: Keypoint, b: Keypoint, reference: str = "horizontal") -> float:
    dx = b.x - a.x
    dy = b.y - a.y
    angle = math.degrees(math.atan2(dy, dx))
    if reference == "horizontal":
        return abs(angle)
    if reference == "vertical":
        return abs(90.0 - abs(angle))
    raise ValueError("reference must be 'horizontal' or 'vertical'")



def figure_height(points: PointMap) -> float:
    ys = [p.y for p in points.values()]
    return max(max(ys) - min(ys), 1e-6)
