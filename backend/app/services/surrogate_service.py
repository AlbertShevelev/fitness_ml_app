from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.schemas.response import (
    FeatureSet,
    Questionnaire,
    SurrogateInput,
    SurrogateInterpretation,
    SurrogateOutput,
    SurrogatePredictRequest,
    SurrogatePredictionResponse,
)

SURROGATE_ARTIFACTS_DIR = Path(os.getenv("SURROGATE_ARTIFACTS_DIR", "mlp_surrogate_artifacts_logU"))
USE_STUB_SURROGATE = os.getenv("USE_STUB_SURROGATE", "0") == "1"

INPUT_COLS = ["Lx", "Ly", "Lz", "E", "nu", "tx", "ty", "tz"]
TARGET_COLS = ["umax", "U", "sigmavm_max", "Rx"]


class SurrogatePredictionError(RuntimeError):
    pass


class SurrogateArtifactsError(RuntimeError):
    pass


@dataclass
class LoadedArtifacts:
    model: Any
    x_scaler: Any
    y_scaler: Any
    config: dict[str, Any]


_cached_artifacts: LoadedArtifacts | None = None


class MLP:
    def __init__(self, torch_module, in_dim: int, out_dim: int):
        nn = torch_module.nn
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def __call__(self, x):
        return self.net(x)

    def to(self, device: str):
        self.net.to(device)
        return self

    def eval(self):
        self.net.eval()

    def load_state_dict(self, state_dict: dict[str, Any]):
        # Совместимость с двумя вариантами сохранения:
        # 1) model.state_dict() -> ключи вида "net.0.weight"
        # 2) model.net.state_dict() -> ключи вида "0.weight"
        if not state_dict:
            raise SurrogateArtifactsError("Пустой state_dict суррогатной модели.")

        keys = list(state_dict.keys())
        if all(k.startswith("net.") for k in keys):
            stripped = {k[len("net."):]: v for k, v in state_dict.items()}
            self.net.load_state_dict(stripped)
        else:
            self.net.load_state_dict(state_dict)



def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))



def _bmi(height_cm: float, weight_kg: float) -> float:
    h_m = height_cm / 100.0
    if h_m <= 0:
        raise SurrogatePredictionError("Некорректный рост для расчёта BMI.")
    return weight_kg / (h_m * h_m)



def _goal_multiplier(goal: str) -> float:
    goal = goal.lower()
    mapping = {
        "fat_loss": 0.85,
        "maintenance": 1.00,
        "muscle_gain": 1.15,
        "recomposition": 1.00,
    }
    return mapping.get(goal, 1.00)



def _experience_multiplier(level: str) -> float:
    level = level.lower()
    mapping = {
        "beginner": 0.75,
        "intermediate": 1.00,
        "advanced": 1.20,
    }
    return mapping.get(level, 0.75)



def _experience_stiffness_bonus(level: str) -> float:
    level = level.lower()
    mapping = {
        "beginner": 0.00,
        "intermediate": 0.50,
        "advanced": 1.00,
    }
    return mapping.get(level, 0.00)



def map_cv_to_surrogate_input(q: Questionnaire, f: FeatureSet) -> SurrogateInput:
    """
    Эвристическое приведение признаков CV и анкеты к входам FEM-surrogate.
    Это не прямое биомеханическое измерение, а прокси-модель для MVP.
    """
    bmi = _bmi(q.height_cm, q.weight_kg)
    bmi_norm = _clamp((bmi - 18.5) / (35.0 - 18.5), 0.0, 1.0)
    height_norm = _clamp((q.height_cm - 150.0) / 50.0, 0.0, 1.0)

    posture_penalty = (
        min(abs(f.torso_tilt_deg) / 20.0, 1.0) * 0.35
        + min(f.shoulder_asymmetry / 0.05, 1.0) * 0.30
        + min(f.pelvis_asymmetry / 0.05, 1.0) * 0.35
    )
    posture_factor = 1.0 + 0.35 * posture_penalty

    Lx = _clamp(
        0.55
        + 0.55 * height_norm
        + 0.45 * _clamp(f.leg_length_ratio, 0.35, 0.70),
        0.5,
        1.5,
    )
    Ly = _clamp(
        0.10
        + 0.42 * _clamp(f.shoulder_width_ratio, 0.10, 0.45)
        + 0.07 * bmi_norm,
        0.1,
        0.4,
    )
    Lz = _clamp(
        0.10
        + 0.30 * _clamp(f.torso_length_ratio, 0.15, 0.45)
        + 0.10 * bmi_norm,
        0.1,
        0.4,
    )

    E = _clamp(
        180000.0
        + 100000.0 * _experience_stiffness_bonus(q.experience_level)
        + 50000.0 * (1.0 - posture_penalty),
        50000.0,
        500000.0,
    )
    nu = _clamp(0.28 + 0.10 * bmi_norm, 0.20, 0.45)

    tx = _clamp(
        q.weight_kg * 9.81 * _goal_multiplier(q.goal) * _experience_multiplier(q.experience_level) * posture_factor,
        100.0,
        5000.0,
    )

    return SurrogateInput(
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        E=E,
        nu=nu,
        tx=tx,
        ty=0.0,
        tz=0.0,
    )



def _load_artifacts() -> LoadedArtifacts:
    global _cached_artifacts
    if _cached_artifacts is not None:
        return _cached_artifacts

    if USE_STUB_SURROGATE:
        raise SurrogateArtifactsError("Stub mode enabled")

    config_path = SURROGATE_ARTIFACTS_DIR / "config.json"
    model_path = SURROGATE_ARTIFACTS_DIR / "mlp_surrogate.pt"
    x_scaler_path = SURROGATE_ARTIFACTS_DIR / "x_scaler.joblib"
    y_scaler_path = SURROGATE_ARTIFACTS_DIR / "y_scaler.joblib"

    missing = [
        str(path) for path in [config_path, model_path, x_scaler_path, y_scaler_path] if not path.exists()
    ]
    if missing:
        raise SurrogateArtifactsError(
            "Не найдены артефакты суррогатной модели: " + ", ".join(missing)
        )

    try:
        import joblib
        import torch
    except Exception as exc:
        raise SurrogateArtifactsError(
            "Не удалось импортировать torch/joblib. Проверьте requirements.txt."
        ) from exc

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    model = MLP(torch, in_dim=len(config["input_cols"]), out_dim=len(config["target_cols"])).to("cpu")
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    x_scaler = joblib.load(x_scaler_path)
    y_scaler = joblib.load(y_scaler_path)

    _cached_artifacts = LoadedArtifacts(
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        config=config,
    )
    return _cached_artifacts



def _inverse_transform_targets(values, target_cols: list[str], log1p_targets: set[str]):
    import numpy as np

    restored = values.copy()
    for i, name in enumerate(target_cols):
        if name in log1p_targets:
            restored[:, i] = np.expm1(restored[:, i])
    return restored



def _predict_stub(inp: SurrogateInput) -> SurrogateOutput:
    area = max(inp.Ly * inp.Lz, 1e-6)
    compliance = (inp.Lx / max(inp.E * area, 1e-6)) * 1.0e5
    umax = compliance * inp.tx * 1.0e-3
    U = 0.5 * inp.tx * umax
    sigmavm_max = (inp.tx / area) * 1.0e-2
    Rx = -inp.tx
    return SurrogateOutput(
        umax=float(umax),
        U=float(U),
        sigmavm_max=float(sigmavm_max),
        Rx=float(Rx),
    )



def _predict_model(inp: SurrogateInput) -> SurrogateOutput:
    artifacts = _load_artifacts()

    try:
        import numpy as np
        import torch
    except Exception as exc:
        raise SurrogatePredictionError("Не удалось импортировать numpy/torch.") from exc

    row = [[getattr(inp, col) for col in artifacts.config["input_cols"]]]
    X = np.asarray(row, dtype=float)
    Xs = artifacts.x_scaler.transform(X)

    with torch.no_grad():
        pred_scaled = artifacts.model.net(torch.tensor(Xs, dtype=torch.float32)).cpu().numpy()

    pred_transformed = artifacts.y_scaler.inverse_transform(pred_scaled)
    pred = _inverse_transform_targets(
        pred_transformed,
        artifacts.config["target_cols"],
        set(artifacts.config.get("log1p_targets", [])),
    )

    data = {name: float(pred[0, i]) for i, name in enumerate(artifacts.config["target_cols"])}
    return SurrogateOutput(**data)



def _build_interpretation(inp: SurrogateInput, out: SurrogateOutput, f: FeatureSet) -> SurrogateInterpretation:
    load_norm = _clamp((inp.tx - 100.0) / (5000.0 - 100.0), 0.0, 1.0)
    posture_norm = _clamp(
        0.4 * min(abs(f.torso_tilt_deg) / 20.0, 1.0)
        + 0.3 * min(f.shoulder_asymmetry / 0.05, 1.0)
        + 0.3 * min(f.pelvis_asymmetry / 0.05, 1.0),
        0.0,
        1.0,
    )
    score = int(round(100.0 * (0.7 * load_norm + 0.3 * posture_norm)))

    if score < 35:
        level = "low"
        summary = "Низкий прогнозируемый механический стимул. Допустимо увеличение тренировочного объёма или интенсивности."
        progression = "Увеличивать нагрузку на 5–10% не чаще одного раза в 1–2 недели при сохранении техники."
    elif score < 70:
        level = "moderate"
        summary = "Умеренный прогнозируемый механический стимул. Режим подходит для базового персонализированного плана."
        progression = "Сохранять текущую нагрузку 1 неделю, затем повышать объём постепенно, контролируя технику и восстановление."
    else:
        level = "high"
        summary = "Высокий прогнозируемый механический стимул. Желательна осторожная прогрессия и контроль техники выполнения."
        progression = "Не увеличивать одновременно объём и интенсивность. Предпочтительна пошаговая прогрессия с промежуточной неделей стабилизации."

    return SurrogateInterpretation(
        load_score=score,
        level=level,
        summary=summary,
        progression_hint=progression,
    )



def predict_from_cv(payload: SurrogatePredictRequest) -> SurrogatePredictionResponse:
    inp = map_cv_to_surrogate_input(payload.questionnaire, payload.features)

    warnings: list[str] = [
        "Преобразование CV-признаков в входы суррогата выполнено эвристически и используется как прокси-оценка для MVP.",
    ]

    if USE_STUB_SURROGATE:
        out = _predict_stub(inp)
        warnings.append("Используется stub-режим surrogate-модели. Для реального предсказания подключите артефакты обучения.")
        stub_mode = True
    else:
        out = _predict_model(inp)
        stub_mode = False

    interpretation = _build_interpretation(inp, out, payload.features)

    return SurrogatePredictionResponse(
        status="ok",
        surrogate_input=inp,
        prediction=out,
        interpretation=interpretation,
        warnings=warnings,
        metadata={
            "engine": "mlp_surrogate",
            "stub_mode": stub_mode,
            "artifacts_dir": str(SURROGATE_ARTIFACTS_DIR),
        },
    )
