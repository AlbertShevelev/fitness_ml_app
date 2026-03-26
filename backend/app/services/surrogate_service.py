from __future__ import annotations

import csv
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

_DEFAULT_TARGET_QUANTILES = {
    "umax": {"q10": 1.0e-4, "q90": 5.0e-3},
    "U": {"q10": 5.0e-3, "q90": 3.0e-1},
    "sigmavm_max": {"q10": 5.0, "q90": 50.0},
    "Rx": {"q10": 150.0, "q90": 2500.0},
}


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
        if not state_dict:
            raise SurrogateArtifactsError("Пустой state_dict суррогатной модели.")

        keys = list(state_dict.keys())
        if all(k.startswith("net.") for k in keys):
            stripped = {k[len("net."):]: v for k, v in state_dict.items()}
            self.net.load_state_dict(stripped)
        else:
            self.net.load_state_dict(state_dict)


@dataclass
class ScoreContext:
    stats_source: str
    stimulus_score: int
    risk_score: int


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
        "fat_loss": 0.90,
        "maintenance": 1.00,
        "muscle_gain": 1.20,
        "recomposition": 1.05,
    }
    return mapping.get(goal, 1.00)



def _experience_multiplier(level: str) -> float:
    level = level.lower()
    mapping = {
        "beginner": 1.00,
        "intermediate": 1.35,
        "advanced": 1.75,
    }
    return mapping.get(level, 1.00)



def _experience_stiffness_bonus(level: str) -> float:
    level = level.lower()
    mapping = {
        "beginner": 0.00,
        "intermediate": 0.65,
        "advanced": 1.20,
    }
    return mapping.get(level, 0.00)


def _weekly_sessions_multiplier(sessions: int) -> float:
    sessions = max(0, int(sessions))
    return _clamp(0.90 + 0.10 * sessions, 1.00, 1.60)


def _training_years_multiplier(years: float) -> float:
    years = max(0.0, float(years))
    return _clamp(1.00 + 0.04 * years, 1.00, 1.40)


def _athlete_class_multiplier(athlete_class: str) -> float:
    athlete_class = athlete_class.lower()
    mapping = {
        "general": 1.00,
        "trained": 1.15,
        "competitive": 1.30,
        "elite": 1.45,
    }
    return mapping.get(athlete_class, 1.00)


def _athlete_class_stiffness_bonus(athlete_class: str) -> float:
    athlete_class = athlete_class.lower()
    mapping = {
        "general": 0.00,
        "trained": 0.25,
        "competitive": 0.55,
        "elite": 0.80,
    }
    return mapping.get(athlete_class, 0.00)



def map_cv_to_surrogate_input(q: Questionnaire, f: FeatureSet) -> SurrogateInput:
    bmi = _bmi(q.height_cm, q.weight_kg)
    bmi_norm = _clamp((bmi - 18.5) / (35.0 - 18.5), 0.0, 1.0)
    height_norm = _clamp((q.height_cm - 150.0) / 50.0, 0.0, 1.0)

    posture_penalty = (
        min(abs(f.torso_tilt_deg) / 20.0, 1.0) * 0.35
        + min(f.shoulder_asymmetry / 0.05, 1.0) * 0.30
        + min(f.pelvis_asymmetry / 0.05, 1.0) * 0.35
    )
    posture_efficiency = 1.0 - 0.20 * posture_penalty

    sessions_mult = _weekly_sessions_multiplier(q.weekly_sessions)
    years_mult = _training_years_multiplier(q.training_years)
    athlete_mult = _athlete_class_multiplier(q.athlete_class)

    Lx = _clamp(
        0.55 + 0.55 * height_norm + 0.45 * _clamp(f.leg_length_ratio, 0.35, 0.70),
        0.5,
        1.5,
    )
    Ly = _clamp(
        0.10 + 0.42 * _clamp(f.shoulder_width_ratio, 0.10, 0.45) + 0.07 * bmi_norm,
        0.1,
        0.4,
    )
    Lz = _clamp(
        0.10 + 0.30 * _clamp(f.torso_length_ratio, 0.15, 0.45) + 0.10 * bmi_norm,
        0.1,
        0.4,
    )

    E = _clamp(
        180000.0
        + 100000.0 * _experience_stiffness_bonus(q.experience_level)
        + 70000.0 * _athlete_class_stiffness_bonus(q.athlete_class)
        + 50000.0 * (1.0 - posture_penalty),
        50000.0,
        500000.0,
    )
    nu = _clamp(0.28 + 0.10 * bmi_norm, 0.20, 0.45)

    tx = _clamp(
        q.weight_kg
        * 9.81
        * _goal_multiplier(q.goal)
        * _experience_multiplier(q.experience_level)
        * sessions_mult
        * years_mult
        * athlete_mult
        * posture_efficiency,
        100.0,
        5000.0,
    )

    return SurrogateInput(Lx=Lx, Ly=Ly, Lz=Lz, E=E, nu=nu, tx=tx, ty=0.0, tz=0.0)



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

    missing = [str(path) for path in [config_path, model_path, x_scaler_path, y_scaler_path] if not path.exists()]
    if missing:
        raise SurrogateArtifactsError("Не найдены артефакты суррогатной модели: " + ", ".join(missing))

    try:
        import joblib
        import torch
    except Exception as exc:
        raise SurrogateArtifactsError("Не удалось импортировать torch/joblib. Проверьте requirements.txt.") from exc

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    model = MLP(torch, in_dim=len(config["input_cols"]), out_dim=len(config["target_cols"])).to("cpu")
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    x_scaler = joblib.load(x_scaler_path)
    y_scaler = joblib.load(y_scaler_path)

    _cached_artifacts = LoadedArtifacts(model=model, x_scaler=x_scaler, y_scaler=y_scaler, config=config)
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



def _read_quantiles_from_test_predictions() -> tuple[dict[str, dict[str, float]], str] | None:
    csv_path = SURROGATE_ARTIFACTS_DIR / "test_predictions.csv"
    if not csv_path.exists():
        return None

    buckets: dict[str, list[float]] = {name: [] for name in TARGET_COLS}
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for name in TARGET_COLS:
                    key = f"true_{name}"
                    if key in row and row[key] not in (None, ""):
                        try:
                            value = float(row[key])
                        except ValueError:
                            continue
                        if math.isfinite(value):
                            buckets[name].append(abs(value) if name == "Rx" else value)
    except Exception:
        return None

    try:
        import numpy as np
    except Exception:
        return None

    quantiles: dict[str, dict[str, float]] = {}
    for name, values in buckets.items():
        if len(values) < 10:
            return None
        arr = np.asarray(values, dtype=float)
        q10 = float(np.quantile(arr, 0.10))
        q90 = float(np.quantile(arr, 0.90))
        if q90 <= q10:
            return None
        quantiles[name] = {"q10": q10, "q90": q90}
    return quantiles, "test_predictions.csv"



def _get_target_quantiles() -> tuple[dict[str, dict[str, float]], str]:
    if USE_STUB_SURROGATE:
        return _DEFAULT_TARGET_QUANTILES, "fallback_default_stub"

    try:
        artifacts = _load_artifacts()
    except Exception:
        return _DEFAULT_TARGET_QUANTILES, "fallback_default_no_artifacts"

    quantiles = artifacts.config.get("target_quantiles")
    if isinstance(quantiles, dict) and all(name in quantiles for name in TARGET_COLS):
        return quantiles, "config.target_quantiles"

    from_csv = _read_quantiles_from_test_predictions()
    if from_csv is not None:
        return from_csv

    return _DEFAULT_TARGET_QUANTILES, "fallback_default"



def _norm_log(value: float, q10: float, q90: float, *, abs_value: bool = False) -> float:
    base = abs(value) if abs_value else value
    base = max(float(base), 0.0)
    lo = math.log1p(max(q10, 0.0))
    hi = math.log1p(max(q90, 0.0))
    if hi <= lo:
        return 0.5
    return _clamp((math.log1p(base) - lo) / (hi - lo), 0.0, 1.0)



def _risk_band(risk_score: int) -> str:
    if risk_score >= 75:
        return "high"
    if risk_score >= 45:
        return "moderate"
    return "low"



def _build_interpretation(inp: SurrogateInput, out: SurrogateOutput, f: FeatureSet) -> tuple[SurrogateInterpretation, ScoreContext]:
    target_q, source = _get_target_quantiles()

    posture = _clamp(
        0.4 * min(abs(f.torso_tilt_deg) / 20.0, 1.0)
        + 0.3 * min(f.shoulder_asymmetry / 0.05, 1.0)
        + 0.3 * min(f.pelvis_asymmetry / 0.05, 1.0),
        0.0,
        1.0,
    )

    z_U = _norm_log(out.U, target_q["U"]["q10"], target_q["U"]["q90"])
    z_sigma = _norm_log(out.sigmavm_max, target_q["sigmavm_max"]["q10"], target_q["sigmavm_max"]["q90"])
    z_umax = _norm_log(out.umax, target_q["umax"]["q10"], target_q["umax"]["q90"])
    z_Rx = _norm_log(out.Rx, target_q["Rx"]["q10"], target_q["Rx"]["q90"], abs_value=True)

    stimulus = _clamp(0.65 * z_U + 0.20 * z_sigma + 0.15 * z_Rx, 0.0, 1.0)
    risk = _clamp(0.45 * z_sigma + 0.35 * z_umax + 0.20 * posture, 0.0, 1.0)

    stimulus_score = int(round(100.0 * stimulus))
    risk_score = int(round(100.0 * risk))

    if stimulus_score < 30:
        level = "low"
    elif stimulus_score < 60:
        level = "moderate"
    else:
        level = "high"

    risk_band = _risk_band(risk_score)
    if risk_band == "high":
        summary = (
            "Полезный механический стимул оценён как "
            f"{level}, однако риск перегрузки повышен. Рекомендуется снизить агрессивность прогрессии и ограничить объём."
        )
        progression = (
            "Не увеличивать одновременно вес и объём. Сначала стабилизировать технику и при необходимости провести разгрузочную неделю."
        )
    elif level == "low" and risk_band == "low":
        summary = (
            "Механический стимул недостаточен при невысоком риске перегрузки. Допустимо постепенное увеличение рабочего объёма или интенсивности."
        )
        progression = (
            "Повышать нагрузку на 2.5–5% либо добавить 1 подход в основных упражнениях не чаще одного раза в 1–2 недели."
        )
    elif level == "moderate" and risk_band != "high":
        summary = (
            "Механический стимул находится в рабочей зоне. Текущий режим подходит для базового персонализированного плана с умеренной прогрессией."
        )
        progression = (
            "Сохранять текущую нагрузку 1 неделю, затем повышать объём постепенно, контролируя технику и восстановление."
        )
    else:
        summary = (
            "Механический стимул высокий и в целом достаточен. Основная задача — удерживать качество техники и избегать избыточного накопления утомления."
        )
        progression = (
            "Не ускорять прогрессию без необходимости. Повышение нагрузки выполнять пошагово, добавляя либо вес, либо объём, но не оба параметра сразу."
        )

    interpretation = SurrogateInterpretation(
        load_score=stimulus_score,
        stimulus_score=stimulus_score,
        risk_score=risk_score,
        level=level,
        summary=summary,
        progression_hint=progression,
    )
    return interpretation, ScoreContext(stats_source=source, stimulus_score=stimulus_score, risk_score=risk_score)



def predict_from_cv(payload: SurrogatePredictRequest) -> SurrogatePredictionResponse:
    inp = map_cv_to_surrogate_input(payload.questionnaire, payload.features)

    warnings: list[str] = [
        "Преобразование CV-признаков во входы суррогата выполнено эвристически и используется как прокси-оценка для MVP.",
    ]

    if USE_STUB_SURROGATE:
        out = _predict_stub(inp)
        warnings.append("Используется stub-режим surrogate-модели. Для реального предсказания подключите артефакты обучения.")
        stub_mode = True
    else:
        out = _predict_model(inp)
        stub_mode = False

    interpretation, score_ctx = _build_interpretation(inp, out, payload.features)
    if score_ctx.stats_source.startswith("fallback_default"):
        warnings.append(
            "Для расчёта stimulus_score и risk_score использованы резервные диапазоны. Для более точной нормировки добавьте target_quantiles в config.json или test_predictions.csv в директорию артефактов."
        )

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
            "score_stats_source": score_ctx.stats_source,
            "questionnaire_profile": {
                "weekly_sessions": payload.questionnaire.weekly_sessions,
                "training_years": payload.questionnaire.training_years,
                "athlete_class": payload.questionnaire.athlete_class,
            },
            "mapper_components": {
                "goal_multiplier": _goal_multiplier(payload.questionnaire.goal),
                "experience_multiplier": _experience_multiplier(payload.questionnaire.experience_level),
                "weekly_sessions_multiplier": _weekly_sessions_multiplier(payload.questionnaire.weekly_sessions),
                "training_years_multiplier": _training_years_multiplier(payload.questionnaire.training_years),
                "athlete_class_multiplier": _athlete_class_multiplier(payload.questionnaire.athlete_class),
            },
        },
    )
