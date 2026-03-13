import io
import os
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model

MODEL_PATH = Path(os.getenv("BMI_MODEL_PATH", "bmi_cnn_oversampled.h5"))
IMG_SIZE = int(os.getenv("BMI_IMAGE_SIZE", "224"))
IMAGE_SIZE = (IMG_SIZE, IMG_SIZE)

CLASS_NAMES = ["Normal", "Obese", "Overweight", "Underweight"]

GENDER_TO_FLOAT = {"female": 0.0, "male": 1.0}

app = FastAPI(title="BMI Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None


@app.on_event("startup")
def startup_event() -> None:
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found: {MODEL_PATH.resolve()}. "
            "Положите bmi_cnn_oversampled.h5 рядом с app.py или задайте BMI_MODEL_PATH."
        )
    model = load_model(MODEL_PATH, compile=False)


@app.get("/health")
def health() -> dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=500, detail="Модель не загружена.")

    return {
        "status": "ok",
        "model_path": str(MODEL_PATH.resolve()),
        "image_size": IMAGE_SIZE,
        "class_names": CLASS_NAMES,
        "num_inputs": len(model.inputs),
        "input_shapes": [tuple(dim if dim is not None else -1 for dim in inp.shape) for inp in model.inputs],
        "output_shape": tuple(dim if dim is not None else -1 for dim in model.output.shape),
    }


def _validate_age(age: int) -> int:
    if age < 5 or age > 120:
        raise HTTPException(status_code=400, detail="Возраст вне допустимого диапазона.")
    return age


def _validate_gender(gender: str) -> str:
    gender = gender.strip().lower()
    if gender not in GENDER_TO_FLOAT:
        raise HTTPException(status_code=400, detail="gender должен быть female или male.")
    return gender


def _read_image(file_bytes: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Файл не является корректным изображением.") from exc
    return img


def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.resize(IMAGE_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def build_model_inputs(image_tensor: np.ndarray, gender: str, age: int) -> Any:
    if model is None:
        raise RuntimeError("Model is not loaded.")

    # Exact extra feature order from main2.py: [gender_enc, age]
    extra = np.array([[GENDER_TO_FLOAT[gender], float(age)]], dtype=np.float32)

    if len(model.inputs) == 1:
        # Fallback if user later loads image-only model.
        return image_tensor

    if len(model.inputs) != 2:
        raise HTTPException(
            status_code=500,
            detail=f"Ожидалась модель с 1 или 2 входами, получено: {len(model.inputs)}",
        )

    first_shape = tuple(dim if dim is not None else -1 for dim in model.inputs[0].shape)
    second_shape = tuple(dim if dim is not None else -1 for dim in model.inputs[1].shape)

    first_is_image = len(first_shape) == 4 and first_shape[-1] == 3
    second_is_image = len(second_shape) == 4 and second_shape[-1] == 3

    if first_is_image and not second_is_image:
        return [image_tensor, extra]
    if second_is_image and not first_is_image:
        return [extra, image_tensor]

    # Default to the training-script order.
    return [image_tensor, extra]


def postprocess_prediction(pred: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=np.float32)
    pred = np.squeeze(pred)

    if pred.ndim != 1:
        raise HTTPException(status_code=500, detail=f"Неожиданная форма выхода модели: {pred.shape}")

    if len(pred) != len(CLASS_NAMES):
        raise HTTPException(
            status_code=500,
            detail=(
                f"Число вероятностей ({len(pred)}) не совпадает с числом классов ({len(CLASS_NAMES)})."
            ),
        )

    idx = int(np.argmax(pred))
    bmi_category = CLASS_NAMES[idx]
    confidence = float(pred[idx])

    return {
        "bmi_category": bmi_category,
        "confidence": confidence,
        "probabilities": {name: float(prob) for name, prob in zip(CLASS_NAMES, pred)},
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    gender: str = Form(...),
    age: int = Form(...),
) -> dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=500, detail="Модель не загружена.")

    gender = _validate_gender(gender)
    age = _validate_age(age)

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Пустой файл.")

    image = _read_image(file_bytes)
    image_tensor = preprocess_image(image)
    model_inputs = build_model_inputs(image_tensor, gender, age)

    try:
        pred = model.predict(model_inputs, verbose=0)
        return postprocess_prediction(pred)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ошибка инференса: {exc}") from exc
