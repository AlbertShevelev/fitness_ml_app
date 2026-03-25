# CV backend MVP: FastAPI + MediaPipe Pose Landmarker

Минимальный backend для анализа фронтального фото пользователя.

## Что умеет
- принимает изображение через `POST /api/v1/cv/analyze`
- извлекает pose landmarks через MediaPipe Pose Landmarker
- считает производные признаки позы и пропорций
- возвращает JSON, пригодный для интеграции с Flutter
- поддерживает режим заглушки через `USE_STUB_POSE=1`

## Структура проекта
```text
app/
  main.py
  api/routes/cv.py
  schemas/response.py
  services/
    pose_service.py
    quality_service.py
    feature_service.py
  utils/geometry.py
models/
requirements.txt
run.py
```

## Установка
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

MediaPipe для Python устанавливается через пакет `mediapipe`, а Pose Landmarker требует локальный файл модели `pose_landmarker.task` в указанном пути. Это соответствует официальному руководству MediaPipe для Python. [1]

## Быстрый старт в режиме заглушки
```bash
cp .env.example .env
export USE_STUB_POSE=1
python run.py
```

Проверка:
```bash
curl http://127.0.0.1:8000/health
```

## Реальный запуск с MediaPipe
1. Скачай официальный файл модели `pose_landmarker.task`.
2. Положи его в папку `models/`.
3. Установи:
```bash
export USE_STUB_POSE=0
export POSE_LANDMARKER_MODEL_PATH=models/pose_landmarker.task
python run.py
```

## Пример запроса
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/cv/analyze" \
  -F "image=@/absolute/path/to/photo.jpg" \
  -F "gender=male" \
  -F "age=24" \
  -F "height_cm=180" \
  -F "weight_kg=78" \
  -F "view_type=front"
```

## Пример ответа
```json
{
  "status": "ok",
  "metadata": {
    "gender": "male",
    "age": 24,
    "height_cm": 180.0,
    "weight_kg": 78.0,
    "view_type": "front"
  },
  "quality": {
    "body_detected": true,
    "body_fully_visible": true,
    "photo_quality_score": 0.88,
    "keypoint_confidence_mean": 0.90,
    "visible_keypoints_ratio": 1.0
  },
  "keypoints": {
    "left_shoulder": {"x": 0.42, "y": 0.22, "z": -0.08, "confidence": 0.93}
  },
  "features": {
    "torso_tilt_deg": 1.8,
    "shoulder_tilt_deg": 0.2,
    "pelvis_tilt_deg": 0.5,
    "left_knee_angle_deg": 177.0,
    "right_knee_angle_deg": 176.8,
    "left_hip_angle_deg": 173.2,
    "right_hip_angle_deg": 172.8,
    "shoulder_width_ratio": 0.23,
    "hip_width_ratio": 0.18,
    "torso_length_ratio": 0.30,
    "leg_length_ratio": 0.47,
    "shoulder_asymmetry": 0.01,
    "pelvis_asymmetry": 0.01
  },
  "warnings": []
}
```

## Что подключать дальше
- Flutter-клиент вместо старого `/predict`
- сохранение `features` как входов для суррогатной модели
- позже: режим `view_type=side`, силуэт, сегментация, прогресс по серии фото

## Источник
[1] Официальное руководство MediaPipe Pose Landmarker для Python: установка `mediapipe`, импорт `mediapipe.tasks.python.vision`, локальный путь к `pose_landmarker.task`, создание детектора через `PoseLandmarker.create_from_options`. См. официальный гайд Google AI Edge.
