from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.schemas.response import (
    Questionnaire,
    RecommendationGenerateRequest,
    RecommendationResponse,
    SurrogateInterpretation,
    SurrogateOutput,
)
from pydantic import BaseModel, Field


class RecommendationError(RuntimeError):
    pass


class WorkoutExercise(BaseModel):
    name: str
    sets: int = Field(..., ge=1, le=10)
    reps: str
    rest_sec: int = Field(..., ge=0, le=600)
    tempo: str = '2-0-2'
    notes: str = ''


class WorkoutDay(BaseModel):
    day_label: str
    focus: str
    exercises: List[WorkoutExercise]


class NutritionPlan(BaseModel):
    kcal_target: int
    protein_g: int
    fat_g: int
    carbs_g: int
    water_ml: int
    notes: List[str]


class ProgressionRule(BaseModel):
    week: str
    condition: str
    action: str
    rationale: str


class RecommendationPlan(BaseModel):
    plan_id: str
    title: str
    summary: str
    weekly_frequency: int = Field(..., ge=2, le=7)
    difficulty: str
    workout_days: List[WorkoutDay]
    nutrition: NutritionPlan
    progression_rules: List[ProgressionRule]
    safety_notes: List[str]


# ---------- Internal helpers ----------

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _bmi(q: Questionnaire) -> float:
    h = q.height_cm / 100.0
    return q.weight_kg / (h * h)


def _activity_factor(level: str) -> float:
    return {
        'beginner': 1.40,
        'intermediate': 1.55,
        'advanced': 1.70,
    }.get(level.lower(), 1.40)


def _goal_kcal_adjustment(goal: str) -> int:
    return {
        'fat_loss': -350,
        'maintenance': 0,
        'recomposition': -100,
        'muscle_gain': 250,
    }.get(goal.lower(), 0)


def _base_load_modifier(level: str) -> float:
    return {
        'low': 0.85,
        'moderate': 1.00,
        'high': 1.10,
    }.get(level.lower(), 1.00)


def _difficulty_label(q: Questionnaire, s: SurrogateInterpretation) -> str:
    if q.experience_level == 'advanced' and s.level == 'high':
        return 'advanced'
    if q.experience_level in {'intermediate', 'advanced'} and s.level in {'moderate', 'high'}:
        return 'intermediate'
    return 'beginner'


def _training_frequency(q: Questionnaire, s: SurrogateInterpretation) -> int:
    base = {
        'beginner': 3,
        'intermediate': 4,
        'advanced': 5,
    }.get(q.experience_level.lower(), 3)

    if s.level == 'low':
        return max(3, base - 1)
    if s.level == 'high':
        return min(5, base)
    return base


def _estimated_tdee(q: Questionnaire) -> int:
    # Mifflin-St Jeor approximation
    sex_bonus = 5 if q.gender.lower() == 'male' else -161
    bmr = 10 * q.weight_kg + 6.25 * q.height_cm - 5 * q.age + sex_bonus
    return int(round(bmr * _activity_factor(q.experience_level)))


def _build_nutrition(q: Questionnaire, s: SurrogateInterpretation) -> NutritionPlan:
    tdee = _estimated_tdee(q)
    kcal = tdee + _goal_kcal_adjustment(q.goal)

    if q.goal == 'muscle_gain':
        protein = round(q.weight_kg * 2.0)
        fat = round(q.weight_kg * 0.9)
    elif q.goal == 'fat_loss':
        protein = round(q.weight_kg * 2.1)
        fat = round(q.weight_kg * 0.8)
    else:
        protein = round(q.weight_kg * 1.8)
        fat = round(q.weight_kg * 0.85)

    protein_kcal = protein * 4
    fat_kcal = fat * 9
    carbs = max(80, round((kcal - protein_kcal - fat_kcal) / 4))
    water_ml = int(max(1800, q.weight_kg * 30 + 500))

    notes = [
        'Распределять белок на 3–5 приёмов пищи в течение дня.',
        'Основную часть углеводов переносить на периоды до и после тренировки.',
        'Контролировать изменение массы тела 1 раз в неделю в одинаковых условиях.',
    ]
    if q.goal == 'fat_loss':
        notes.append('Поддерживать умеренный дефицит калорий без резкого снижения потребления белка.')
    if s.level == 'low':
        notes.append('При признаках утомления не снижать калорийность дополнительно.')

    return NutritionPlan(
        kcal_target=int(kcal),
        protein_g=int(protein),
        fat_g=int(fat),
        carbs_g=int(carbs),
        water_ml=water_ml,
        notes=notes,
    )


def _ex(name: str, sets: int, reps: str, rest: int, tempo: str = '2-0-2', notes: str = '') -> WorkoutExercise:
    return WorkoutExercise(name=name, sets=sets, reps=reps, rest_sec=rest, tempo=tempo, notes=notes)


def _plan_full_body_intro(load_modifier: float) -> List[WorkoutDay]:
    sets_main = 2 if load_modifier < 0.95 else 3
    return [
        WorkoutDay(
            day_label='День 1',
            focus='Базовое full-body занятие',
            exercises=[
                _ex('Приседание с собственным весом', sets_main, '10–12', 75),
                _ex('Ягодичный мост', sets_main, '12–15', 60),
                _ex('Тяга резиновой ленты к поясу', sets_main, '10–12', 75),
                _ex('Отжимания от опоры', sets_main, '8–10', 90),
                _ex('Планка', 2, '20–30 сек', 45, notes='Сохранять нейтральное положение корпуса.'),
            ],
        ),
        WorkoutDay(
            day_label='День 2',
            focus='Техника и устойчивость',
            exercises=[
                _ex('Выпад назад без веса', sets_main, '8–10 на каждую ногу', 75),
                _ex('Румынская тяга с лёгким весом', sets_main, '10–12', 75),
                _ex('Тяга верхнего блока / резины', sets_main, '10–12', 75),
                _ex('Жим гантелей лёжа лёгкий', sets_main, '10–12', 90),
                _ex('Dead bug', 2, '8–10 на сторону', 45),
            ],
        ),
        WorkoutDay(
            day_label='День 3',
            focus='Контроль движения и умеренный объём',
            exercises=[
                _ex('Гоблет-присед лёгкий', sets_main, '10–12', 75),
                _ex('Шаги на платформу', sets_main, '10 на каждую ногу', 60),
                _ex('Горизонтальная тяга', sets_main, '10–12', 75),
                _ex('Жим гантелей сидя', sets_main, '8–10', 90),
                _ex('Боковая планка', 2, '20 сек на сторону', 45),
            ],
        ),
    ]


def _plan_recomp_base(load_modifier: float) -> List[WorkoutDay]:
    sets_main = 3 if load_modifier <= 1.0 else 4
    return [
        WorkoutDay(
            day_label='День 1',
            focus='Низ тела',
            exercises=[
                _ex('Приседание', sets_main, '8–10', 90),
                _ex('Румынская тяга', sets_main, '8–10', 90),
                _ex('Выпады', 3, '10 на ногу', 75),
                _ex('Подъёмы на носки', 3, '12–15', 45),
                _ex('Планка', 3, '30–40 сек', 45),
            ],
        ),
        WorkoutDay(
            day_label='День 2',
            focus='Верх тела',
            exercises=[
                _ex('Жим гантелей лёжа', sets_main, '8–10', 90),
                _ex('Тяга горизонтального блока', sets_main, '10–12', 75),
                _ex('Жим вверх сидя', 3, '8–10', 90),
                _ex('Тяга к лицу', 3, '12–15', 60),
                _ex('Скручивания', 3, '12–15', 45),
            ],
        ),
        WorkoutDay(
            day_label='День 3',
            focus='Смешанная силовая работа',
            exercises=[
                _ex('Гоблет-присед', sets_main, '10–12', 75),
                _ex('Тяга гантели в наклоне', sets_main, '10–12', 75),
                _ex('Отжимания', 3, '8–12', 75),
                _ex('Ягодичный мост', 3, '12–15', 60),
                _ex('Pallof press', 3, '10–12', 45),
            ],
        ),
        WorkoutDay(
            day_label='День 4',
            focus='Кардио + мобильность',
            exercises=[
                _ex('Ходьба / велотренажёр', 1, '25–35 мин', 0, notes='Интенсивность: зона 2.'),
                _ex('Мобилизация тазобедренных суставов', 2, '6–8 мин', 0),
                _ex('Мобилизация грудного отдела', 2, '6–8 мин', 0),
            ],
        ),
    ]


def _plan_muscle_gain_upper_lower(load_modifier: float) -> List[WorkoutDay]:
    sets_main = 4 if load_modifier >= 1.0 else 3
    return [
        WorkoutDay(
            day_label='День 1',
            focus='Низ тела А',
            exercises=[
                _ex('Приседание', sets_main, '6–8', 120),
                _ex('Румынская тяга', sets_main, '8–10', 120),
                _ex('Болгарские выпады', 3, '8–10 на ногу', 90),
                _ex('Сгибание ног', 3, '10–12', 60),
                _ex('Подъёмы на носки', 4, '12–15', 45),
            ],
        ),
        WorkoutDay(
            day_label='День 2',
            focus='Верх тела А',
            exercises=[
                _ex('Жим лёжа', sets_main, '6–8', 120),
                _ex('Тяга штанги / блока', sets_main, '8–10', 90),
                _ex('Жим сидя', 3, '8–10', 90),
                _ex('Тяга верхнего блока', 3, '10–12', 75),
                _ex('Сгибание на бицепс + разгибание на трицепс', 3, '10–12', 60),
            ],
        ),
        WorkoutDay(
            day_label='День 3',
            focus='Низ тела B',
            exercises=[
                _ex('Тяга классическая / трап-гриф', sets_main, '5–6', 150),
                _ex('Жим ногами', 3, '10–12', 90),
                _ex('Ягодичный мост / hip thrust', 3, '8–10', 90),
                _ex('Разгибание ног', 3, '12–15', 60),
                _ex('Антиротация корпуса', 3, '10–12', 45),
            ],
        ),
        WorkoutDay(
            day_label='День 4',
            focus='Верх тела B',
            exercises=[
                _ex('Жим гантелей под углом', sets_main, '8–10', 90),
                _ex('Горизонтальная тяга', sets_main, '8–10', 90),
                _ex('Разведения в стороны', 3, '12–15', 60),
                _ex('Тяга к лицу', 3, '12–15', 60),
                _ex('Пресс', 3, '12–15', 45),
            ],
        ),
    ]


def _plan_strength_advanced(load_modifier: float) -> List[WorkoutDay]:
    return [
        WorkoutDay(
            day_label='День 1',
            focus='Силовой присед',
            exercises=[
                _ex('Приседание', 5, '4–6', 150),
                _ex('Пауза-присед', 3, '3–4', 150),
                _ex('Румынская тяга', 4, '6–8', 120),
                _ex('Кор', 3, '30–40 сек', 45),
            ],
        ),
        WorkoutDay(
            day_label='День 2',
            focus='Силовой жим',
            exercises=[
                _ex('Жим лёжа', 5, '4–6', 150),
                _ex('Жим узким хватом', 3, '6–8', 120),
                _ex('Тяга штанги', 4, '6–8', 120),
                _ex('Задняя дельта', 3, '12–15', 60),
            ],
        ),
        WorkoutDay(
            day_label='День 3',
            focus='Силовая тяга',
            exercises=[
                _ex('Становая тяга', 5, '3–5', 180),
                _ex('Фронтальный присед', 3, '5–6', 150),
                _ex('Подтягивания / тяга верхнего блока', 4, '6–8', 90),
                _ex('Антиэкстензия корпуса', 3, '10–12', 45),
            ],
        ),
        WorkoutDay(
            day_label='День 4',
            focus='Объёмный верх',
            exercises=[
                _ex('Жим гантелей', 4, '8–10', 90),
                _ex('Тяга сидя', 4, '8–10', 90),
                _ex('Жим вверх', 3, '8–10', 90),
                _ex('Локальная работа на руки', 3, '10–12', 60),
            ],
        ),
        WorkoutDay(
            day_label='День 5',
            focus='Восстановительное кондиционирование',
            exercises=[
                _ex('Кардио низкой интенсивности', 1, '20–30 мин', 0),
                _ex('Мобильность', 1, '10–15 мин', 0),
            ],
        ),
    ]


def _select_template(q: Questionnaire, s: SurrogateInterpretation) -> RecommendationPlan:
    load_modifier = _base_load_modifier(s.level)
    difficulty = _difficulty_label(q, s)
    frequency = _training_frequency(q, s)
    nutrition = _build_nutrition(q, s)

    if q.goal == 'fat_loss':
        title = 'Снижение жировой массы: базовый адаптивный план'
        summary = 'Комбинация 3–4 силовых сессий с умеренным объёмом и одной восстановительной кардио-сессией.'
        workout_days = _plan_recomp_base(load_modifier)
        plan_id = 'fat_loss_base'
    elif q.goal == 'maintenance':
        title = 'Поддержание формы: стабилизирующий план'
        summary = 'Поддержание силовых качеств и двигательного контроля без избыточной нагрузки.'
        workout_days = _plan_recomp_base(load_modifier)[:frequency]
        plan_id = 'maintenance_base'
    elif q.goal == 'recomposition':
        title = 'Рекомпозиция: смешанный силовой план'
        summary = 'Приоритет сохранения и постепенного роста силовых показателей при контроле суммарной утомляемости.'
        workout_days = _plan_recomp_base(load_modifier)
        plan_id = 'recomp_base'
    else:
        if q.experience_level == 'advanced' and s.level == 'high':
            title = 'Набор мышечной массы: силовой продвинутый план'
            summary = 'Высокий тренировочный потенциал позволяет использовать силовой сплит с контролируемой прогрессией.'
            workout_days = _plan_strength_advanced(load_modifier)
            plan_id = 'muscle_gain_advanced'
            frequency = 5
        elif q.experience_level == 'beginner' or s.level == 'low':
            title = 'Набор мышечной массы: вводный full-body план'
            summary = 'Формирование техники базовых движений и постепенное увеличение рабочего объёма.'
            workout_days = _plan_full_body_intro(load_modifier)
            plan_id = 'muscle_gain_intro'
            frequency = 3
        else:
            title = 'Набор мышечной массы: upper/lower план'
            summary = 'Сбалансированный 4-дневный сплит для гипертрофии с умеренно-высокой нагрузкой.'
            workout_days = _plan_muscle_gain_upper_lower(load_modifier)
            plan_id = 'muscle_gain_ul'
            frequency = 4

    workout_days = workout_days[:frequency]

    progression_rules = [
        ProgressionRule(
            week='Недели 1–2',
            condition='Все подходы выполнены с сохранением техники и субъективной нагрузкой не выше RPE 7–8.',
            action='Увеличить рабочий вес на 2.5–5% или добавить 1–2 повторения в каждом основном упражнении.',
            rationale='Постепенная прогрессия создаёт достаточный стимул без резкого роста риска перегрузки.',
        ),
        ProgressionRule(
            week='Любая неделя',
            condition='Появляется выраженное ухудшение техники, боль или устойчивое ощущение чрезмерной утомлённости.',
            action='Снизить объём на 20–30% на 1 неделю и вернуться к предыдущему уровню нагрузки.',
            rationale='Временное снижение нагрузки уменьшает вероятность накопления избыточного утомления.',
        ),
        ProgressionRule(
            week='Каждая 4-я неделя',
            condition='План выполнялся полностью без пропусков и без негативных симптомов.',
            action='Провести разгрузочную неделю: сохранить упражнения, но уменьшить объём на 30–40%.',
            rationale='Разгрузка улучшает восстановление и позволяет сохранить долгосрочную адаптацию.',
        ),
    ]

    safety_notes = [
        'Перед основной частью занятия выполнять разминку 7–10 минут.',
        'Прекращать упражнение при возникновении острой боли.',
        'Поддерживать нейтральное положение корпуса и контролировать скорость эксцентрической фазы.',
    ]
    if s.level == 'low':
        safety_notes.append('При низком surrogate-уровне избегать тренировок до отказа в первые 2–3 недели.')
    if s.level == 'high':
        safety_notes.append('Даже при высоком уровне нагрузки увеличение веса выполнять только при стабильной технике.')

    return RecommendationPlan(
        plan_id=plan_id,
        title=title,
        summary=summary,
        weekly_frequency=frequency,
        difficulty=difficulty,
        workout_days=workout_days,
        nutrition=nutrition,
        progression_rules=progression_rules,
        safety_notes=safety_notes,
    )


# ---------- Public API ----------

class RecommendationGenerateRequestModel(BaseModel):
    questionnaire: Questionnaire
    surrogate_prediction: SurrogateOutput
    surrogate_interpretation: SurrogateInterpretation


class RecommendationResponseModel(BaseModel):
    status: str
    plan: RecommendationPlan
    explanation: List[str]
    metadata: dict


def generate_recommendation(payload: RecommendationGenerateRequest):
    q = payload.questionnaire
    s_out = payload.surrogate_prediction
    s_interp = payload.surrogate_interpretation

    if s_out.U < 0 or s_out.umax < 0 or s_out.sigmavm_max < 0:
        raise RecommendationError('Получены некорректные значения surrogate-предсказания.')

    plan = _select_template(q, s_interp)

    explanation = [
        f'Цель пользователя: {q.goal}.',
        f'Уровень подготовки: {q.experience_level}.',
        f'Оценка нагрузки surrogate-модуля: {s_interp.level} (score={s_interp.load_score}).',
        'План выбран с учётом сочетания тренировочной цели, предполагаемого уровня переносимости нагрузки и необходимости постепенной прогрессии.',
    ]

    return RecommendationResponse(
        status='ok',
        plan=plan.model_dump(),
        explanation=explanation,
        metadata={
            'selection_source': 'rule_based_v1',
            'surrogate_U': s_out.U,
            'surrogate_umax': s_out.umax,
            'surrogate_sigmavm_max': s_out.sigmavm_max,
            'surrogate_Rx': s_out.Rx,
        },
    )
