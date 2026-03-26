from __future__ import annotations

from typing import List

from app.schemas.response import (
    NutritionPlan,
    ProgressionRule,
    Questionnaire,
    RecommendationGenerateRequest,
    RecommendationPlan,
    RecommendationResponse,
    SurrogateInterpretation,
    SurrogateOutput,
    WorkoutDay,
    WorkoutExercise,
)


class RecommendationError(RuntimeError):
    pass


# ---------- Internal helpers ----------

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))



def _stimulus_score(s: SurrogateInterpretation) -> int:
    return int(getattr(s, 'stimulus_score', getattr(s, 'load_score', 50)))



def _risk_score(s: SurrogateInterpretation) -> int:
    if hasattr(s, 'risk_score'):
        return int(getattr(s, 'risk_score'))
    return {'low': 30, 'moderate': 55, 'high': 80}.get(getattr(s, 'level', 'moderate'), 55)



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



def _difficulty_label(q: Questionnaire, s: SurrogateInterpretation) -> str:
    stim = _stimulus_score(s)
    risk = _risk_score(s)
    exp = q.experience_level.lower()

    if exp == 'advanced' and risk < 45 and stim >= 60:
        return 'advanced'
    if exp in {'intermediate', 'advanced'} and risk < 70:
        return 'intermediate'
    return 'beginner'



def _training_frequency(q: Questionnaire, s: SurrogateInterpretation) -> int:
    stim = _stimulus_score(s)
    risk = _risk_score(s)
    base = {
        'beginner': 3,
        'intermediate': 4,
        'advanced': 5,
    }.get(q.experience_level.lower(), 3)

    if risk >= 85:
        return max(2, base - 2)
    if risk >= 70:
        return max(2, base - 1)
    if stim < 35 and risk < 40 and base < 5:
        return base + 1
    return base



def _volume_modifier(s: SurrogateInterpretation) -> float:
    stim = _stimulus_score(s) / 100.0
    risk = _risk_score(s) / 100.0
    value = 0.90 + 0.35 * (1.0 - risk) + 0.25 * (0.50 - stim)
    return _clamp(value, 0.75, 1.25)



def _estimated_tdee(q: Questionnaire) -> int:
    sex_bonus = 5 if q.gender.lower() == 'male' else -161
    bmr = 10 * q.weight_kg + 6.25 * q.height_cm - 5 * q.age + sex_bonus
    return int(round(bmr * _activity_factor(q.experience_level)))



def _build_nutrition(q: Questionnaire, s: SurrogateInterpretation) -> NutritionPlan:
    stim = _stimulus_score(s)
    risk = _risk_score(s)
    tdee = _estimated_tdee(q)
    kcal = tdee + _goal_kcal_adjustment(q.goal)

    if risk >= 75 and q.goal == 'fat_loss':
        kcal = max(kcal, tdee - 250)
    if stim < 35 and risk < 45 and q.goal in {'muscle_gain', 'recomposition'}:
        kcal += 100

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
    if stim < 35 and risk < 45:
        carbs += 20
        kcal += 80
    water_ml = int(max(1800, q.weight_kg * 30 + 500 + (250 if risk >= 75 else 0)))

    notes = [
        'Распределять белок на 3-5 приёмов пищи в течение дня.',
        'Основную часть углеводов переносить на периоды до и после тренировки.',
        'Контролировать изменение массы тела 1 раз в неделю в одинаковых условиях.',
    ]
    if q.goal == 'fat_loss':
        notes.append('Поддерживать умеренный дефицит калорий без резкого снижения потребления белка.')
    if risk >= 75:
        notes.append('Не сочетать повышенный тренировочный стресс с агрессивным снижением калорийности.')
    if stim < 35 and risk < 45:
        notes.append('Допускается немного более высокий приём углеводов вокруг тренировки для усиления тренировочного стимула.')

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



def _plan_full_body_intro(volume_modifier: float) -> List[WorkoutDay]:
    sets_main = 2 if volume_modifier < 0.95 else 3
    return [
        WorkoutDay(
            day_label='День 1',
            focus='Базовое full-body занятие',
            exercises=[
                _ex('Приседание с собственным весом', sets_main, '10-12', 75),
                _ex('Ягодичный мост', sets_main, '12-15', 60),
                _ex('Тяга резиновой ленты к поясу', sets_main, '10-12', 75),
                _ex('Отжимания от опоры', sets_main, '8-10', 90),
                _ex('Планка', 2, '20-30 сек', 45, notes='Сохранять нейтральное положение корпуса.'),
            ],
        ),
        WorkoutDay(
            day_label='День 2',
            focus='Техника и устойчивость',
            exercises=[
                _ex('Выпад назад без веса', sets_main, '8-10 на каждую ногу', 75),
                _ex('Румынская тяга с лёгким весом', sets_main, '10-12', 75),
                _ex('Тяга верхнего блока / резины', sets_main, '10-12', 75),
                _ex('Жим гантелей лёжа лёгкий', sets_main, '10-12', 90),
                _ex('Dead bug', 2, '8-10 на сторону', 45),
            ],
        ),
        WorkoutDay(
            day_label='День 3',
            focus='Контроль движения и умеренный объём',
            exercises=[
                _ex('Гоблет-присед лёгкий', sets_main, '10-12', 75),
                _ex('Шаги на платформу', sets_main, '10 на каждую ногу', 60),
                _ex('Горизонтальная тяга', sets_main, '10-12', 75),
                _ex('Жим гантелей сидя', sets_main, '8-10', 90),
                _ex('Боковая планка', 2, '20 сек на сторону', 45),
            ],
        ),
    ]



def _plan_recomp_base(volume_modifier: float) -> List[WorkoutDay]:
    sets_main = 3 if volume_modifier <= 1.0 else 4
    return [
        WorkoutDay(
            day_label='День 1',
            focus='Низ тела',
            exercises=[
                _ex('Приседание', sets_main, '8-10', 90),
                _ex('Румынская тяга', sets_main, '8-10', 90),
                _ex('Выпады', 3, '10 на ногу', 75),
                _ex('Подъёмы на носки', 3, '12-15', 45),
                _ex('Планка', 3, '30-40 сек', 45),
            ],
        ),
        WorkoutDay(
            day_label='День 2',
            focus='Верх тела',
            exercises=[
                _ex('Жим гантелей лёжа', sets_main, '8-10', 90),
                _ex('Тяга горизонтального блока', sets_main, '10-12', 75),
                _ex('Жим вверх сидя', 3, '8-10', 90),
                _ex('Тяга к лицу', 3, '12-15', 60),
                _ex('Скручивания', 3, '12-15', 45),
            ],
        ),
        WorkoutDay(
            day_label='День 3',
            focus='Смешанная силовая работа',
            exercises=[
                _ex('Гоблет-присед', sets_main, '10-12', 75),
                _ex('Тяга гантели в наклоне', sets_main, '10-12', 75),
                _ex('Отжимания', 3, '8-12', 75),
                _ex('Ягодичный мост', 3, '12-15', 60),
                _ex('Pallof press', 3, '10-12', 45),
            ],
        ),
        WorkoutDay(
            day_label='День 4',
            focus='Кардио + мобильность',
            exercises=[
                _ex('Ходьба / велотренажёр', 1, '25-35 мин', 0, notes='Интенсивность: зона 2.'),
                _ex('Мобилизация тазобедренных суставов', 2, '6-8 мин', 0),
                _ex('Мобилизация грудного отдела', 2, '6-8 мин', 0),
            ],
        ),
    ]



def _plan_muscle_gain_upper_lower(volume_modifier: float) -> List[WorkoutDay]:
    sets_main = 4 if volume_modifier >= 1.0 else 3
    return [
        WorkoutDay(
            day_label='День 1',
            focus='Низ тела А',
            exercises=[
                _ex('Приседание', sets_main, '6-8', 120),
                _ex('Румынская тяга', sets_main, '8-10', 120),
                _ex('Болгарские выпады', 3, '8-10 на ногу', 90),
                _ex('Сгибание ног', 3, '10-12', 60),
                _ex('Подъёмы на носки', 4, '12-15', 45),
            ],
        ),
        WorkoutDay(
            day_label='День 2',
            focus='Верх тела А',
            exercises=[
                _ex('Жим лёжа', sets_main, '6-8', 120),
                _ex('Тяга штанги / блока', sets_main, '8-10', 90),
                _ex('Жим сидя', 3, '8-10', 90),
                _ex('Тяга верхнего блока', 3, '10-12', 75),
                _ex('Сгибание на бицепс + разгибание на трицепс', 3, '10-12', 60),
            ],
        ),
        WorkoutDay(
            day_label='День 3',
            focus='Низ тела B',
            exercises=[
                _ex('Тяга классическая / трап-гриф', sets_main, '5-6', 150),
                _ex('Жим ногами', 3, '10-12', 90),
                _ex('Ягодичный мост / hip thrust', 3, '8-10', 90),
                _ex('Разгибание ног', 3, '12-15', 60),
                _ex('Антиротация корпуса', 3, '10-12', 45),
            ],
        ),
        WorkoutDay(
            day_label='День 4',
            focus='Верх тела B',
            exercises=[
                _ex('Жим гантелей под углом', sets_main, '8-10', 90),
                _ex('Горизонтальная тяга', sets_main, '8-10', 90),
                _ex('Разведения в стороны', 3, '12-15', 60),
                _ex('Тяга к лицу', 3, '12-15', 60),
                _ex('Пресс', 3, '12-15', 45),
            ],
        ),
    ]



def _plan_strength_advanced(_: float) -> List[WorkoutDay]:
    return [
        WorkoutDay(
            day_label='День 1',
            focus='Силовой присед',
            exercises=[
                _ex('Приседание', 5, '4-6', 150),
                _ex('Пауза-присед', 3, '3-4', 150),
                _ex('Румынская тяга', 4, '6-8', 120),
                _ex('Кор', 3, '30-40 сек', 45),
            ],
        ),
        WorkoutDay(
            day_label='День 2',
            focus='Силовой жим',
            exercises=[
                _ex('Жим лёжа', 5, '4-6', 150),
                _ex('Жим узким хватом', 3, '6-8', 120),
                _ex('Тяга штанги', 4, '6-8', 120),
                _ex('Задняя дельта', 3, '12-15', 60),
            ],
        ),
        WorkoutDay(
            day_label='День 3',
            focus='Силовая тяга',
            exercises=[
                _ex('Становая тяга', 5, '3-5', 180),
                _ex('Фронтальный присед', 3, '5-6', 150),
                _ex('Подтягивания / тяга верхнего блока', 4, '6-8', 90),
                _ex('Антиэкстензия корпуса', 3, '10-12', 45),
            ],
        ),
        WorkoutDay(
            day_label='День 4',
            focus='Объёмный верх',
            exercises=[
                _ex('Жим гантелей', 4, '8-10', 90),
                _ex('Тяга сидя', 4, '8-10', 90),
                _ex('Жим вверх', 3, '8-10', 90),
                _ex('Локальная работа на руки', 3, '10-12', 60),
            ],
        ),
        WorkoutDay(
            day_label='День 5',
            focus='Восстановительное кондиционирование',
            exercises=[
                _ex('Кардио низкой интенсивности', 1, '20-30 мин', 0),
                _ex('Мобильность', 1, '10-15 мин', 0),
            ],
        ),
    ]



def _apply_risk_adjustments(days: List[WorkoutDay], s: SurrogateInterpretation) -> List[WorkoutDay]:
    risk = _risk_score(s)
    if risk < 75:
        return days

    adjusted: List[WorkoutDay] = []
    for day in days:
        exercises = []
        for ex in day.exercises:
            name = ex.name
            notes = ex.notes
            if 'Становая тяга' in name:
                name = 'Тяга трап-грифа / румынская тяга умеренная'
                notes = (notes + ' Избегать предельных весов.').strip()
            elif 'Приседание' in name and 'с собственным весом' not in name:
                name = 'Гоблет-присед / жим ногами'
                notes = (notes + ' Предпочесть контролируемую амплитуду.').strip()
            elif 'Прыж' in name or 'плио' in name.lower():
                name = 'Кардио низкой ударной нагрузки'
                notes = (notes + ' Исключить баллистический компонент.').strip()

            sets = max(1, ex.sets - 1) if ex.sets >= 3 else ex.sets
            rest = max(ex.rest_sec, 60) if ex.rest_sec > 0 else 0
            exercises.append(WorkoutExercise(name=name, sets=sets, reps=ex.reps, rest_sec=rest, tempo=ex.tempo, notes=notes))

        focus = day.focus if 'щадящий режим' in day.focus.lower() else f'{day.focus} (щадящий режим)'
        adjusted.append(WorkoutDay(day_label=day.day_label, focus=focus, exercises=exercises))
    return adjusted



def _build_progression_rules(s: SurrogateInterpretation) -> List[ProgressionRule]:
    stim = _stimulus_score(s)
    risk = _risk_score(s)

    if risk >= 75:
        return [
            ProgressionRule(
                week='Ближайшие 1-2 недели',
                condition='Техника нестабильна, нагрузка воспринимается тяжело либо сохраняется выраженная утомлённость.',
                action='Снизить объём на 25-35%, не повышать рабочий вес и использовать запас 3-4 повторения до отказа.',
                rationale='Высокий risk_score указывает на необходимость ограничить рост нагрузки и стабилизировать переносимость.',
            ),
            ProgressionRule(
                week='После стабилизации техники',
                condition='Все упражнения выполняются без боли и с устойчивой техникой в течение минимум 2 тренировок подряд.',
                action='Повысить нагрузку только по одному параметру: либо +2.5% к весу, либо +1 подход в одном базовом упражнении.',
                rationale='При высоком риске одновременное увеличение объёма и интенсивности нежелательно.',
            ),
            ProgressionRule(
                week='Каждая 4-я неделя',
                condition='Даже при отсутствии жалоб план выполнялся без пропусков.',
                action='Провести разгрузочную неделю с уменьшением общего объёма на 30-40%.',
                rationale='Периодическая разгрузка снижает вероятность накопления перегрузки.',
            ),
        ]

    if stim < 45 and risk < 45:
        return [
            ProgressionRule(
                week='Недели 1-2',
                condition='Подходы выполнены с хорошей техникой и субъективной нагрузкой не выше RPE 7.',
                action='Добавить 1 подход в 1-2 основных упражнениях или увеличить рабочий вес на 2.5-5%.',
                rationale='Низкий stimulus_score при низком risk_score означает, что стимул можно безопасно усилить.',
            ),
            ProgressionRule(
                week='Недели 3-4',
                condition='Темп выполнения контролируется, восстановление остаётся стабильным.',
                action='Сохранить новую нагрузку 1 неделю, затем повторно повысить один параметр прогрессии.',
                rationale='Пошаговое усиление нагрузки позволяет увеличить тренировочный стимул без резкого роста риска.',
            ),
            ProgressionRule(
                week='Каждая 5-я неделя',
                condition='Нагрузка последовательно нарастала без ухудшения самочувствия.',
                action='Снизить объём на 20-25% на одну неделю для восстановления.',
                rationale='Короткая разгрузка помогает сохранить долгосрочную адаптацию.',
            ),
        ]

    return [
        ProgressionRule(
            week='Недели 1-2',
            condition='Все подходы выполнены с сохранением техники и субъективной нагрузкой не выше RPE 7-8.',
            action='Сначала удерживать текущую схему 1 неделю, затем увеличить рабочий вес на 2.5-5% или добавить 1-2 повторения.',
            rationale='Средняя зона stimulus_score требует умеренной, а не агрессивной прогрессии.',
        ),
        ProgressionRule(
            week='Любая неделя',
            condition='Появляется выраженное ухудшение техники, боль или устойчивое ощущение чрезмерной утомлённости.',
            action='Снизить объём на 20-30% на 1 неделю и вернуться к предыдущему уровню нагрузки.',
            rationale='Даже при приемлемом risk_score временное снижение нагрузки уменьшает вероятность накопления утомления.',
        ),
        ProgressionRule(
            week='Каждая 4-я неделя',
            condition='План выполнялся полностью без пропусков и без негативных симптомов.',
            action='Провести разгрузочную неделю: сохранить упражнения, но уменьшить объём на 25-35%.',
            rationale='Разгрузка улучшает восстановление и позволяет сохранять долгосрочную адаптацию.',
        ),
    ]



def _build_safety_notes(s: SurrogateInterpretation) -> List[str]:
    stim = _stimulus_score(s)
    risk = _risk_score(s)
    notes = [
        'Перед основной частью занятия выполнять разминку 7-10 минут.',
        'Прекращать упражнение при возникновении острой боли.',
        'Поддерживать нейтральное положение корпуса и контролировать скорость эксцентрической фазы.',
    ]
    if risk >= 75:
        notes.append('При высоком risk_score избегать одновременного увеличения тренировочного объёма и интенсивности.')
        notes.append('Предпочитать контролируемую технику, умеренную амплитуду и отсутствие баллистических движений.')
    elif stim < 45 and risk < 45:
        notes.append('Низкий risk_score допускает осторожное повышение объёма, но только при сохранении качества техники.')
    else:
        notes.append('Повышать рабочий вес только после двух последовательных тренировок с устойчивой техникой.')
    return notes



def _select_template(q: Questionnaire, s_out: SurrogateOutput, s: SurrogateInterpretation) -> RecommendationPlan:
    stim = _stimulus_score(s)
    risk = _risk_score(s)
    volume_modifier = _volume_modifier(s)
    difficulty = _difficulty_label(q, s)
    frequency = _training_frequency(q, s)
    nutrition = _build_nutrition(q, s)

    goal = q.goal.lower()
    exp = q.experience_level.lower()

    if goal == 'fat_loss':
        if risk >= 75:
            title = 'Снижение жировой массы: щадящий адаптивный план'
            summary = 'Умеренный дефицит калорий и силовые сессии с ограниченным объёмом при повышенном риске перегрузки.'
            workout_days = _plan_full_body_intro(volume_modifier)
            plan_id = 'fat_loss_gentle'
        else:
            title = 'Снижение жировой массы: базовый адаптивный план'
            summary = 'Комбинация 3-4 силовых сессий с умеренным объёмом и одной восстановительной кардио-сессией.'
            workout_days = _plan_recomp_base(volume_modifier)
            plan_id = 'fat_loss_base'
    elif goal == 'maintenance':
        if risk >= 75:
            title = 'Поддержание формы: стабилизирующий щадящий план'
            summary = 'Поддержание двигательного контроля и силовых качеств без агрессивной прогрессии нагрузки.'
            workout_days = _plan_full_body_intro(volume_modifier)
            plan_id = 'maintenance_gentle'
        else:
            title = 'Поддержание формы: стабилизирующий план'
            summary = 'Поддержание силовых качеств и двигательного контроля без избыточной нагрузки.'
            workout_days = _plan_recomp_base(volume_modifier)
            plan_id = 'maintenance_base'
    elif goal == 'recomposition':
        title = 'Рекомпозиция: смешанный силовой план'
        summary = 'Приоритет сохранения и постепенного роста силовых показателей при контроле суммарной утомляемости.'
        workout_days = _plan_recomp_base(volume_modifier if risk < 75 else min(volume_modifier, 0.95))
        plan_id = 'recomp_base'
    else:
        if risk >= 75:
            title = 'Набор мышечной массы: щадящий full-body план'
            summary = 'Наращивание тренировочного стимула при ограничении риска перегрузки и более медленной прогрессии.'
            workout_days = _plan_full_body_intro(volume_modifier)
            plan_id = 'muscle_gain_gentle'
            frequency = min(frequency, 3)
        elif exp == 'advanced' and risk < 40 and stim >= 60:
            title = 'Набор мышечной массы: силовой продвинутый план'
            summary = 'Достаточный стимул и контролируемый риск позволяют использовать силовой сплит с пошаговой прогрессией.'
            workout_days = _plan_strength_advanced(volume_modifier)
            plan_id = 'muscle_gain_advanced'
            frequency = 5
        elif exp == 'beginner' and stim < 60:
            title = 'Набор мышечной массы: вводный full-body план'
            summary = 'Формирование техники базовых движений и постепенное увеличение рабочего объёма.'
            workout_days = _plan_full_body_intro(volume_modifier)
            plan_id = 'muscle_gain_intro'
            frequency = min(frequency, 3)
        else:
            title = 'Набор мышечной массы: upper/lower план'
            summary = 'Сбалансированный сплит для гипертрофии, адаптированный по стимулу и риску перегрузки.'
            workout_days = _plan_muscle_gain_upper_lower(volume_modifier)
            plan_id = 'muscle_gain_ul'
            frequency = min(max(frequency, 4), 4)

    workout_days = _apply_risk_adjustments(workout_days, s)
    workout_days = workout_days[:frequency]

    summary = (
        f'{summary} Текущая логика учитывает stimulus_score={stim} и risk_score={risk}, '
        f'полученные из предсказаний U, sigmavm_max, umax и Rx.'
    )

    return RecommendationPlan(
        plan_id=plan_id,
        title=title,
        summary=summary,
        weekly_frequency=frequency,
        difficulty=difficulty,
        workout_days=workout_days,
        nutrition=nutrition,
        progression_rules=_build_progression_rules(s),
        safety_notes=_build_safety_notes(s),
    )


# ---------- Public API ----------

def generate_recommendation(payload: RecommendationGenerateRequest):
    q = payload.questionnaire
    s_out = payload.surrogate_prediction
    s_interp = payload.surrogate_interpretation

    if s_out.U < 0 or s_out.umax < 0 or s_out.sigmavm_max < 0:
        raise RecommendationError('Получены некорректные значения surrogate-предсказания.')

    plan = _select_template(q, s_out, s_interp)
    stim = _stimulus_score(s_interp)
    risk = _risk_score(s_interp)

    explanation = [
        f'Цель пользователя: {q.goal}.',
        f'Уровень подготовки: {q.experience_level}.',
        f'Суррогатная модель предсказала U={s_out.U:.4g}, sigmavm_max={s_out.sigmavm_max:.4g}, umax={s_out.umax:.4g}, Rx={s_out.Rx:.4g}.',
        f'Из этих величин сформированы stimulus_score={stim} и risk_score={risk}.',
        'stimulus_score управляет требуемым объёмом стимула, а risk_score ограничивает частоту, интенсивность и темп прогрессии.',
        'Итоговый шаблон выбран по сочетанию цели, уровня подготовки и двух индексов суррогатной интерпретации.',
    ]

    return RecommendationResponse(
        status='ok',
        plan=plan,
        explanation=explanation,
        metadata={
            'selection_source': 'rule_based_v2_surrogate_scores',
            'stimulus_score': stim,
            'risk_score': risk,
            'surrogate_U': s_out.U,
            'surrogate_umax': s_out.umax,
            'surrogate_sigmavm_max': s_out.sigmavm_max,
            'surrogate_Rx': s_out.Rx,
        },
    )
