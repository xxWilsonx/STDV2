import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
import pandas as pd

def setup_fuzzy_corrector():
    stress = ctrl.Antecedent(np.arange(1, 6, 0.1), 'stress')
    interest = ctrl.Antecedent(np.arange(1, 6, 0.1), 'interest')
    time_avail = ctrl.Antecedent(np.arange(0, 41, 1), 'time_avail')
    grade_variance = ctrl.Antecedent(np.linspace(0, 100, 101), 'grade_variance')
    min_grade = ctrl.Antecedent(np.arange(20, 101, 1), 'min_grade')
    attendance = ctrl.Antecedent(np.arange(0, 101, 1), 'attendance')

    correction = ctrl.Consequent(np.arange(-2, 2.1, 0.1), 'correction')

    stress['very_low'] = fuzz.gaussmf(stress.universe, 1.5, 0.8)
    stress['low'] = fuzz.gaussmf(stress.universe, 2.5, 0.8)
    stress['medium'] = fuzz.gaussmf(stress.universe, 3.5, 0.8)
    stress['high'] = fuzz.gaussmf(stress.universe, 4.5, 0.8)
    stress['very_high'] = fuzz.gaussmf(stress.universe, 5.0, 0.8)

    interest['very_low'] = fuzz.gaussmf(interest.universe, 1.0, 0.8)
    interest['low'] = fuzz.gaussmf(interest.universe, 2.0, 0.8)
    interest['medium'] = fuzz.gaussmf(interest.universe, 3.0, 0.8)
    interest['high'] = fuzz.gaussmf(interest.universe, 4.0, 0.8)
    interest['very_high'] = fuzz.gaussmf(interest.universe, 5.0, 0.8)

    time_avail['very_low'] = fuzz.gaussmf(time_avail.universe, 0, 5)
    time_avail['low'] = fuzz.gaussmf(time_avail.universe, 10, 5)
    time_avail['medium'] = fuzz.gaussmf(time_avail.universe, 20, 5)
    time_avail['high'] = fuzz.gaussmf(time_avail.universe, 30, 5)
    time_avail['very_high'] = fuzz.gaussmf(time_avail.universe, 40, 5)

    grade_variance['very_low'] = fuzz.gaussmf(grade_variance.universe, 0, 10)
    grade_variance['low'] = fuzz.gaussmf(grade_variance.universe, 20, 10)
    grade_variance['medium'] = fuzz.gaussmf(grade_variance.universe, 40, 10)
    grade_variance['high'] = fuzz.gaussmf(grade_variance.universe, 60, 10)
    grade_variance['very_high'] = fuzz.gaussmf(grade_variance.universe, 80, 10)

    min_grade['very_low'] = fuzz.gaussmf(min_grade.universe, 30, 10)
    min_grade['low'] = fuzz.gaussmf(min_grade.universe, 50, 10)
    min_grade['medium'] = fuzz.gaussmf(min_grade.universe, 65, 10)
    min_grade['high'] = fuzz.gaussmf(min_grade.universe, 80, 10)
    min_grade['very_high'] = fuzz.gaussmf(min_grade.universe, 95, 10)

    attendance['very_low'] = fuzz.gaussmf(attendance.universe, 30, 10)
    attendance['low'] = fuzz.gaussmf(attendance.universe, 50, 10)
    attendance['medium'] = fuzz.gaussmf(attendance.universe, 70, 10)
    attendance['high'] = fuzz.gaussmf(attendance.universe, 85, 10)
    attendance['very_high'] = fuzz.gaussmf(attendance.universe, 100, 10)

    correction['strong_reduce'] = fuzz.gaussmf(correction.universe, -1.8, 0.3)
    correction['reduce'] = fuzz.gaussmf(correction.universe, -1.0, 0.3)
    correction['slight_reduce'] = fuzz.gaussmf(correction.universe, -0.5, 0.2)
    correction['neutral'] = fuzz.gaussmf(correction.universe, 0, 0.2)
    correction['slight_increase'] = fuzz.gaussmf(correction.universe, 0.5, 0.2)
    correction['increase'] = fuzz.gaussmf(correction.universe, 1.0, 0.3)
    correction['strong_increase'] = fuzz.gaussmf(correction.universe, 1.8, 0.3)

    rules = [

        ctrl.Rule(stress['high'] | stress['very_high'], correction['slight_reduce']),
        ctrl.Rule(stress['very_high'] & (interest['low'] | interest['very_low']), correction['reduce']),

        ctrl.Rule(interest['high'] & time_avail['medium'] & stress['low'], correction['slight_increase']),
        ctrl.Rule(interest['very_high'] & time_avail['high'] & stress['low'], correction['increase']),

        ctrl.Rule(grade_variance['high'] & interest['high'], correction['slight_reduce']),
        ctrl.Rule(grade_variance['very_high'] & interest['medium'], correction['reduce']),

        ctrl.Rule(min_grade['low'] | min_grade['very_low'], correction['slight_reduce']),
        ctrl.Rule(min_grade['very_low'] & attendance['low'], correction['reduce']),

        ctrl.Rule(attendance['low'] | attendance['very_low'], correction['slight_reduce']),
        ctrl.Rule(attendance['very_low'] & min_grade['low'], correction['reduce']),

        ctrl.Rule(
            (stress['high'] | stress['very_high']) &
            (min_grade['low'] | min_grade['very_low']),
            correction['reduce']
        ),
        ctrl.Rule(
            attendance['very_high'] &
            interest['high'] &
            grade_variance['low'],
            correction['slight_increase']
        ),
        ctrl.Rule(
            interest['very_high'] &
            time_avail['very_high'] &
            stress['very_low'] &
            grade_variance['low'],
            correction['increase']
        ),
        ctrl.Rule(
            interest['very_low'] &
            attendance['low'] &
            min_grade['low'],
            correction['strong_reduce']
        ),

        ctrl.Rule(
            interest['medium'] &
            stress['medium'] &
            time_avail['medium'],
            correction['neutral']
        ),
        ctrl.Rule(
            (grade_variance['medium'] | grade_variance['low']) &
            (min_grade['medium'] | min_grade['high']) &
            attendance['medium'],
            correction['neutral']
        ),
        ctrl.Rule(
            (stress['low'] | stress['medium']) &
            (interest['medium'] | interest['high']) &
            (time_avail['medium'] | time_avail['high']),
            correction['neutral']
        )
    ]

    correction_system = ctrl.ControlSystem(rules)
    return correction_system


def apply_fuzzy_correction(points, student_data, subjects):
    try:
        stress_val = max(1.0, min(5.0, float(student_data.get('stress_level', 3.0))))
        interest_val = max(1.0, min(5.0, float(student_data.get('interest', 3.0))))
        time_val = max(0.0, min(40.0, float(student_data.get('study_hours', 20.0))))
        attendance_val = max(0.0, min(100.0, float(student_data.get('attendance', 80.0))))

        grade_list = [float(student_data.get(f'grade_{subj}', 0.0)) for subj in subjects]
        grade_var = float(np.var(grade_list)) if grade_list else 0.0
        min_grade_val = float(min(grade_list)) if grade_list else 70.0
        min_grade_val = max(20.0, min(100.0, min_grade_val))
        grade_var = max(0.0, min(100.0, grade_var))

        correction_system = setup_fuzzy_corrector()
        simulator = ctrl.ControlSystemSimulation(correction_system)

        simulator.input['stress'] = stress_val
        simulator.input['interest'] = interest_val
        simulator.input['time_avail'] = time_val
        simulator.input['grade_variance'] = grade_var
        simulator.input['min_grade'] = min_grade_val
        simulator.input['attendance'] = attendance_val

        simulator.compute()
        correction = float(simulator.output['correction'])

        corrected_points = max(0, min(10, points + correction))

        advice = []

        if stress_val >= 4.0:
            advice.append("Требуется консультация препода")
        elif stress_val >= 3.0:
            advice.append("Рекомендуется краткосрочный отдых")

        low_subjects = []
        for subj in subjects:
            grade = float(student_data.get(f'grade_{subj}', 0.0))
            if grade < 50:
                low_subjects.append(f"{subj} ({int(grade)})")

        if low_subjects:
            subjects_str = ", ".join(low_subjects)
            advice.append(f"Срочно улучшить оценки по: {subjects_str}")

        if attendance_val < 70:
            advice.append(f"Повысить посещаемость (сейчас: {int(attendance_val)}%)")

        if time_val < 15 and min_grade_val < 60:
            advice.append("Увеличить учебные часы для улучшения результатов")
        elif time_val > 35 and stress_val > 3.0:
            advice.append("Сбалансировать учебное время для снижения стресса")

        if interest_val < 2.5:
            advice.append("Исследовать причины низкой мотивации к учебе")

        if grade_var > 50:
            advice.append("Сбалансировать усилия по всем предметам")

        if not advice:
            advice.append("Ваша учебная нагрузка в порядке. Продолжайте в том же духе!")

        return corrected_points, advice, correction

    except Exception as e:
        print(f"\n⚠️ Fuzzy correction error: {str(e)}")
        print(f"Inputs: stress={stress_val}, interest={interest_val}, time={time_val}")
        print(f"grade_var={grade_var}, min_grade={min_grade_val}, attendance={attendance_val}")
        return points, ["Система рекомендаций временно недоступна"], 0.0


def calculate_points(student_data):
    academic = max(0.0, min(100.0, float(student_data['academic'])))
    interest_val = max(1.0, min(5.0, float(student_data['interest'])))
    stress_val = max(1.0, min(5.0, float(student_data['stress'])))
    time_val = max(0.0, min(40.0, float(student_data['time'])))

    academic_norm = academic / 10
    interest_norm = (interest_val - 1) * 2.5
    time_norm = time_val / 4.0
    stress_penalty = stress_val * 0.8

    points = (
                     academic_norm * 0.6 +
                     interest_norm * 0.25 +
                     time_norm * 0.15
             ) - stress_penalty

    return max(0.0, min(10.0, round(points, 1)))