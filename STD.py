import pandas as pd
import numpy as np

NUM_STUDENTS = 80000
np.random.seed(32)

students = pd.DataFrame({
    'student_id': np.arange(1, NUM_STUDENTS + 1),

    'interest': np.clip(
        np.round(np.random.normal(3.0, 1.5, NUM_STUDENTS)),
        1, 5
    ).astype(int),

    'attendance': np.clip(
        np.round(np.random.normal(80, 15, NUM_STUDENTS)),
        0, 100
    ),
})

study_hours_base = np.clip(
    np.random.weibull(2.0, NUM_STUDENTS) * 8,
    4, 40
)
students['study_hours'] = np.clip(
    np.round(
        study_hours_base +
        students['interest'] * 1.8 +
        np.random.choice([-3, 0, 3], NUM_STUDENTS, p=[0.25, 0.5, 0.25])
    ),
    4, 40
).astype(int)

subjects = ['math', 'physics', 'literature', 'biology', 'chemistry']
for subj in subjects:
    base_score = (
            0.7 * students['study_hours'] * 0.5 +
            0.3 * students['interest'] * 15 +
            0.1 * students['attendance'] * 0.8 +
            np.random.normal(0, 8, NUM_STUDENTS)
    )
    students[f'grade_{subj}'] = np.clip(
        np.round(base_score),
        20, 100
    ).astype(int)

students['stress_level'] = np.clip(
    np.round(np.random.normal(3.0, 1.0, NUM_STUDENTS)),
    1, 5
).astype(int)

students['stress_label'] = students['stress_level'].map({
    1: 'very_low',
    2: 'low',
    3: 'medium',
    4: 'high',
    5: 'very_high'
})

students['interest_label'] = students['interest'].map({
    1: 'very_low',
    2: 'low',
    3: 'medium',
    4: 'high',
    5: 'very_high'
})

students.to_csv('Dena.csv', index=False)
print("Датасет успешно сгенерирован!")
print(f"Размер датасета: {students.shape}")
print("\nПример данных:")
print(students.head(3))
print("\nПроверка распределений:")
print("Интерес:")
print(students['interest'].value_counts().sort_index())
print("\nСтресс:")
print(students['stress_level'].value_counts().sort_index())
print("\nСреднее время учебы (ч/нед):", students['study_hours'].mean())
print("Средняя оценка:", students.filter(regex='grade_').mean().mean())