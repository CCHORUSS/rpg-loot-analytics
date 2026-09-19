"""
MACHINE LEARNING PIPELINE CHEAT SHEET
=====================================
"""

import numpy as np
import pandas as pd
import joblib

# Model Selection & Validation
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

# Preprocessing
from sklearn.preprocessing import StandardScaler

# Algorithms (Estimators)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Evaluation Metrics
from sklearn.metrics import (
    mean_absolute_error,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ==============================================================================
# 1. CATEGORICAL ENCODING (One-Hot Encoding)
# ==============================================================================
# WHEN: В датасете есть категориальные текстовые признаки ('Warrior', 'Mage') без естественного порядка.
# WHY: Алгоритмы работают только с числами. Предотвращает ложную иерархию (чтобы 2 не считалось больше 1).
# CODE:
# df = pd.get_dummies(df, columns=['class'])


# ==============================================================================
# 2. FEATURE MATRIX (X) & TARGET VECTOR (y) SPLIT
# ==============================================================================
# WHEN: Всегда перед обучением модели.
# WHY: Защита от Data Leakage (утечки целевого признака в обучающие данные).
# CODE:
# X = df.drop('damage', axis=1)  # Features: все столбцы, кроме таргета
# y = df['damage']               # Target: исключительно целевая переменная


# ==============================================================================
# 3. TASK & ALGORITHM SELECTION (Regression vs Classification)
# ==============================================================================
# Regression:     Предсказание непрерывной числовой величины (Continuous value: урон, цена, HP).
# Classification: Предсказание дискретной категории/метки (Discrete label: бинарная 0/1 или мультикласс).

# Baseline Models:
# - LinearRegression():    Линейная зависимость для непрерывных чисел.
# - LogisticRegression():  Базовая сигмоидальная функция для вероятностей классов (0/1).
# - KNeighborsClassifier(): Instance-based алгоритм по дистанциям k ближайших соседей.

# Ensemble Methods (Random Forest):
# - RandomForestRegressor():  Ансамбль решающих деревьев с усреднением для регрессии.
# - RandomForestClassifier(): Ансамбль деревьев с мажоритарным голосованием за класс.


# ==============================================================================
# 4. TRAIN / TEST SPLIT (Holdout Validation)
# ==============================================================================
# WHEN: Базовый этап оценки обобщающей способности (Generalization).
# WHY: Оценка качества на отложенных данных (Unseen data), которые модель не запоминала.
# CODE:
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ==============================================================================
# 5. FEATURE SCALING (StandardScaler)
# ==============================================================================
# WHEN: Использование Distance-based моделей (KNN, SVM) или Gradient Descent моделей (Logistic Regression, Neural Nets).
# NOTE: Для Tree-based моделей (RandomForest, DecisionTree) масштабирование НЕ требуется.
# WHY: Предотвращает доминирование признаков с большим размахом значений над признаками с малым размахом.
# CODE:
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)  # Вычисление mean/std и масштабирование трейна
# X_test_scaled  = scaler.transform(X_test)       # Только масштабирование теста (защита от Data Leakage)


# ==============================================================================
# 6. MODEL TRAINING & INFERENCE PIPELINE
# ==============================================================================
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)       # Training / Fitting (обучение весов)
# predictions = model.predict(X_test)  # Inference / Prediction (получение предсказаний)


# ==============================================================================
# 7. REGRESSION METRICS
# ==============================================================================
# - MAE (Mean Absolute Error): Средняя абсолютная ошибка в базовых единицах измерения.
# mae = mean_absolute_error(y_test, predictions)


# ==============================================================================
# 8. CLASSIFICATION METRICS & CONFUSION MATRIX
# ==============================================================================
# WHEN: Оценка качества дискретных предсказаний, особенно при Class Imbalance (несбалансированных классах).

# Metrics:
# - Precision: Доля истинных срабатываний среди всех объектов, которым модель присвоила класс 1.
# - Recall: Доля найденных объектов класса 1 от их реального суммарного количества в выборке.
# - F1-score: Гармоническое среднее между Precision и Recall.
# prec = precision_score(y_test, predictions)
# rec  = recall_score(y_test, predictions)
# f1   = f1_score(y_test, predictions)

# Confusion Matrix Layout:
#                 Predicted 0         Predicted 1
# Actual 0  [  True Negative (TN) , False Positive (FP) ]  <- FP: Type I Error (ложная тревога)
# Actual 1  [ False Negative (FN) ,  True Positive (TP) ]  <- FN: Type II Error (пропуск события)
# cm = confusion_matrix(y_test, predictions)
# tn, fp, fn, tp = cm.ravel()


# ==============================================================================
# 9. CROSS-VALIDATION
# ==============================================================================
# WHEN: Финальная валидация стабильности модели без привязки к удачному разбиению.
# WHY: Минимизирует дисперсию оценки за счет ротации фолдов.
# CODE:
# cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
# mean_mae = (-cv_scores).mean()


# ==============================================================================
# 10. FEATURE IMPORTANCE ANALYSIS
# ==============================================================================
# WHEN: Feature Selection (отбор признаков) и интерпретация работы ансамбля.
# WHY: Удаление шумовых/неинформативных колонок и проверка модели на артефакты.
# CODE:
# importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)


# ==============================================================================
# 11. HYPERPARAMETER TUNING (GridSearchCV)
# ==============================================================================
# WHEN: Оптимизация гиперпараметров для предотвращения Overfitting (переобучения) и Underfitting (недообучения).

# --- TREE HYPERPARAMETERS (Сетка перебора в param_grid) ---
# - n_estimators: Число базовых деревьев в ансамбле (больше = стабильнее, но дольше compute).
# - max_depth: Ограничение глубины ветвления (None = рост до упора; фиксация числа спасает от шума).
# - min_samples_split: Минимальное число сэмплов в узле для совершения следующего сплита.

# --- SEARCH CONFIGURATION ---
# - cv=5: K-Fold кросс-валидация на 5 фолдов для каждой уникальной комбинации параметров.
# - scoring: Целевая функция оптимизации (например, 'neg_mean_absolute_error').
# - n_jobs=-1: Параллелизация вычислений на все доступные ядра процессора.

# CODE:
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [3, 5, 10, None],
#     'min_samples_split': [2, 5, 10]
# }
# grid_search = GridSearchCV(
#     estimator=RandomForestRegressor(random_state=42),
#     param_grid=param_grid,
#     cv=5,
#     scoring='neg_mean_absolute_error',
#     n_jobs=-1
# )
# grid_search.fit(X_train, y_train)

# --- POST-TUNING ATTRIBUTES ---
# best_params = grid_search.best_params_       # Словарь лучших гиперпараметров
# best_model  = grid_search.best_estimator_    # Готовый Estimator с лучшими весами


# ==============================================================================
# 12. MODEL PERSISTENCE & INFERENCE DEPLOYMENT (joblib)
# ==============================================================================
# WHEN: Передача обученной модели в Production (бэкенд, скрипты, сервисы).
# WHY: Моментальная загрузка весов без необходимости повторного обучения (Fitting).

# Serialization (Сохранение):
# joblib.dump(best_model, 'model.pkl')

# Deserialization (Загрузка в отдельном скрипте):
# loaded_model = joblib.load('model.pkl')
# single_inference = loaded_model.predict(new_unseen_data)
# ==============================================================================
# ==============================================================================
# 13. FEATURE SCALING: StandardScaler & KNN (K-Nearest Neighbors)
# ==============================================================================
# KEY CONCEPT:
# - Модели на расстояниях (KNN, SVM) и градиентных спусках (LogisticRegression, Нейросети)
#   ломаются, если одни признаки в сотни раз больше других.
# - Древесным моделям (RandomForest, DecisionTree) масштабирование НЕ нужно.
# - ВАЖНО: У KNN НЕТ атрибута feature_importances_ (он не строит деревья).

# TERMINOLOGY:
# - Feature Scaling: Приведение признаков с разными диапазонами к единому масштабу.
# - Standardization (Z-Score): Сдвиг среднего к 0 и приведение разброса (std) к 1.
# - Data Leakage: Ошибка, когда параметры будущего теста просачиваются в обучение.
# - K-Nearest Neighbors (KNN): Классификатор, голосующий большинством k ближайших точек.
# - fit_transform(): Считает параметры (среднее/отклонение) и сразу трансформирует данные.
# - transform(): Трансформирует данные по уже посчитанным ранее параметрам.

# - from sklearn.preprocessing import StandardScaler
# - from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Шаг 1: Сначала разбиваем данные (КРИТИЧНО для защиты от Data Leakage)
# - X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Шаг 2: Создаем объект скейлера
# - scaler = StandardScaler()

# Шаг 3: Считаем среднее и разброс ТОЛЬКО по трейну и масштабируем его
# - X_train_scaled = scaler.fit_transform(X_train)

# Шаг 4: Тест масштабируем ТОЛЬКО по меркам трейна (метод .fit() здесь ЗАПРЕЩЕН)
# - X_test_scaled = scaler.transform(X_test)

# Шаг 5: Обучаем модель, чувствительную к масштабу
# - knn_model = KNeighborsClassifier(n_neighbors=5)
# - knn_model.fit(X_train_scaled, y_train)

# Шаг 6: Получаем точно такие же предсказания (классы 0 или 1)
# - predictions = knn_model.predict(X_test_scaled)

# Шаг 7: Оценка качества (работает точно так же, как в RandomForest)
# print(confusion_matrix(y_test, predictions))
# print(classification_report(y_test, predictions))