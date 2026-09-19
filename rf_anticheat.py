import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)
n_samples = 1000

df = pd.DataFrame({
    'apm': np.random.randint(70,450, n_samples),
    'headshot_rate': np.random.uniform(0.05, 0.95, n_samples),
    'reaction_time_ms': np.random.uniform(120.0, 350.0, n_samples),
    'hours_played': np.random.randint(5, 3000, n_samples)
})

base = ((df['reaction_time_ms'] < 210) & (df['headshot_rate'] > 0.5) & (df['hours_played'] < 400)).astype(int)
noise = np.random.rand(n_samples) < 0.01
df['is_cheater'] = np.where(noise, 1 - base, base)

X = df.drop('is_cheater', axis=1)
y = df['is_cheater']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    class_weight='balanced',
    random_state=42
    )
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
cm = confusion_matrix(y_test, pred)
f1score = f1_score(y_test, pred)
accuracy = accuracy_score(y_test, pred)
print(f"f1 score: {f1score}")
print(f"accuracy: {accuracy}")
print(f"confusion matrix:\n{cm}")
importance = pd.Series(rf.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)
print(f"importance {importance}")
probs = rf.predict_proba(X_test)[:,1]
preds_tuned = (probs >= 0.2).astype(int)
print(f"confusion matrix:\n{confusion_matrix(y_test, preds_tuned)}")
print(f"f1 score: {f1_score(y_test, preds_tuned)}")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5]
}
grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='f1'
)
grid.fit(X_train, y_train)
best_rf = grid.best_estimator_
best_params = grid.best_params_

print("Лучшие параметры леса:", best_params)
print("Лучший F1 на кросс-валидации:", grid.best_score_)