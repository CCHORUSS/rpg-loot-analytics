import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

np.random.seed(42)
n_samples = 2000

df = pd.DataFrame({
    'apm': np.random.randint(70, 150, n_samples),
    'headshot_rate': np.random.uniform(0.1, 1, n_samples),
    'reaction_time_ms': np.random.randint(100, 200, n_samples),
    'hours_played': np.random.randint(10, 1000, n_samples),
})
base = (df['apm'] > 70) & (df['headshot_rate'] > 0.7) & (df['reaction_time_ms'] < 100) & (df['hours_played'] < 600)
noise = np.random.rand(n_samples) < 0.015
df['is_cheater'] = np.where(noise, 1 - base, base)
X = df.drop('is_cheater', axis=1)
y = df['is_cheater']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
num_neg = (y_train == 0).sum()
num_pos = (y_train == 1).sum()
scale_weigth = num_neg / num_pos
model = XGBClassifier()
model.fit(X_train, y_train)
model = XGBClassifier(
    n_estimators=1000,
    max_depth=4,
    learning_rate=0.05,
    scale_pos_weight=scale_weigth,
    random_state=42,
    eval_metric='logloss',
    early_stopping_rounds=10
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
print(model.best_iteration)
pred = model.predict_proba(X_test)[:, 1]
preds = (pred >= 0.2).astype(int)
print(classification_report(y_test, preds))
print(confusion_matrix(y_test, preds))
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Feature Importance:")
print(importance)
