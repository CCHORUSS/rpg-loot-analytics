import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from antic import base_condition


np.random.seed(42)
n_samples = 1000
df = pd.DataFrame({
    'apm': np.random.randint(80, 401, n_samples),
'ping_variance':np.random.uniform(1.0, 50.0, n_samples),
})
base_condition = (df['apm'] > 340) & (df['ping_variance'] < 5.0)
noise = np.random.rand(n_samples) < 0.015
df['is_cheater'] = np.where(noise, 1 - base_condition.astype(int), base_condition.astype(int))
X = df.drop('is_cheater', axis=1)
y = df['is_cheater']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(class_weight='balanced', random_state=42)),
])

pipe.fit(X_train, y_train)
probs = pipe.predict_proba(X_test)[:, 1]
preds_soft = (probs >= 0.3).astype(int)
preds_hard = (probs >= 0.7).astype(int)

print(confusion_matrix(y_test, preds_soft))
print(confusion_matrix(y_test, preds_hard))