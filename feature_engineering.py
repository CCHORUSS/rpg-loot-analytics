import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

from xgb_anticheat import scale_weigth

np.random.seed(42)
n_samples = 2000

df = pd.DataFrame({
    'kda': np.random.uniform(0.5, 0.8, n_samples),
    'hours_played': np.random.randint(10, 1000, n_samples),
    'click_std_ms': np.random.randint(5, 120, n_samples)
})
df['kda_per_hour'] = df['kda'] / (df['hours_played'] + 1)
df['macro_suspicion'] = 1 / (df['click_std_ms'] + 0.1)

condition = (df['macro_suspicion'] > 0.15) | (df['kda_per_hour'] > 0.05)
df['is_cheater'] = condition.astype(int)
X = df.drop('is_cheater', axis=1)
y = df['is_cheater']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss',
    scale_pos_weight=scale_weight,
)

model.fit(X_train, y_train)

importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importance)
print(f"cheaters{df['is_cheater'].sum()} from {len(df)}")