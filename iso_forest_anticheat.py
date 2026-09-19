import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


np.random.seed(42)

normal_players = pd.DataFrame({
    'apm': np.random.normal(180, 30, 1000),
    'reaction_ms': np.random.normal(240, 20, 1000)
})

cheaters = pd.DataFrame({
    'apm': [450, 520, 160],
    'reaction_ms': [110, 95, 80]
})

X = pd.concat([normal_players, cheaters], ignore_index=True)
print(len(X))
print(X.head())
print(X.tail())
iso = IsolationForest(
    n_estimators=100,
    contamination=0.005,
    random_state=42
)
iso.fit(X)

print(iso.decision_function(X))
preds = iso.predict(X)
scores = iso.decision_function(X)

X['is_outlier'] = preds
X['anomaly_score'] = scores
suspects = X[X['is_outlier'] == -1].sort_values(by='anomaly_score')
print(f"suspects: {len(suspects)}")
print(suspects)